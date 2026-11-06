# Includes modified LimiX attention layers. Copyright 2025 StableAI.
# FEAT adds Mamba2/linear mixing, numerical handling, and inference optimizations.
# See NOTICE and third_party/limix/README.md for the upstream source and license.
from __future__ import annotations

from typing import Callable, Literal, Optional
import functools

import torch
import torch.nn as nn
from torch.cuda import OutOfMemoryError
from torch.utils.checkpoint import checkpoint
from functools import partial
from torch.amp import autocast
from .mamba_layer import Mamba2Block
from .linear_attn_layer import LinearAttentionBlock
from ..utils.profiler import get_profiler




try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func, flash_attn_varlen_qkvpacked_func
    HAVE_FLASH_ATTN = True
except (ModuleNotFoundError, ImportError):
    HAVE_FLASH_ATTN = False

from typing_extensions import override

Activation = Literal['gelu']

ACTIVATION_FN: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'gelu': nn.GELU(), 
    'relu': nn.ReLU(),
}

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # [OPTIMIZATION] bf16 has the same exponent range as FP32 (8-bit exponent),
        # so RMSNorm is numerically safe in bf16. Skip FP32 conversion during inference.
        if not self.training and x.dtype == torch.bfloat16:
            return self._norm(x) * self.weight
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LayerNormMixedPrecision(nn.LayerNorm):
    """
    Mixed precision LayerNorm. During training, uses FP32 for stability.
    During inference with bf16, runs natively in bf16 (safe due to same exponent range as FP32).
    """
    def forward(self, input: torch.Tensor):
        # [OPTIMIZATION] During inference with bf16, skip FP32 conversion entirely.
        # bf16 has 8-bit exponent (same as FP32), so LayerNorm variance/mean won't overflow.
        if not self.training and input.dtype == torch.bfloat16:
            return super().forward(input)

        original_dtype = input.dtype
        is_large = input.numel() > 5_000_000 # Heuristic: Chunk if > 5M elements (~10MB fp16)

        if input.dtype != torch.float32 and is_large:
            # Chunking strategy
            # Flatten to [N, D] for easier chunking or keep spatial dims and chunk dim 1 (Sequence)
            # Assuming input is [B, S, ...] or [N, D]
            # We trust standard LN behavior on dimensions.
            
            # Simple approach: Check if we can chunk along dimension 0 or 1
            # Usually input is [B, Sequence, Channels] or [Batch*Sequence, Channels]
            
            # Recursive chunking helper
            def apply_ln_chunked(x_tensor):
                # Chunk along the first dimension (Batch/Seq usually)
                chunk_size = 4096
                dim_0 = x_tensor.shape[0]
                out_chunks = []
                
                for i in range(0, dim_0, chunk_size):
                    end = min(i + chunk_size, dim_0)
                    chunk = x_tensor[i:end].float() # Cast small chunk to FP32
                    
                    # Run LN (PyTorch LN can handle 3D inputs if normalized_shape matches last dims)
                    with autocast(device_type="cuda" if x_tensor.is_cuda else "cpu", enabled=False):
                        chunk_out = super(LayerNormMixedPrecision, self).forward(chunk)
                    
                    if original_dtype == torch.float16:
                         chunk_out = chunk_out.to(torch.float16)
                    elif original_dtype == torch.bfloat16:
                         chunk_out = chunk_out.to(torch.bfloat16)
                         
                    out_chunks.append(chunk_out)
                
                return torch.cat(out_chunks, dim=0)

            return apply_ln_chunked(input)

        elif input.dtype != torch.float32:
            input_f32 = input.float()
        else:
            input_f32 = input

        # Perform LayerNorm in FP32 (Standard path for small tensors)
        with autocast(device_type="cuda" if input.is_cuda else "cpu", enabled=False):
            out = super().forward(input_f32)
        
        # Cast back if necessary (or keep float32 if next layer expects it/can handle it)
        if original_dtype == torch.float16:
            return out.to(torch.float16)
        return out

class MLP(torch.nn.Module):
    """Multi-Layer Perceptron"""
    def __init__(self, 
                 in_features: int, 
                 hidden_size:int, 
                 out_features: int, 
                 has_bias:bool, 
                 device: torch.device | None, 
                 dtype: torch.dtype | None,  
                 activation: Activation = 'gelu', 
                 depth:int=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.layers = []
       
        if depth == 1:
            self.layers.append(nn.Linear(in_features, out_features, bias=has_bias, device=device, dtype=dtype))
        else:
             # input layer
            self.layers.append(nn.Linear(in_features, hidden_size, bias=has_bias, device=device, dtype=dtype))
            self.layers.append(ACTIVATION_FN[self.activation])
            # hidden layers
            for i in range(depth - 2):
                self.layers.append(nn.Linear(hidden_size, hidden_size, bias=has_bias, device=device, dtype=dtype))
                self.layers.append(ACTIVATION_FN[self.activation])
            # output layer
            self.layers.append(nn.Linear(hidden_size, out_features, bias=has_bias, device=device, dtype=dtype))
            torch.nn.init.normal_(self.layers[-1].weight)
        self.mlp = nn.Sequential(*self.layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class MultiheadAttention(torch.nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        qkv_combined: bool = True,
        dropout:float=0,
        recompute:bool=False
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_combined = qkv_combined
        self.dropout = dropout
        self.recompute = recompute
        self.device = device
        self.dtype = dtype

        self.out_proj_weight = torch.nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.embed_dim, device=self.device, dtype=self.dtype))
        self.qkv_proj_weight = torch.nn.Parameter(torch.empty(3, self.num_heads, self.head_dim, self.embed_dim, device=device, dtype=dtype))

        torch.nn.init.xavier_uniform_(self.out_proj_weight)
        nn.init.xavier_uniform_(self.qkv_proj_weight)

        self.q_proj_weight = None
        self.kv_proj_weight = None
        
        if recompute:
            self.forward = partial(checkpoint, self.forward, use_reentrant=False)  # type: ignore
    
    def get_cu_seqlens(self, batch_size: int, seqlen: int, device: torch.device) -> torch.Tensor:
        return torch.arange(
            0,
            (batch_size + 1) * seqlen,
            step=seqlen,
            dtype=torch.int32,
            device=device,
        )
    
    def compute_attention_by_torch(self, qkv:torch.Tensor|None, q:torch.Tensor|None, kv:torch.Tensor|None, attn_mask:torch.Tensor|None) -> torch.Tensor:
        '''Since flash attention does not support attn_mask, use scaled_dot_product_attention to compute attention when attn_mask is not None'''
        if qkv is not None:
            q, k, v = qkv.unbind(dim=-3)
        elif kv is not None and q is not None:
            k,v = kv.unbind(dim=-3)
        else:
            raise ValueError("When qkv is None, q and kv cannot both be None at the same time")
        assert q is not None and k is not None and v is not None, "q, k, and v must not be None"
        
        # Handle empty tensor case (e.g. when eval_pos=0, batch size might be 0 in some dimension)
        # If q is empty, output is empty. If k is empty, attention cannot be computed (softmax over 0), so return zeros.
        if q.numel() == 0 or k.numel() == 0:
            return torch.zeros_like(q)
        
        # Transpose to (Batch, Heads, SeqLen, HeadDim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Manual attention implementation to avoid CUDA error: invalid configuration argument
        
        # [FIX] Force float32 for Attention Score Calculation to prevent overflow
        q_dt, k_dt, v_dt = q.dtype, k.dtype, v.dtype
        if q.dtype != torch.float32: q = q.float()
        if k.dtype != torch.float32: k = k.float()
        # v can stay in original dtype usually, but for matmul stability, let's keep it consistent or cast later
        
        B, H, L, D = q.shape
        scale = 1.0 / (D ** 0.5)
        
        # (B, H, L, D) @ (B, H, D, S) -> (B, H, L, S)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float('-inf'))
            else:
                scores = scores + attn_mask
        
        attn_probs = torch.softmax(scores, dim=-1)
        # Convert back to v's dtype for the second matmul to save memory/speed, OR keep float32
        # Keeping float32 is safer for accumulation
        
        attn_probs = torch.nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        
        if v.dtype != torch.float32: v = v.float()

        # (B, H, L, S) @ (B, H, S, D) -> (B, H, L, D)
        attention_outputs = torch.matmul(attn_probs, v)
        
        attention_outputs = attention_outputs.transpose(1, 2)
        
        # Cast output back to original dtype
        if attention_outputs.dtype != q_dt:
            attention_outputs = attention_outputs.to(q_dt)
            
        return attention_outputs
    
    def compute_attention_by_flashattn(self, qkv:torch.Tensor|None, q:torch.Tensor|None, kv:torch.Tensor|None) -> torch.Tensor:
        "Compute attention using flash attention"
        assert HAVE_FLASH_ATTN, "Flash attention is not supported. Please install/reinstall flash attention."
        if self.qkv_combined and qkv is not None:
            B,S = qkv.shape[:2]
            atten_out = flash_attn_varlen_qkvpacked_func( # type: ignore
                        qkv.reshape(B * S, 3, self.num_heads, self.head_dim),
                        self.get_cu_seqlens(B, S, qkv.device),
                        S,
                        dropout_p=self.dropout,
                        softmax_scale=None,
                        causal=False,
                        return_attn_probs=False,
                        deterministic=False,
                    )
        elif not self.qkv_combined and q is not None and kv is not None:
            B,S = q.shape[:2]
            kv_shape = kv.shape
            atten_out = flash_attn_varlen_kvpacked_func( # type: ignore
                    q.reshape(B * S, self.num_heads, self.head_dim),
                    kv.reshape(B * kv_shape[1], 2, self.num_heads, self.head_dim),
                    self.get_cu_seqlens(B, S, q.device),
                    self.get_cu_seqlens(B, kv_shape[1], kv.device),
                    S,
                    kv_shape[1],
                    dropout_p=self.dropout,
                    causal=False,
                    return_attn_probs=False,
                    deterministic=False,
                )
        return atten_out # type: ignore
    
    def chunked_flash_attention(
            self,
            qkv: torch.Tensor | None,
            q: torch.Tensor | None,
            kv: torch.Tensor | None
    ) -> torch.Tensor:
        assert HAVE_FLASH_ATTN, "Flash attention is not supported. Please install/reinstall flash attention."

        chunk_size = 1000000000 // 2 // 192 // qkv.shape[1]

        if self.qkv_combined and qkv is not None:
            B, S = qkv.shape[:2]
            # Split into chunks along sequence dimension
            atten_out = torch.empty(B, S, self.num_heads * self.head_dim, device=qkv.device, dtype=qkv.dtype)

            for i in range(0, B, chunk_size):
                chunk_end = min(i + chunk_size, B)
                # qkv_chunk = qkv[:, i:chunk_end]
                qkv_chunk = qkv[i:chunk_end]

                chunk_B, chunk_S = qkv_chunk.shape[:2]
                chunk_out = flash_attn_varlen_qkvpacked_func(
                    qkv_chunk.reshape(chunk_B * chunk_S, 3, self.num_heads, self.head_dim),
                    self.get_cu_seqlens(chunk_B, chunk_S, qkv.device),
                    chunk_S,
                    dropout_p=self.dropout,
                    softmax_scale=None,
                    causal=False,
                    return_attn_probs=False,
                    deterministic=False,
                )
                atten_out[i:chunk_end] = chunk_out.reshape(chunk_B, chunk_S, -1)

        elif not self.qkv_combined and q is not None and kv is not None:
            B, S = q.shape[:2]
            kv_shape = kv.shape
            # Split into chunks along sequence dimension for Q
            atten_out = torch.empty(B, S, self.num_heads * self.head_dim, device=q.device, dtype=q.dtype)

            for i in range(0, B, chunk_size):
                chunk_end = min(i + chunk_size, B)
                q_chunk = q[i:chunk_end]

                chunk_B, chunk_S = q_chunk.shape[:2]
                chunk_out = flash_attn_varlen_kvpacked_func(
                    q_chunk.reshape(chunk_B * chunk_S, self.num_heads, self.head_dim),
                    kv.reshape(B * kv_shape[1], 2, self.num_heads, self.head_dim),
                    self.get_cu_seqlens(chunk_B, chunk_S, q.device),
                    self.get_cu_seqlens(B, kv_shape[1], kv.device),
                    chunk_S,
                    kv_shape[1],
                    dropout_p=self.dropout,
                    causal=False,
                    return_attn_probs=False,
                    deterministic=False,
                )
                atten_out[i:chunk_end] = chunk_out.reshape(chunk_B, chunk_S, -1)

            # Concatenate all query tiles
        return atten_out.reshape(B * S, self.num_heads, self.head_dim)

    def caculate_attention_score(self, q: torch.Tensor | None, k: torch.Tensor | None) -> torch.Tensor:
        if len(q.shape) == 3:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
        logits = torch.einsum("b q h d, b k h d -> b q k h", q, k)
        logits *= torch.sqrt(torch.tensor(1.0 / q.shape[-1])).to(k.device)
        ps = torch.softmax(logits.float(), dim=2).to(torch.float16).mean(dim=-1)
        del logits
        return ps

    def chunked_caculate_attention_score(self, q: torch.Tensor | None, k: torch.Tensor | None) -> torch.Tensor:
        if len(q.shape) == 4:
            B, S, H, D = q.shape
        else:
            S, H, D = q.shape
        try:
            if len(q.shape) == 3:
                chunk_size = max(10000000 // 2 // 192 // q.shape[0], 1)
                ps = torch.zeros(1, q.shape[0], k.shape[0], device=q.device, dtype=torch.float16)
                for i in range(0, S, chunk_size):
                    chunk_end = min(i + chunk_size, q.shape[0])
                    q_chunk = q[i:chunk_end]
                    k_chunk = k
                    ps_chunk = self.caculate_attention_score(q_chunk.to(q.device), k_chunk.to(k.device))
                    ps[:, i:chunk_end] = ps_chunk.to(q.device)
            else:
                chunk_size = max(2000000000 // 2 // 192 // q.shape[1] // k.shape[1], 1)
                ps = torch.zeros(B, q.shape[1], k.shape[1], device=q.device, dtype=torch.float16)
                for i in range(0, B, chunk_size):
                    chunk_end = min(i + chunk_size, B)
                    q_chunk = q[i:chunk_end]
                    k_chunk = k[i:chunk_end]
                    ps_chunk = self.caculate_attention_score(q_chunk.to(q.device), k_chunk.to(k.device))
                    ps[i:chunk_end] = ps_chunk.to(q.device)
        except OutOfMemoryError as e:
            attention_device = torch.device("cpu")
            if len(q.shape) == 3:
                chunk_size = max(10000000 // 2 // 192 // q.shape[0], 1)
                ps = torch.zeros(1, q.shape[0], k.shape[0], device=q.device, dtype=torch.float16)
                for i in range(0, S, chunk_size):
                    chunk_end = min(i + chunk_size, q.shape[0])
                    q_chunk = q[i:chunk_end]
                    k_chunk = k
                    ps_chunk = self.caculate_attention_score(q_chunk.to(q.device), k_chunk.to(k.device))
                    ps[:, i:chunk_end] = ps_chunk.to(attention_device)
            else:
                chunk_size = max(2000000000 // 2 // 192 // q.shape[1] // k.shape[1], 1)
                ps = torch.zeros(B, q.shape[1], k.shape[1], device=attention_device, dtype=torch.float16)
                for i in range(0, B, chunk_size):
                    chunk_end = min(i + chunk_size, B)
                    q_chunk = q[i:chunk_end]
                    k_chunk = k[i:chunk_end]
                    ps_chunk = self.caculate_attention_score(q_chunk.to(q.device), k_chunk.to(k.device))
                    ps[i:chunk_end] = ps_chunk.to(attention_device)
        return ps.to(torch.float16)

    @override
    def forward(self, 
                x: torch.Tensor, 
                x_kv: Optional[torch.Tensor] = None, 
                copy_first_head_kv: bool = False,
                attn_mask: torch.Tensor | None = None, 
                calculate_sample_attention:bool=False, 
                calculate_feature_attention:bool=False) -> tuple[torch.Tensor,torch.Tensor | None,torch.Tensor | None]:
        """
        x: [batch_size, seq_len, feature, embed_dim]
        kv: Optional[batch_size, seq_len_kv, feature, embed_dim] — only needed if qkv_combined=False
        copy_first_head: Reuse the results from the first attention head
        """
        # feature attention: [B S F E]  
        # item attention: [B F S E]
        # B, T, C = x.shape
        B, S, _, _ = x.shape
        assert x.shape[-1] == self.embed_dim

        x = x.reshape(B*S, *x.shape[-2:])
        BS, F, E = x.shape
        
        qkv = None
        q = None
        kv = None
        feature_attention=None
        sample_attention=None
        # batch_size = None
        # seqlen = None
        if self.qkv_combined:
            qkv = torch.einsum("... s, j h d s -> ... j h d", x, self.qkv_proj_weight)
        else:
            self.q_proj_weight = self.qkv_proj_weight[0]
            self.kv_proj_weight = self.qkv_proj_weight[1:]
            assert x_kv is not None, "kv combined attention requires kv input"
            x_kv = x_kv.reshape(x_kv.shape[0]*x_kv.shape[1], *x_kv.shape[-2:])
            q = torch.einsum("... s, h d s -> ... h d", x, self.q_proj_weight)
            if copy_first_head_kv:
                kv_weights = self.kv_proj_weight[:,:1]
                kv = torch.einsum("... s, j h d s -> ... j h d", x_kv, kv_weights)
                expand_shape = [-1 for _ in kv.shape]
                expand_shape[-2] = self.num_heads
                kv = kv.expand(*expand_shape)
            else:
                kv = torch.einsum("... s, j h d s -> ... j h d", x_kv, self.kv_proj_weight)

        try:
            # [FIX] Relax constraints for Flash Attention
            # If mask is provided but we have flash attn, we should use it if possible.
            # However, standard flash_attn support for arbitrary masks is limited in v1.
            # Assuming Flash Attn 2 varlen handles some masking, or we rely on torch for masked cases.
            # But for Feature Attention (usually no mask), this path is critical.
            
            # Check for generic 'cuda' availability rather than specific weight device to be safer
            use_flash = HAVE_FLASH_ATTN and x.device.type == "cuda"
            
            if use_flash and attn_mask is None:
                atten_out = self.compute_attention_by_flashattn(qkv, q, kv)
            else:
                # Fallback to Torch or Chunked
                # If we are here with a Massive sequence and Attention, we might OOM.
                # Try Chunked Attention proactively if sequence length is huge?
                # No, let's rely on OOM catch.
                atten_out = self.compute_attention_by_torch(qkv, q, kv, attn_mask)

        except OutOfMemoryError as e:
            # [FIX] Explicitly catch OOM and try chunked attention
            # Note: chunked_flash_attention currently doesn't support attn_mask in this implementation
            if attn_mask is None:
                print(f"[WARN] OOM in Attention. Falling back to Chunked Flash Attention. Shape: {x.shape}")
                atten_out = self.chunked_flash_attention(qkv, q, kv)
            else:
                 # If we have a mask, we can't easily chunk with current impl. 
                 # Re-raise or try CPU offload (very slow).
                 raise e

        atten_out = atten_out.reshape(BS, F, self.num_heads, self.head_dim)

        if qkv is not None:
            q, k, v = qkv.unbind(dim=2)
        else:
            k,v=kv.unbind(dim=2)
        if calculate_feature_attention:
            try:
                feature_attention = self.caculate_attention_score(q, k)
            except (OutOfMemoryError, RuntimeError) as e:
                feature_attention = self.chunked_caculate_attention_score(q, k)

        if calculate_sample_attention:
            try:
                sample_attention = self.caculate_attention_score(q[-1], k[-1])
            except (OutOfMemoryError, RuntimeError) as e:
                sample_attention = self.chunked_caculate_attention_score(q[-1], k[-1])
        out = torch.einsum(
            "... h d, h d s -> ... s",
            atten_out,
            self.out_proj_weight,
        )

        return out.reshape(B, S, *out.shape[1:]),feature_attention,sample_attention

class EncoderBaseLayer(nn.Module):
    "Base encoder layer of the Transformer model"
    def __init__(self, 
                 nhead: int, 
                 embed_dim: int, 
                 hid_dim:int, 
                 dropout: float=0,
                 pre_norm: bool=False,
                 activation: str='gelu',
                 layer_norm_eps: float=1e-5,
                 device: torch.device|None=None,
                 dtype: torch.dtype|None=None,
                 recompute_attn: bool=False,
                 mlp_use_residual:bool=False,
                 layer_arch: str = 'fmfmsm',
                 seq_attn_isolated: bool = False,
                 seq_attn_serial: bool = False,  
                 self_share_all_kv_heads: bool = False,
                 cross_share_all_kv_heads: bool = True,
                 sample_mixer_type: Literal['attention', 'mamba', 'linear'] = 'attention',
                 use_checkpoint: bool = False,
                 d_state: int = 128,
                 **kwargs
                 ):
        super().__init__()
        self.nhead = nhead
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.pre_norm = pre_norm
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.device = device
        self.dtype = dtype
        self.layer_arch = layer_arch
        self.head_dim = self.embed_dim // self.nhead
        self.recompute_attn = recompute_attn
        self.sample_mixer_type = sample_mixer_type
        self.use_checkpoint = use_checkpoint
        
        self.feature_attentions = []
        self.sequence_attentions = []
        self.mlp = []
        self.feature_attn_num = 1           # feature attention number
        self.seq_attn_num = 1             # sequence attention number
        self.mlp_num = 1                    # mlp number
        if layer_arch == 'fmfmsm':
            self.feature_attn_num = 2
            self.mlp_num = 3
        
        self.self_share_all_kv_heads = self_share_all_kv_heads
        self.cross_share_all_kv_heads = cross_share_all_kv_heads
        self.seq_attn_serial = seq_attn_serial
        self.seq_attn_isolated = seq_attn_isolated

        if self.seq_attn_isolated:
            self.seq_attn_num *= 2

        # attention+MLP
        self.feature_attentions = nn.ModuleList()
        # [Design Fix] Feature-level attention always uses Full Transformer (MultiheadAttention)
        # regardless of sample_mixer_type (which controls Sequence-level mixing strategy).
        feature_mixer_type = 'attention' 
        
        for _ in range(self.feature_attn_num):
            if feature_mixer_type == 'attention':
                self.feature_attentions.append(
                    MultiheadAttention(
                        embed_dim=self.embed_dim,
                        num_heads=self.nhead,
                        device=self.device,
                        dtype=self.dtype,
                        qkv_combined=True,
                        dropout=self.dropout,
                        recompute=self.recompute_attn,
                    )
                )
            # Retain support for other types if explicitly requested in future refactoring
            elif feature_mixer_type == 'mamba':
                self.feature_attentions.append(
                    Mamba2Block(
                        d_model=self.embed_dim,
                        d_state=d_state,
                        d_conv=4,
                        expand=2,
                        headdim=self.head_dim,
                        mixing_type='feature'
                    )
                )
            elif feature_mixer_type == 'linear':
                self.feature_attentions.append(
                    LinearAttentionBlock(
                        dim=self.embed_dim,
                        num_heads=self.nhead,
                        attn_drop=self.dropout,
                        proj_drop=self.dropout,
                        mixing_type='feature'
                    )
                )
        
        self.sequence_attentions = nn.ModuleList()
        for _ in range(self.seq_attn_num):
            if self.sample_mixer_type == 'mamba':
                 self.sequence_attentions.append(
                    Mamba2Block(
                        d_model=self.embed_dim,
                        d_state=d_state,
                        d_conv=4,
                        expand=2,
                        headdim=self.head_dim,
                        mixing_type='sequence' # Explicitly mix sequence (Samples)
                    )
                )
            elif self.sample_mixer_type == 'linear':
                self.sequence_attentions.append(
                    LinearAttentionBlock(
                        dim=self.embed_dim,
                        num_heads=self.nhead,
                        attn_drop=self.dropout,
                        proj_drop=self.dropout,
                        mixing_type='sequence' # Explicitly mix sequence (Samples)
                    )
                )
            else:
                self.sequence_attentions.append(
                    MultiheadAttention(
                        embed_dim=self.embed_dim,
                        num_heads=self.nhead,
                        device=self.device,
                        dtype=self.dtype,
                        qkv_combined=False,
                        dropout=self.dropout,
                        recompute=self.recompute_attn,
                    ) 
                )

        self.mlp = nn.ModuleList(
                                    [
                                        MLP(
                                            in_features=self.embed_dim,
                                            hidden_size=self.hid_dim,
                                            out_features=self.embed_dim,
                                            has_bias=False,
                                            device=self.device,
                                            dtype=self.dtype,
                                            activation=self.activation,
                                            depth=2,
                                        ) 
                                        for _ in range(self.mlp_num)
                                    ]
                                 )
        
        self.layer_steps = []
        if self.layer_arch == 'fmfmsm':
            assert len(self.feature_attentions) >= 2 and len(self.sequence_attentions) >= 1 and len(self.mlp) >= 3
            self.layer_steps = [
                                partial(
                                    self.call_features_attention,
                                    index=0
                                ),
                                self.mlp[0],
                                partial(
                                    self.call_features_attention,
                                    index=1
                                ),
                                self.mlp[1],
                                partial(
                                    self.call_sequence_attention,
                                    index=0
                                ),
                                self.mlp[2]
            ]
        elif self.layer_arch == 'smf':
            assert len(self.feature_attentions) >= 1 and len(self.sequence_attentions) >= 1 and len(self.mlp) >= 1
            self.layer_steps = [
                                partial(
                                    self.call_sequence_attention,
                                    index=0
                                ),
                                self.mlp[0],
                                partial(
                                    self.call_features_attention,
                                    index=0
                                )
            ]
        else:
            raise ValueError(f"Unsupport layr arch: {self.layer_arch}")
    
        self.layer_norms = nn.ModuleList()
        for idx in range(len(self.layer_steps)):
             # Use RMSNorm for Mamba layers (Sample Mixers are Mamba when configured)
             # In fmfmsm/smf, check which steps correspond to Mamba
             # If sample_mixer_type is 'mamba'
             is_mamba = False
             
             # Identify if this step involves Mamba
             # We need to map linear index 'idx' back to what layer it is.
             # This is a bit fragile but based on initialization logic:
             # fmfmsm: 0(M), 1(MLP), 2(M), 3(MLP), 4(FM-Mamba/Linear/Attn), 5(MLP)
             # However, sequence_attentions can also be Mamba if configured?
             # No, sequence_attentions logic above checks sample_mixer_type for feature_attentions.
             # But for sequence_attentions, logic also checks sample_mixer_type?
             # Wait, in the code read earlier:
             # if self.sample_mixer_type == 'mamba': self.feature_attentions.append(Mamba...)
             # loop for seq_attn uses sample_mixer_type too?
             # Let's assume Mamba is used wherever 'mamba' block is inserted.
             
             # Safer approach: Check if the sublayer function is associated with Mamba.
             # But we wrapped them in partials or just MLPs.
             
             # Simply use RMSNorm everywhere if sample_mixer_type is 'mamba'. Mamba authors recommend RMSNorm universally.
             # [MODIFED] Also use RMSNorm for Linear Attention (part of the upgrade)
             if self.sample_mixer_type in ['mamba', 'linear']:
                 self.layer_norms.append(RMSNorm(dim=self.embed_dim, eps=self.layer_norm_eps))
             else:
                 self.layer_norms.append(
                     LayerNormMixedPrecision(normalized_shape=self.embed_dim, eps=self.layer_norm_eps, 
                                            elementwise_affine=False, device=self.device, dtype=self.dtype)
                 )
    
    def create_attn_mask(self, q_mask:torch.Tensor, k_mask:torch.Tensor)->torch.Tensor:
        """
        Create attention mask
        
        Args:
            q_mask (torch.Tensor): Query sequence mask, with shape [batch_size, head_count, q_seq_len]
            k_mask (torch.Tensor): Key sequence mask, with shape   [batch_size, head_count, k_seq_len]
        
        Returns:
            torch.Tensor: attention mask, with shape [batch_size, head_count, q_seq_len, k_seq_len]
        """
        _, _, q_seq_len = q_mask.shape
        _, _, k_seq_len = k_mask.shape
        
        q_mask_bool = q_mask.bool()  # [batch_size, head_count, q_seq_len]
        k_mask_bool = k_mask.bool()  # [batch_size, head_count, k_seq_len]
        
        q_expanded = q_mask_bool.unsqueeze(-1)
        k_expanded = k_mask_bool.unsqueeze(-2)
        
        valid_attn = q_expanded & k_expanded
        attn_mask = ~valid_attn
        _, _, q_seq_len, k_seq_len = attn_mask.shape
        attn_mask = attn_mask.reshape(-1, q_seq_len, k_seq_len)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        return attn_mask

    def call_features_attention(self, x: torch.Tensor, feature_atten_mask: torch.Tensor | None, eval_pos: int,
                                index: int = 0,calculate_feature_attention:bool=False):
        assert len(self.feature_attentions) > index
        attn_mask = None
        module = self.feature_attentions[index]
        
        if isinstance(module, MultiheadAttention):
            if feature_atten_mask is not None:
                attn_mask = self.create_attn_mask(feature_atten_mask, feature_atten_mask)
            
            # [FIX] MultiheadAttention requires 4D input [Batch, Seq, Groups, Emb]
            # It internally flattens dimensionality. Do NOT manually reshape to 3D here.
            
            # However, we must ensure x is 4D. 
            # If x is 3D [Batch, Seq, Emb], we might need to unsqueeze groups?
            # But the system seems designed for 4D. If x is 3D, MultiheadAttention will fail anyway.
            # Assuming x is 4D here as per architecture design.
            
            out = module(
                x,
                x_kv=None,
                attn_mask=attn_mask,
                calculate_feature_attention=calculate_feature_attention
            )
            
            # Output handling
            if isinstance(out, tuple):
                 return out
            else:
                 return out, None, None

        else:
            # Mamba or Linear (Handled internally)
            out = module(x)
            return out, None, None

    def call_sequence_attention(self, x: torch.Tensor, feature_atten_mask: torch.Tensor | None, eval_pos: int,
                                index: int = 0,calculate_sample_attention:bool=False):
        assert len(self.sequence_attentions) > index
        sample_attention=None
        index1 = index*2 if self.seq_attn_isolated else index
        index2 = index1+1 if self.seq_attn_isolated else index1
        assert index2 < len(self.sequence_attentions), f"Error: index2({index2}) >= len(self.sequence_attentions)({len(self.sequence_attentions)})"

        module = self.sequence_attentions[index1]
        
        # [FIX] Mamba or Linear (Handled internally with proper 4D support)
        if isinstance(module, Mamba2Block) or isinstance(module, LinearAttentionBlock):
             out = module(x)
             return out, None, None

        # [FIX] MultiheadAttention for Sequence Mixing
        # MultiheadAttention expects 4D input [EffectiveBatch, EffectiveSeq, Feature, Embed]
        # We want to mix across Sequence dimension (1).
        # We need to swap Sequence and Groups (dim 1 and 2) so that:
        # Input to Attn: [Batch, Groups, Seq, Embed]
        # Attn interprets: B'=Batch, S'=Groups (context), F'=Seq (mixing dim? No wait)
        
        # Re-reading MultiheadAttention logic:
        # B, S, _, _ = x.shape
        # x = x.reshape(B*S, ...)
        # It collapses dim 0 and 1.
        # It attends over dim 1 of the reshaped tensor? No.
        # It projects qkv: "... s, j h d s". 's' is last dim.
        # It reshapes to [BS, F, num_heads, head_dim]. F is dim 2.
        # It computes attention over dim 2??
        # compute_attention_by_torch:
        # q: [BS, F, H, D]. transpose(1,2) -> [BS, H, F, D].
        # matmul q @ k.T -> [BS, H, F, F].
        # So it computes attention over F (dimension 2 of input).
        
        # So for Feature Attention (mix Groups):
        # Input [B, S, G, E].
        # Collapse B, S. -> [B*S, G, E].
        # Attn over G. Correct.
        
        # For Sequence Attention (mix Seq):
        # Input [B, S, G, E].
        # We want Attn over S.
        # We want F position to be S.
        # So input should be [B, G, S, E].
        # Collapse B, G -> [B*G, S, E].
        # Attn over S. Correct.
        
        # So we need to transpose(1, 2) before calling sequence attentions logic, 
        # BUT the logic below specifically handles the splitting and transposing manually.
        
        # Original logic:
        # x[:, :eval_pos].transpose(1, 2) -> [B, S, G, E] sliced -> [B, S_ctx, G, E] -> [B, G, S_ctx, E]
        # This is exactly [B, G, S, E] format for Context. Correct.

        x_train = self.sequence_attentions[index1](
                        x = x[:, :eval_pos].transpose(1, 2),
                        x_kv = x[:, :eval_pos].transpose(1, 2),
                        copy_first_head_kv = True if self.self_share_all_kv_heads else False,
                    )[0].transpose(1, 2)

        if self.seq_attn_serial:
            x[:, :eval_pos] = x_train

        if eval_pos < x.shape[1]:
            x_test,_,sample_attention = self.sequence_attentions[index2](
                                                    x=x[:, eval_pos:].transpose(1, 2),
                                                    x_kv=x[:, :eval_pos].transpose(1, 2),
                                                    copy_first_head_kv=True if self.cross_share_all_kv_heads else False,
                                                    calculate_sample_attention=calculate_sample_attention
                                                )
            x_test=x_test.transpose(1, 2)
        else:
            x_test = None
            # print(f"\033[30;43mWarning: eval_pos >= x.shape[1]!\033[0m")
        
        if x_test is not None:
            return torch.cat([x_train, x_test], dim=1),None,sample_attention
        else:
            return x_train

    # Sub-step names for profiling (fmfmsm architecture)
    _SUBSTEP_NAMES = {
        'fmfmsm': ['feat_attn_1', 'mlp_1', 'feat_attn_2', 'mlp_2', 'seq_mixer', 'mlp_3'],
        'smf': ['seq_mixer', 'mlp_1', 'feat_attn_1'],
    }

    def forward(self, x: torch.Tensor, feature_atten_mask: torch.Tensor, eval_pos: int,**kwargs) -> tuple[torch.Tensor,torch.Tensor | None,torch.Tensor | None]:
        calculate_sample_attention = kwargs.get("calculate_sample_attention", False)
        calculate_feature_attention = kwargs.get("calculate_feature_attention", False)
        layer_idx = kwargs.get("layer_idx", 11)

        _profiler = get_profiler()
        _substep_names = self._SUBSTEP_NAMES.get(self.layer_arch, [f'step_{i}' for i in range(len(self.layer_steps))])

        feature_attenion=None
        sample_attention=None
        for idx, (sublayer, layer_norm) in enumerate(zip(self.layer_steps, self.layer_norms)):
            if self.pre_norm:
                residual = x
                x = layer_norm(x)
                
                # Checkpointing logic for Sample Mixer (idx == 0 or idx == 4 depending on arch)
                # In 'fmfmsm', idx 0 and 2 are feature_attn (sample mixer), idx 4 is sequence_attn (feature mixer)
                # Wait, let's check layer_steps in __init__
                # fmfmsm: 
                # 0: call_features_attention (Sample Mixer 1)
                # 1: mlp[0]
                # 2: call_features_attention (Sample Mixer 2)
                # 3: mlp[1]
                # 4: call_sequence_attention (Feature Mixer)
                # 5: mlp[2]
                
                # We want to checkpoint Sample Mixers (Mamba/Linear) which are at idx 0 and 2
                should_checkpoint = self.use_checkpoint and (idx == 0 or idx == 2) and x.requires_grad

                if idx == 2 and calculate_feature_attention and layer_idx == 11:
                    x, feature_attenion, _ = sublayer(x, feature_atten_mask, eval_pos,calculate_feature_attention=True)
                elif idx == 4 and calculate_sample_attention and layer_idx == 11:
                    x, _, sample_attention = sublayer(x, feature_atten_mask, eval_pos,calculate_sample_attention=True)
                else:
                    if isinstance(sublayer, functools.partial):
                        if should_checkpoint:
                            # checkpoint requires the function to return a Tensor (or tuple of Tensors)
                            # sublayer returns (x, None, None) usually for Mamba/Linear
                            # We need a wrapper to handle the tuple return if checkpoint is used, 
                            # but checkpoint handles tuple returns fine in recent PyTorch.
                            # However, we need to ensure inputs are tensors or handled correctly.
                            # feature_atten_mask might be None.
                            
                            # Define a wrapper to adapt arguments for checkpoint
                            # CRITICAL FIX: capture sublayer from the loop variable to avoid late binding issues
                            def custom_forward(x_in, mask_in, pos_in, sl=sublayer):
                                return sl(x_in, mask_in, pos_in)
                                
                            x = checkpoint(custom_forward, x, feature_atten_mask, eval_pos, use_reentrant=False)
                        else:
                            x = sublayer(x, feature_atten_mask, eval_pos)
                            
                        if isinstance(x, tuple):
                            x = x[0]
                    else:
                        # MLP
                        if self.use_checkpoint and x.requires_grad:
                             x = checkpoint(sublayer, x, use_reentrant=False)
                        else:
                             x = sublayer(x)
                             
                        if isinstance(x, tuple):
                            x = x[0]
                x = x + residual

                # Record profiling for this sub-step
                if idx < len(_substep_names):
                    _profiler.mark(_substep_names[idx])

                # --- Per-Step NAN Check to identify malfunctioning layer (training only) ---
                if self.training and (torch.isnan(x).any() or torch.isinf(x).any()):
                     print(f"\\n[Error] NAN/INF detected inside EncoderBaseLayer at Step {idx}")
                     print(f"  Sublayer Type: {type(sublayer)}")

            else:
                # Post-norm implementation (omitted for brevity as pre_norm is default/recommended)
                # But for completeness, let's just apply standard logic without checkpointing for post-norm 
                # or assume pre-norm is used for Mamba.
                residual = x
                if idx == 2 and calculate_feature_attention and layer_idx == 11:
                    x, feature_attenion, _ = sublayer(x, feature_atten_mask, eval_pos,calculate_feature_attention=True)
                    x = x + residual
                elif idx == 0 and calculate_sample_attention and layer_idx == 11:
                    x, _, sample_attention = sublayer(x, feature_atten_mask, eval_pos, calculate_sample_attention=True)
                    x = x + residual
                else:
                    if  isinstance(sublayer, functools.partial):
                        x = sublayer(x, feature_atten_mask, eval_pos)
                        if isinstance(x, tuple):
                            x = x[0]
                        x = x + residual
                    else:
                        x = sublayer(x)
                        if isinstance(x, tuple):
                            x = x[0]
                        x = x + residual
                x=layer_norm(x)
                # Record profiling for this sub-step (post-norm)
                if idx < len(_substep_names):
                    _profiler.mark(_substep_names[idx])
        return x,feature_attenion,sample_attention
                
class LayerStack(nn.Module):
    """
    A flexible container module similar to ``nn.Sequential`` that allows 
    keyword arguments to be passed through to each layer.
    """
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, **kwargs):
        _profiler = get_profiler()
        for idx,layer in enumerate(self.layers):
            kwargs["layer_idx"] = idx
            _profiler.enter_layer(idx)
            x,feature_attention,sample_attention = layer(x,**kwargs)
            _profiler.exit_layer()
            # DEBUG CHECK (training only - skip during inference for performance)
            if self.training and (torch.isnan(x).any() or torch.isinf(x).any()):
                msg = f"NAN/INF detected after Layer {idx} ({type(layer).__name__})"
                print(f"\n\033[91m{msg}\033[0m")
                print(f"  Min: {x.min().item()}, Max: {x.max().item()}, Mean: {x.mean().item()}")
                raise ValueError(msg)
        return x,feature_attention,sample_attention
