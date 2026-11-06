import torch
import torch.nn as nn
import torch.nn.functional as F

# RMSNorm reference: TorchScale, Copyright (c) 2022 Microsoft, MIT License.
# FEAT uses affine normalization and retains its BF16 inference precision path.
# See third_party/torchscale/README.md and LICENSE for source and license details.
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if not self.training and x.dtype == torch.bfloat16:
            return self._norm(x) * self.weight
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LinearAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., mixing_type='sequence'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.mixing_type = mixing_type # 'sequence' or 'feature'
        assert mixing_type in ['sequence', 'feature'], "mixing_type must be 'sequence' or 'feature'"

        # [IMPROVEMENT] Depthwise Conv for Local Context
        # Adds local inductive bias which linear attention lacks. 
        # Kernel size 3 captures immediate neighbors.
        self.dwc = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # [IMPROVEMENT] Gated Linear Attention (GLA) mechanism
        # Adds a data-dependent gate to the output, similar to Mamba/GLU.
        self.gate_proj = nn.Linear(dim, dim)
        
        # [REMOVED] Redundant Self-Norm (EncoderBaseLayer handles Pre-Norm)
        # self.norm = RMSNorm(dim)

        # [IMPROVEMENT] QK Norm for training stability
        # Stabilizes the attention logits prevents value explosion
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        # [IMPROVEMENT] Output Normalization
        # Stabilizes the block output similar to Mamba
        self.out_norm = RMSNorm(dim)

    def forward(self, x):
        is_4d = x.dim() == 4
        if is_4d:
            B_orig, S_orig, G_orig, E_orig = x.shape
            if self.mixing_type == 'sequence':
                # Mix across Sequence (S), treat Groups (G) as batch
                x = x.permute(0, 2, 1, 3).reshape(B_orig*G_orig, S_orig, E_orig)
            else: # feature
                # Mix across Groups (G), treat Sequence (S) as batch
                x = x.reshape(B_orig*S_orig, G_orig, E_orig)
        
        # [REMOVED] Redundant Norm. Input x is already normalized by EncoderBaseLayer
        # x = self.norm(x)

        # [IMPROVEMENT] Apply Depthwise Convolution for Local Context
        # (B, L, C) -> (B, C, L) -> Conv -> (B, C, L) -> (B, L, C)
        # We replace x with the convolved version to strictly enforce local mixing before global attention.
        x = self.dwc(x.transpose(1, 2)).transpose(1, 2)
            
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # [IMPROVEMENT] QK Norm
        # Normalize Q and K before feature map. 
        # Reduces sensitivity to initialization and large values.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Linear Attention kernel (Katharopoulos et al. or similar)
        # phi(x) = elu(x) + 1
        # [FIX] Add small epsilon to ensure strict positivity and prevent denominator collapse
        q = F.elu(q) + 1.0 + 1e-4
        k = F.elu(k) + 1.0 + 1e-4
        
        # [MODIFIED] Attempt to respect Mixed Precision if stable
        # But Aggregation (KV sum) can explode in FP16 easily.
        # We keep FP32 for the Summation part only, but avoid casting whole input if unnecessary.
        
        # Implement Chunking for Memory Efficiency to avoid OOM on huge sequences
        # Instead of casting the whole (B, H, N, D) to float, we chunk it.
        
        # print(f"DEBUG: LinearAttentionBlock chunked forward running! Input shape: {x.shape}")

        B, H, N, D = q.shape

        # Chunk size for processing to keep peak memory low
        CHUNK_SIZE = 4096
        if N > 100000:
            CHUNK_SIZE = 2048
        elif N > 50000:
            CHUNK_SIZE = 3072

        # [OPTIMIZATION] Use bf16 accumulation during inference (sufficient precision),
        # use FP32 accumulation during training (safer for gradient flow).
        accum_dtype = x.dtype if (not self.training and x.dtype == torch.bfloat16) else torch.float32

        # 1. Compute KV: (B, H, D, D)
        kv = torch.zeros(B, H, D, D, device=q.device, dtype=accum_dtype)
        k_sum = torch.zeros(B, H, D, device=q.device, dtype=accum_dtype)

        for i in range(0, N, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, N)
            k_chunk = k[:, :, i:end, :].to(accum_dtype)
            v_chunk = v[:, :, i:end, :].to(accum_dtype)
            kv += torch.matmul(k_chunk.transpose(-1, -2), v_chunk)
            k_sum += k_chunk.sum(dim=2)

        # 2. Compute Output: out = q @ kv / z
        out_list = []

        for i in range(0, N, CHUNK_SIZE):
            end = min(i + CHUNK_SIZE, N)
            q_chunk = q[:, :, i:end, :].to(accum_dtype)
            z_chunk = 1.0 / (torch.matmul(q_chunk, k_sum.unsqueeze(-1)).squeeze(-1) + 1e-6)
            out_chunk = torch.matmul(q_chunk, kv)
            out_chunk = out_chunk * z_chunk.unsqueeze(-1)
            out_list.append(out_chunk.to(x.dtype))

        out = torch.cat(out_list, dim=2)
        
        # Cast back to input dtype
        out = out.reshape(B, N, C)
        
        # [IMPROVEMENT] Apply Gating
        # Gating helps select relevant information, masking out noise
        # Using the convolved x (local context aware) for gating
        # Switch to SiLU for stronger non-linearity and consistency with SwiGLU/Mamba
        gate = F.silu(self.gate_proj(x)) 
        out = out * gate
        
        out = self.proj_drop(self.proj(out))
        
        # [IMPROVEMENT] Apply Output Norm
        out = self.out_norm(out)
        
        if is_4d:
            out = out.reshape(B_orig, G_orig, S_orig, E_orig).permute(0, 2, 1, 3) if self.mixing_type == 'sequence' else out.reshape(B_orig, S_orig, G_orig, E_orig)
            
        return out
