import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba2
except ImportError:
    Mamba2 = None

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

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

class Mamba2Block(nn.Module):
    def __init__(self, d_model, d_state=128, d_conv=4, expand=2, headdim=64, bidirectional=True, mixing_type='sequence'):
        super().__init__()
        
        self.bidirectional = bidirectional
        self.mixing_type = mixing_type # 'sequence' or 'feature'
        assert mixing_type in ['sequence', 'feature'], "mixing_type must be 'sequence' or 'feature'"
        
        # [IMPROVEMENT] Prioritize Mamba2 if available for better performance and stability
        self.use_mamba1 = False
        if Mamba2 is not None:
             self.use_mamba1 = False
        elif Mamba is not None:
            self.use_mamba1 = True
        
        def _get_mamba_layer():
            if not self.use_mamba1 and Mamba2 is not None:
                return Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim
                )
            elif self.use_mamba1:
                return Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
            else:
                return None
        
        self.mamba = _get_mamba_layer()
        if self.bidirectional:
            self.mamba_rev = _get_mamba_layer()
            # [IMPROVEMENT] Learned Gated Fusion for Bidirectional Mamba
            # Instead of simple averaging, use concatenation and projection
            # This allows the model to weigh forward and backward context dynamically.
            self.fusion_linear = nn.Linear(d_model * 2, d_model)
        else:
            self.mamba_rev = None
            
        # [IMPROVEMENT] Output Normalization for Bidirectional Fusion
        # Stabilizes the merged representation from forward and backward passes
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        # x: [Batch, Seq, Dim] or [Batch, Seq, Groups, Dim]
        if self.mamba is None:
            raise ImportError("mamba_ssm is not installed. Please install it to use Mamba2Block.")
            
        if x.dim() == 4:
            B, S, G, E = x.shape
            if self.mixing_type == 'sequence':
                # Mix across Sequence (S). Treat Groups (G) as independent batch elements.
                # Shape: [B, S, G, E] -> [B, G, S, E] -> [B*G, S, E]
                x_in = x.permute(0, 2, 1, 3).reshape(B*G, S, E)
            else: # mixing_type == 'feature'
                # Mix across Groups (G). Treat Sequence (S) as independent batch elements.
                # Shape: [B, S, G, E] -> [B, S, G, E] -> [B*S, G, E]
                # Note: Mamba sees 'G' as the sequence length here.
                x_in = x.reshape(B*S, G, E)
        else:
            x_in = x
            
        # Mamba kernel can be sensitive to FP16 ranges with large sequence lengths/activations
        # [MODIFIED] Relaxed constraint: Allow Autocast (BF16/FP16) if user enabled it.
        # But ensure we respect mamba implementation requirements.
        # Mamba 2 handles low precision well.
        
        dtype_orig = x_in.dtype

        # Chunked Mamba processing for very long sequences
        MAMBA_CHUNK_THRESHOLD = 500_000
        total_elements = x_in.shape[0] * x_in.shape[1]

        def _run_mamba(mamba_module, inp):
            if total_elements > MAMBA_CHUNK_THRESHOLD and inp.shape[0] > 1:
                chunk_bs = max(1, MAMBA_CHUNK_THRESHOLD // inp.shape[1])
                chunks = []
                for ci in range(0, inp.shape[0], chunk_bs):
                    chunks.append(mamba_module(inp[ci:ci+chunk_bs]))
                return torch.cat(chunks, dim=0)
            return mamba_module(inp)

        out = _run_mamba(self.mamba, x_in)

        if self.bidirectional and self.mamba_rev is not None:
                x_flipped = torch.flip(x_in, dims=[1])
                out_rev = _run_mamba(self.mamba_rev, x_flipped)
                out_rev = torch.flip(out_rev, dims=[1])
                del x_flipped

                out = self.fusion_linear(torch.cat([out, out_rev], dim=-1))
                del out_rev

        out = self.out_norm(out)

        # CRITICAL: cast back to input dtype — Mamba may output fp32 internally
        if out.dtype != dtype_orig:
            out = out.to(dtype_orig)

        if x.dim() == 4:
            if self.mixing_type == 'sequence':
                # [B*G, S, E] -> [B, G, S, E] -> [B, S, G, E]
                out = out.reshape(B, G, S, E).permute(0, 2, 1, 3)
            else: # feature
                # [B*S, G, E] -> [B, S, G, E]
                out = out.reshape(B, S, G, E)
            
        return out
