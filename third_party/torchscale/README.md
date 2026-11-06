# RMSNorm reference and license

The RMSNorm implementations in `feat/model/mamba_layer.py` and `feat/model/linear_attn_layer.py` use the [TorchScale RMSNorm reference](https://github.com/microsoft/torchscale/blob/4d1e0e82e5adf86dd424f1463192635b73fc8efc/torchscale/component/rms_norm.py), Copyright (c) 2022 Microsoft, under the MIT License.

Reference revision: [`4d1e0e82e5adf86dd424f1463192635b73fc8efc`](https://github.com/microsoft/torchscale/tree/4d1e0e82e5adf86dd424f1463192635b73fc8efc) (2024-04-11). The upstream [LICENSE](https://github.com/microsoft/torchscale/blob/4d1e0e82e5adf86dd424f1463192635b73fc8efc/LICENSE) is preserved verbatim in [LICENSE](LICENSE).

FEAT specializes the reference to affine normalization (`elementwise_affine=True`) and retains its existing BF16 inference precision path. The normalization arithmetic, parameter names, and checkpoint layout are unchanged. The reference is explicitly adopted for this release; this record does not assert that TorchScale was the historical source of earlier unannotated snippets.

Only these small RMSNorm implementations are covered by this attribution. TorchScale is not an installation dependency, and no TorchScale model weights are included.

License SHA-256: `c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383`.
