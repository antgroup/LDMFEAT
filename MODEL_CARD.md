# FEAT-38M

FEAT-38M supports in-context tabular classification, regression, missing-value imputation, and feature-relation analysis. Inputs consist of tabular features and observed context labels; query labels are not required for prediction.

## Artifact

| Field | Value |
| --- | --- |
| Filename | `FEAT-38M.pt` |
| Size | 155,418,367 bytes |
| SHA-256 | `84acc3161c52ce67f99a43d202008b5c9f25826aff2d5b917cbfe6ced8ba6ade` |
| Top-level checkpoint keys | `state_dict`, `config` |
| Tensors | 439 |
| Stored tensor elements, including buffers | 38,807,278 |
| Layers / embedding dimension / attention heads | 12 / 256 / 8 |
| Sample mixer | Mamba2, with linear attention every fourth layer |
| Features per group | 2 |
| Mamba state dimension | 256 |
| Training optimizer / scaler state | Not included |

The tensor element count includes model buffers and is not an exact trainable-parameter count. The serialized archive retains the historical container name `FEAT-32M-slim`; the public artifact filename and checksum above identify the supplied weights.

## Runtime

The supplied checkpoint requires Mamba2 and a supported Linux/NVIDIA CUDA environment. Load it through `feat.utils.loading.load_model`, which uses `torch.load(..., weights_only=True)`, validates required keys, strictly loads the model state, and freezes parameters for inference. Installation details are in [README.md](README.md).

This inference release retains the `linear` numeric encoder used by FEAT-38M. Unused RBF encoder variants and the legacy training factory are omitted; checkpoints requiring those variants are outside this release's scope.

## Scope and limitations

- Training data provenance and training recipes are not documented in this source snapshot. No training datasets or benchmark datasets are bundled.
- This model card does not claim independent benchmark reproduction. Validate predictions and evaluation settings on the intended datasets.
- Feature-relation matrices describe model attention and do not establish causal relationships.
- Imputation estimates missing values; it does not recover verified ground truth. Categorical columns require observed categories.
- Context length and feature count are constrained by available memory and compute.

## License and distribution

LDMFEAT's own code and the FEAT-38M model weights are licensed under [Apache-2.0](LICENSE), as confirmed by the maintainer. Included third-party adaptations retain the license notices listed in [NOTICE](NOTICE). Weights are distributed separately through the project's model release; see [checkpoint instructions](checkpoints/readme.md) for availability and verification.
