# LimiX source attribution

Parts of FEAT's encoder, attention-layer, transformer, and model-loading implementations are adapted from [LimiX](https://github.com/limix-ldm-ai/LimiX).

The verified upstream source baseline is commit [`76f81eb6ecda88ea3a40e03e098e70104a916ddf`](https://github.com/limix-ldm-ai/LimiX/tree/76f81eb6ecda88ea3a40e03e098e70104a916ddf), dated 2025-11-20. Its [Apache-2.0 license](https://github.com/limix-ldm-ai/LimiX/blob/76f81eb6ecda88ea3a40e03e098e70104a916ddf/LICENSE.txt), including `Copyright 2025 StableAI`, is preserved verbatim in [LICENSE.txt](LICENSE.txt).

| FEAT file | Upstream source |
| --- | --- |
| `feat/model/encoders.py` | `model/encoders.py` |
| `feat/model/layer.py` | `model/layer.py` |
| `feat/model/feat_model.py` | `model/transformer.py` |
| `feat/utils/loading.py` | `utils/loading.py` |

The retained source snapshot matches 44 of the 50 upstream files byte for byte, including the license, encoder, attention layer, and README. It also contains local modifications. This commit identifies a reproducible source baseline; it does not imply that every file in FEAT is an unchanged copy of that revision.

FEAT modifications include Mamba2/linear sample mixing, positional encoding, numerical handling, inference optimizations, checkpoint loading, and removal of unused helpers and encoder variants. Modified source files carry attribution notices. The pinned upstream tree has no separate NOTICE file.

License SHA-256: `e2b41d2f855c6ccdeb357d23bcfa8aad870b7c1b4b93523ef65791596b1afb79`.
