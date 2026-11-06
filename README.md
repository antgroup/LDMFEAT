# LDMFEAT

FEAT is a pretrained tabular foundation model for in-context classification, regression, missing-value imputation, and feature-relation analysis. This repository provides inference and evaluation code for the FEAT-38M checkpoint. Model weights are distributed separately from the source code.

Repository: [antgroup/LDMFEAT](https://github.com/antgroup/LDMFEAT).

## What is included

| Capability | Entry point | Recommended setting |
| --- | --- | --- |
| Unlabeled classification or regression | `scripts/predict.py` | classification: mean over 5 shuffled passes; regression: median over 5 shuffled passes |
| Classification benchmark evaluation | `scripts/evaluate_classification.py` | `--aggregation mean` |
| Regression benchmark evaluation | `scripts/evaluate_regression.py` | `--aggregation median` |
| Missing-value imputation | `scripts/impute.py` | automatic type detection and chunked reconstruction |
| Feature-relation heatmap | `scripts/extract_relations.py` | save as PDF, SVG, or PNG; the raw matrix is also saved as NPY |

The evaluators default to probability averaging for classification and median aggregation for regression. Select non-default strategies on validation data only; do not use test targets to choose an aggregation strategy.

## Repository layout

```text
LDMFEAT/
├── checkpoints/
│   ├── readme.md
│   └── FEAT-38M.pt  # download separately
├── feat/
│   ├── inference/
│   ├── model/
│   └── utils/
├── scripts/
│   ├── predict.py
│   ├── evaluate_classification.py
│   ├── evaluate_regression.py
│   ├── impute.py
│   └── extract_relations.py
├── LICENSE
├── README.md
├── requirements.txt
└── requirements-cuda.txt
```

## Model weights

Place `FEAT-38M.pt` in `checkpoints/`. See [checkpoint instructions](checkpoints/readme.md) for distribution status and checksum verification.

The checkpoint contains 439 model tensors and the model configuration, with 38,807,278 stored tensor elements including buffers. It has no optimizer or automatic mixed precision scaler state.

- Size: 155,418,367 bytes (about 148.22 MiB)
- SHA-256: `84acc3161c52ce67f99a43d202008b5c9f25826aff2d5b917cbfe6ced8ba6ade`
- Architecture and limitations: [MODEL_CARD.md](MODEL_CARD.md)

## Installation

Use Python 3.11 on Linux with an NVIDIA CUDA GPU. The supplied checkpoint uses Mamba2 kernels; CPU-only and macOS execution of this checkpoint are not supported by this release. Install a CUDA-enabled PyTorch build matching your system from the [official PyTorch instructions](https://pytorch.org/get-started/locally/) before installing Mamba.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja
# Install CUDA-enabled PyTorch (>=2.6,<3) for your system first.
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-cuda.txt
python -c "import torch; from mamba_ssm import Mamba2; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
```

Mamba installation depends on the PyTorch/CUDA/compiler combination; consult the [upstream installation guide](https://github.com/state-spaces/mamba#installation) if its extensions fail to build. FlashAttention is optional; the source includes a PyTorch attention fallback.

Run the commands below from the repository root. Every entry point defaults to `checkpoints/FEAT-38M.pt`; use `--model_path` to override it. Supply your own CSV files; datasets are not bundled.

## Prediction

`scripts/predict.py` performs label-free query inference from a labeled context table.

- The context CSV contains feature columns and one target column.
- The query CSV contains the same feature columns without targets.
- `--target` defaults to the final context column.
- Classification output contains the predicted label and one probability column per context class.
- Regression output contains a `prediction` column.

```bash
python scripts/predict.py \
  --task classification \
  --context data/context.csv \
  --query data/query.csv \
  --target label \
  --output outputs/classification_predictions.csv

python scripts/predict.py \
  --task regression \
  --context data/context.csv \
  --query data/query.csv \
  --target target \
  --output outputs/regression_predictions.csv
```

Inference defaults to five context-order passes with probability averaging for classification and median aggregation for regression. `--n_passes`, `--max_cells`, `--seed`, `--device`, and `--no_shuffle` are available for controlled changes.

## Benchmark evaluation

Each benchmark root uses the following layout. The target must be the final column in both CSV files.

```text
data/classification/
└── benchmark_name/
    └── dataset_name/
        ├── train.csv
        └── test.csv
```

The regression layout is identical under `data/regression/`.

```bash
python scripts/evaluate_classification.py \
  --eval_cls_data_path data/classification \
  --n_passes 5 \
  --aggregation mean

python scripts/evaluate_regression.py \
  --eval_reg_data_path data/regression \
  --n_passes 5 \
  --aggregation median
```

For dataset-level parallelism across GPUs, launch either evaluator with `torchrun`:

```bash
torchrun --nproc_per_node=4 scripts/evaluate_classification.py \
  --eval_cls_data_path data/classification
```

Classification reports accuracy, macro F1, and ROC AUC. Regression reports R², RMSE, and target-standardized RMSE. The evaluators also support context subsampling, nearest-neighbor context selection, and large-context distillation; inspect `--help` before enabling them.

## Missing-value imputation

The imputer preserves observed cells and fills only missing cells. Object or categorical columns use observed-value encoding; continuous columns use robust scaling and bounded inverse reconstruction. A categorical column must contain at least one observed value so that its vocabulary can be inferred.

```bash
python scripts/impute.py \
  --input data/masked.csv \
  --output outputs/imputed.csv
```

## Feature-relation heatmap

The relation extractor reads a labeled CSV, obtains the model's feature attention matrix, renders a heatmap, and writes the raw matrix next to the figure. For a figure named `relation_matrix.pdf`, the matrix is saved as `relation_matrix.npy`.

```bash
python scripts/extract_relations.py \
  --data_path data/table.csv \
  --target_col target \
  --task_type reg \
  --save_path outputs/relation_matrix.pdf \
  --no_title
```

Use `--task_type cls` for classification. `--no_values` is helpful for wide tables.

## Reproducibility and release scope

- Stochastic inference is controlled by `--seed` and defaults to 42.
- Benchmark hyperparameters must be selected without consulting test targets.
- The checkpoint does not contain the optimizer or scaler state needed to restore the original training session.
- Generated datasets, predictions, figures, and result tables are excluded from the source tree.
- Training pipelines and benchmark datasets are outside the scope of this release.

## License

LDMFEAT's own code and the FEAT-38M model weights are released under the Apache License 2.0. Included third-party adaptations retain their original license notices: [LimiX (Apache-2.0)](third_party/limix/README.md) and [TorchScale RMSNorm (MIT)](third_party/torchscale/README.md). See [LICENSE](LICENSE), [NOTICE](NOTICE), and [MODEL_CARD.md](MODEL_CARD.md). Separately installed dependencies retain their own licenses.

## Acknowledgments

The model implementation includes adaptations from [LimiX](https://github.com/limix-ldm-ai/LimiX), and model comments reference the [LimiX research](https://arxiv.org/abs/2509.03505). The verified upstream revision, source mapping, and original license are recorded in [third-party attribution](third_party/limix/README.md). Sequence mixing uses the separately installed [Mamba](https://github.com/state-spaces/mamba) implementation.
