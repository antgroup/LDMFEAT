"""Run label-free classification or regression inference with FEAT."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feat.utils.loading import load_model
from scripts.evaluate_classification import BoostedClassifier
from scripts.evaluate_classification import preprocess_features as preprocess_classification
from scripts.evaluate_regression import BoostedRegressor
from scripts.evaluate_regression import preprocess_features as preprocess_regression


DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "FEAT-38M.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict unlabeled rows from a labeled context table with FEAT."
    )
    parser.add_argument("--task", required=True, choices=["classification", "regression"])
    parser.add_argument("--context", required=True, help="Labeled context CSV")
    parser.add_argument("--query", required=True, help="Unlabeled query CSV")
    parser.add_argument("--output", required=True, help="Destination prediction CSV")
    parser.add_argument(
        "--target",
        default=None,
        help="Target column in the context CSV (default: last column)",
    )
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--n_passes", type=int, default=5)
    parser.add_argument("--max_cells", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_shuffle", action="store_true")
    return parser.parse_args()


def validate_tables(context_df, query_df, target):
    if target not in context_df.columns:
        raise ValueError(f"Target column not found in context CSV: {target}")
    if context_df[target].isna().any():
        raise ValueError("The context target column must not contain missing values.")

    feature_columns = [column for column in context_df.columns if column != target]
    missing = [column for column in feature_columns if column not in query_df.columns]
    extra = [column for column in query_df.columns if column not in feature_columns and column != target]
    if missing or extra:
        raise ValueError(f"Context/query feature mismatch; missing={missing}, extra={extra}")
    if not feature_columns:
        raise ValueError("At least one feature column is required.")

    return context_df[feature_columns], query_df[feature_columns], context_df[target]


def main():
    args = parse_args()
    if args.n_passes < 1:
        raise ValueError("--n_passes must be at least 1")

    context_path = Path(args.context)
    query_path = Path(args.query)
    model_path = Path(args.model_path)
    for label, path in (("Context CSV", context_path), ("Query CSV", query_path), ("Checkpoint", model_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    context_df = pd.read_csv(context_path)
    query_df = pd.read_csv(query_path)
    target = args.target or str(context_df.columns[-1])
    X_context_df, X_query_df, y_context = validate_tables(context_df, query_df, target)

    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)
    model = load_model(model_path, mask_prediction=False)
    model.to(device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_shuffle = not args.no_shuffle

    if args.task == "classification":
        X_context, X_query = preprocess_classification(X_context_df, X_query_df)
        y_codes, classes = pd.factorize(y_context, sort=False)
        if len(classes) < 2:
            raise ValueError("Classification requires at least two context classes.")
        max_cells = args.max_cells or 500_000
        predictor = BoostedClassifier(model, device, max_cells=max_cells, seed=args.seed)
        probabilities = predictor.predict_proba(
            X_context,
            y_codes,
            X_query,
            n_passes=args.n_passes,
            use_shuffle=use_shuffle,
            aggregation="mean",
            strategy="shuffle",
        )
        predicted_codes = np.argmax(probabilities, axis=1)
        output_df = pd.DataFrame({"prediction": classes.take(predicted_codes)})
        for class_index, class_label in enumerate(classes):
            output_df[f"probability::{class_label}"] = probabilities[:, class_index]
    else:
        X_context, X_query = preprocess_regression(X_context_df, X_query_df)
        y_numeric = pd.to_numeric(y_context, errors="coerce").to_numpy(dtype=np.float32)
        if np.isnan(y_numeric).any():
            raise ValueError("Regression targets must be numeric and non-missing.")
        max_cells = args.max_cells or 2_000_000
        predictor = BoostedRegressor(model, device, max_cells=max_cells, seed=args.seed)
        predictions, _, _ = predictor.predict(
            X_context,
            y_numeric,
            X_query,
            n_passes=args.n_passes,
            use_shuffle=use_shuffle,
            aggregation="median",
            strategy="shuffle",
        )
        output_df = pd.DataFrame({"prediction": predictions})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"[FEAT] Wrote {len(output_df)} predictions to {output_path}")


if __name__ == "__main__":
    main()
