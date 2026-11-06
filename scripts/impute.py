"""Command-line missing-value imputation with the FEAT checkpoint."""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feat.inference.imputer import FEATImputer
from feat.utils.loading import load_model


DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "FEAT-38M.pt"


def main():
    parser = argparse.ArgumentParser(
        description="Fill missing cells in a CSV file with FEAT."
    )
    parser.add_argument("--input", required=True, help="CSV containing missing cells")
    parser.add_argument("--output", required=True, help="Destination CSV")
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, e.g. cuda, cuda:0, cpu, or auto",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"FEAT checkpoint not found: {args.model_path}")

    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)

    masked_df = pd.read_csv(args.input)
    if not masked_df.isna().any().any():
        print("[FEAT] Input has no missing cells; writing an unchanged copy.")

    model = load_model(args.model_path, mask_prediction=True)
    model.to(device)
    imputer = FEATImputer(model, device=device)
    imputed_df = imputer.impute(masked_df)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    imputed_df.to_csv(args.output, index=False)
    print(f"[FEAT] Imputed CSV saved to: {args.output}")


if __name__ == "__main__":
    main()
