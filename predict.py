"""Batch scoring: load the saved model and write magnet-temperature
predictions for a CSV of records.

The model is the full pipeline, so the same imputation and preprocessing used
in training are applied here, and records with a missing field are still scored.
"""

import argparse

import joblib
import pandas as pd

from src.config import FEATURES, MODELS_DIR


def predict(df, model):
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")
    out = df.copy()
    out["pm_pred"] = model.predict(df[FEATURES])
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Predict permanent magnet temperature for a batch of records.")
    parser.add_argument("--input", required=True, help="CSV of records to score")
    parser.add_argument("--output", required=True, help="path for the predictions CSV")
    parser.add_argument("--model", default=str(MODELS_DIR / "model.joblib"),
                        help="path to the saved model")
    args = parser.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.input)
    predict(df, model).to_csv(args.output, index=False)
    print(f"wrote {len(df)} predictions to {args.output}")


if __name__ == "__main__":
    main()
