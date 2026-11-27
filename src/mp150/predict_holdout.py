
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "params.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

HOLDOUT_PATH = PROJECT_ROOT / config["data"]["holdout_path"]
OUTPUT_PATH = PROJECT_ROOT / "predictions" / "holdout_predictions.csv"

MODEL_NAME = config["mlflow"]["registered_model_name"]

FEATURES = [
    "loom_age_yr",
    "thread_count",
    "power_cuts_min",
    "weaver_exp_yr",
    "humidity_pct",
    "shift_hours",
    "fabric",
    "shed",
]


def main():

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    # Load September holdout data.
    holdout = pd.read_csv(HOLDOUT_PATH)

    # Validate required columns.
    required_columns = ["record_id"] + FEATURES

    missing_columns = [
        column for column in required_columns
        if column not in holdout.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing holdout columns: {missing_columns}"
        )

    if holdout["record_id"].isna().any():
        raise ValueError("Holdout contains missing record IDs.")

    if holdout["record_id"].duplicated().any():
        raise ValueError("Holdout contains duplicate record IDs.")

    # Load the current champion dynamically.
    model_uri = f"models:/{MODEL_NAME}@champion"
    model = mlflow.sklearn.load_model(model_uri)

    # Generate predictions.
    predictions = model.predict(holdout[FEATURES])

    # Create the exact required output format.
    output = pd.DataFrame(
        {
            "record_id": holdout["record_id"],
            "prediction": predictions,
        }
    )

    # Validate output.
    if len(output) != len(holdout):
        raise ValueError("Prediction count does not match holdout rows.")

    if output["prediction"].isna().any():
        raise ValueError("Predictions contain missing values.")

    # Save output.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)

    print("Champion model:", model_uri)
    print("Holdout records:", len(holdout))
    print("Predictions generated:", len(output))
    print("Output columns:", list(output.columns))
    print("Saved to:", OUTPUT_PATH)

    print("\nFirst five predictions:")
    print(output.head().to_string(index=False))


if __name__ == "__main__":
    main()
