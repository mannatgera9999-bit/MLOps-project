
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "params.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

TRAIN_PATH = PROJECT_ROOT / config["data"]["train_path"]
LATER_PATH = PROJECT_ROOT / config["data"]["later_path"]

MODEL_NAME = config["mlflow"]["registered_model_name"]

NUMERIC_FEATURES = [
    "loom_age_yr",
    "thread_count",
    "power_cuts_min",
    "weaver_exp_yr",
    "humidity_pct",
    "shift_hours",
]

CATEGORICAL_FEATURES = ["fabric", "shed"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = config["data"]["target"]


def calculate_psi(reference, current, bins=10):
    """
    Calculate PSI using reference-data quantile bins.
    A small epsilon avoids division by zero.
    """

    reference = pd.Series(reference).dropna().astype(float)
    current = pd.Series(current).dropna().astype(float)

    if reference.empty or current.empty:
        raise ValueError("PSI cannot be calculated on empty data.")

    # Create bin edges from the reference (training) distribution.
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))

    # Handle a constant reference feature.
    if len(edges) < 2:
        return 0.0

    # Extend the outer edges to include all future values.
    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_bins = pd.cut(
        reference,
        bins=edges,
        include_lowest=True,
    )

    current_bins = pd.cut(
        current,
        bins=edges,
        include_lowest=True,
    )

    reference_pct = (
        reference_bins.value_counts(sort=False) / len(reference)
    )

    current_pct = (
        current_bins.value_counts(sort=False) / len(current)
    )

    epsilon = 1e-6

    reference_pct = reference_pct.clip(lower=epsilon)
    current_pct = current_pct.clip(lower=epsilon)

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def calculate_feature_drift(train, later):
    """Calculate PSI for all six numerical input features."""

    results = {}

    print("\nPOPULATION STABILITY INDEX (PSI)")
    print("=" * 50)

    for feature in NUMERIC_FEATURES:

        psi = calculate_psi(
            train[feature],
            later[feature],
            bins=10,
        )

        results[feature] = psi

        print(f"{feature:20s}: {psi:.6f}")

    most_drifted = max(results, key=results.get)

    print("\nMost drifted feature:", most_drifted)
    print("Highest PSI:", round(results[most_drifted], 6))

    return results, most_drifted


def evaluate_v1_by_shed(later):
    """Evaluate the registered v1 champion separately for each shed."""

    model_uri = f"models:/{MODEL_NAME}@champion"

    print("\nLOADING REGISTERED V1 MODEL")
    print("=" * 50)

    model = mlflow.sklearn.load_model(model_uri)

    X_later = later[FEATURES]
    y_later = later[TARGET]

    predictions = model.predict(X_later)

    evaluation_data = later[["shed"]].copy()
    evaluation_data["actual"] = y_later.to_numpy()
    evaluation_data["prediction"] = predictions

    evaluation_data["absolute_error"] = (
        evaluation_data["actual"] - evaluation_data["prediction"]
    ).abs()

    mae_by_group = {}

    print("\nV1 MAE BY SHED")
    print("=" * 50)

    for shed, group in evaluation_data.groupby("shed"):

        mae = mean_absolute_error(
            group["actual"],
            group["prediction"],
        )

        mae_by_group[shed] = float(mae)

        print(
            f"{shed}: MAE = {mae:.4f} "
            f"(records: {len(group)})"
        )

    worst_group = max(mae_by_group, key=mae_by_group.get)

    print("\nWorst-served group:", worst_group)
    print("Highest group MAE:", round(mae_by_group[worst_group], 4))

    return mae_by_group, worst_group


def main():

    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)

    print("Training data:", train.shape)
    print("Later data:", later.shape)

    # 1. Feature drift.
    psi_results, most_drifted = calculate_feature_drift(
        train,
        later,
    )

    # 2. Evaluate v1 separately for each shed.
    mae_by_group, worst_group = evaluate_v1_by_shed(later)

    print("\nMONITORING SUMMARY")
    print("=" * 50)
    print("Most drifted feature:", most_drifted)
    print("Worst-served group:", worst_group)


if __name__ == "__main__":
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    main()
