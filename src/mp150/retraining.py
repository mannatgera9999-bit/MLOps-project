
from pathlib import Path
import hashlib

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "params.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

TRAIN_PATH = PROJECT_ROOT / config["data"]["train_path"]
LATER_PATH = PROJECT_ROOT / config["data"]["later_path"]
RAW_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

TARGET = config["data"]["target"]
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


def calculate_data_sha256():
    """Calculate SHA-256 of the original training CSV."""

    sha256 = hashlib.sha256()

    with open(RAW_TRAIN_PATH, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def build_v2_pipeline():

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    model_config = config["model"]["gradient_boosting"]

    estimator = GradientBoostingRegressor(
        n_estimators=model_config["n_estimators"],
        learning_rate=model_config["learning_rate"],
        max_depth=model_config["max_depth"],
        random_state=config["random_state"],
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ]
    )


def main():

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(
        config["mlflow"]["experiment_name"]
    )

    # Load cleaned datasets.
    train = pd.read_csv(TRAIN_PATH)
    later = pd.read_csv(LATER_PATH)

    # Sort later records chronologically.
    later["date"] = pd.to_datetime(later["date"])
    later = later.sort_values("date").reset_index(drop=True)

    # Split later data into first 70% and last 30%.
    split_index = int(len(later) * 0.70)

    later_train = later.iloc[:split_index].copy()
    later_check = later.iloc[split_index:].copy()

    print("Original cleaned training records:", len(train))
    print("Later records:", len(later))
    print("First 70% of later data:", len(later_train))
    print("Last 30% of later data:", len(later_check))

    # Combine original training data and first 70% of later data.
    combined_train = pd.concat(
        [train, later_train],
        ignore_index=True,
    )

    X_train = combined_train[FEATURES]
    y_train = combined_train[TARGET]

    X_check = later_check[FEATURES]
    y_check = later_check[TARGET]

    # Load v1 from the existing champion alias.
    v1_uri = f"models:/{MODEL_NAME}@champion"
    v1_model = mlflow.sklearn.load_model(v1_uri)

    # Evaluate v1 on the final 30%.
    v1_predictions = v1_model.predict(X_check)

    v1_mae = mean_absolute_error(
        y_check,
        v1_predictions,
    )

    # Train v2 using combined training data.
    v2_model = build_v2_pipeline()
    v2_model.fit(X_train, y_train)

    # Evaluate v2 on the same final 30%.
    v2_predictions = v2_model.predict(X_check)

    v2_mae = mean_absolute_error(
        y_check,
        v2_predictions,
    )

    # Calculate percentage improvement.
    improvement_pct = (
        (v1_mae - v2_mae) / v1_mae
    ) * 100

    promote = improvement_pct >= 12.0

    print("\nRETRAINING RESULTS")
    print("=" * 50)
    print(f"v1 MAE on final 30%: {v1_mae:.4f}")
    print(f"v2 MAE on final 30%: {v2_mae:.4f}")
    print(f"MAE improvement: {improvement_pct:.2f}%")

    if promote:
        print("\nPromotion decision: PROMOTE v2")
    else:
        print("\nPromotion decision: KEEP v1")

    # Record the experiment and register v2.
    with mlflow.start_run(run_name="MP150_v2_retraining") as run:

        mlflow.log_param("model_name", "GradientBoosting")
        mlflow.log_param("training_records", len(combined_train))
        mlflow.log_param("later_training_records", len(later_train))
        mlflow.log_param("check_records", len(later_check))
        mlflow.log_param("promotion_threshold_pct", 12.0)
        mlflow.log_param("data_sha256", calculate_data_sha256())

        mlflow.log_metric("v1_check_mae", v1_mae)
        mlflow.log_metric("v2_check_mae", v2_mae)
        mlflow.log_metric("improvement_pct", improvement_pct)

        model_info = mlflow.sklearn.log_model(
            sk_model=v2_model,
            name="model",
            serialization_format="cloudpickle",
            registered_model_name=MODEL_NAME,
        )

        print("\nV2 registered successfully.")
        print("Model URI:", model_info.model_uri)

    # Update champion only when the promotion rule is satisfied.
    client = mlflow.tracking.MlflowClient()

    if promote:
        version = model_info.registered_model_version

        if version is None:
            raise RuntimeError(
                "Could not determine the registered v2 version."
            )

        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            str(version),
        )

        print("Champion alias updated to version:", version)

    else:
        champion = client.get_model_version_by_alias(
            MODEL_NAME,
            "champion",
        )

        print(
            "Champion alias remains at version:",
            champion.version,
        )

    print("\nRetraining experiment completed.")


if __name__ == "__main__":
    main()
