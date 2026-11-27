
from pathlib import Path
import hashlib

import yaml
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "params.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

DATA_PATH = PROJECT_ROOT / config["data"]["train_path"]
RAW_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

TARGET = config["data"]["target"]
ID_COLUMN = config["data"]["id_column"]
DATE_COLUMN = config["data"]["date_column"]

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


# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------

def load_training_data():
    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return X, y


# --------------------------------------------------
# 3. Build preprocessing pipeline
# --------------------------------------------------

def build_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


# --------------------------------------------------
# 4. Create candidate regression models
# --------------------------------------------------

def get_candidate_models():

    random_forest = RandomForestRegressor(
        n_estimators=config["model"]["random_forest"]["n_estimators"],
        max_depth=config["model"]["random_forest"]["max_depth"],
        min_samples_split=config["model"]["random_forest"]["min_samples_split"],
        random_state=config["random_state"],
        n_jobs=-1,
    )

    gradient_boosting = GradientBoostingRegressor(
        n_estimators=config["model"]["gradient_boosting"]["n_estimators"],
        learning_rate=config["model"]["gradient_boosting"]["learning_rate"],
        max_depth=config["model"]["gradient_boosting"]["max_depth"],
        random_state=config["random_state"],
    )

    return {
        "RandomForest": random_forest,
        "GradientBoosting": gradient_boosting,
    }


# --------------------------------------------------
# 5. Calculate original training-data SHA-256
# --------------------------------------------------

def calculate_data_sha256():

    sha256 = hashlib.sha256()

    with open(RAW_TRAIN_PATH, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# --------------------------------------------------
# 6. Train and compare models using MLflow
# --------------------------------------------------

def train_models():

    X, y = load_training_data()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["random_state"],
    )

    data_sha256 = calculate_data_sha256()

    tracking_config = config["mlflow"]

    mlflow.set_tracking_uri(
        tracking_config["tracking_uri"]
    )

    mlflow.set_experiment(
        tracking_config["experiment_name"]
    )

    candidates = get_candidate_models()

    results = []

    print("\nStarting model experiments...\n")

    for model_name, estimator in candidates.items():

        preprocessor = build_preprocessor()

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", estimator),
            ]
        )

        with mlflow.start_run(run_name=model_name) as run:

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_valid)

            mae = mean_absolute_error(
                y_valid,
                predictions,
            )

            mlflow.log_param("model_name", model_name)
            mlflow.log_param(
                "random_state",
                config["random_state"],
            )
            mlflow.log_param(
                "test_size",
                config["model"]["test_size"],
            )
            mlflow.log_param(
                "features",
                ",".join(FEATURES),
            )
            mlflow.log_param(
                "data_sha256",
                data_sha256,
            )

            mlflow.log_metric("validation_mae", mae)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            results.append(
                {
                    "model_name": model_name,
                    "mae": mae,
                    "run_id": run.info.run_id,
                    "pipeline": pipeline,
                }
            )

            print(f"Model: {model_name}")
            print(f"Validation MAE: {mae:.4f}")
            print(f"Run ID: {run.info.run_id}")
            print("-" * 45)

    # Select the model with the lowest validation MAE.
    best_result = min(results, key=lambda item: item["mae"])

    print("\nBest model selected:")
    print("Model:", best_result["model_name"])
    print("Validation MAE:", round(best_result["mae"], 4))
    print("Run ID:", best_result["run_id"])

    # Register only the best candidate.
    with mlflow.start_run(run_name="register_best_model"):

        mlflow.log_param(
            "selected_model",
            best_result["model_name"],
        )

        mlflow.log_metric(
            "validation_mae",
            best_result["mae"],
        )

        mlflow.log_param(
            "data_sha256",
            data_sha256,
        )

        mlflow.sklearn.log_model(
            sk_model=best_result["pipeline"],
            name="model",
            serialization_format="cloudpickle",
            registered_model_name=tracking_config[
                "registered_model_name"
            ],
        )

    print("\nBest candidate registered in MLflow.")
    print(
        "Registered model name:",
        tracking_config["registered_model_name"],
    )


if __name__ == "__main__":
    train_models()
