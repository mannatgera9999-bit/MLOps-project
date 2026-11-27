
from pathlib import Path

import pandas as pd


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET = "metres_per_shift"
LEAKAGE_COLUMN = "metres_checked_at_folding"
SENTINEL_COLUMN = "power_cuts_min"
SENTINEL_VALUE = -999

# The 15 target values identified during data validation.
# Each entry maps record_id to its original, incorrectly scaled target.
TARGET_CORRECTIONS = {
    "R00550": 701.0,
    "R00642": 637.0,
    "R00178": 524.0,
    "R00325": 508.0,
    "R00638": 444.0,
    "R00067": 439.0,
    "R00078": 382.0,
    "R00338": 379.0,
    "R00275": 351.0,
    "R00726": 348.0,
    "R00148": 342.0,
    "R00163": 342.0,
    "R00693": 316.0,
    "R00690": 303.0,
    "R00181": 189.0,
}

# Normalize fabric spelling variations.
FABRIC_MAPPING = {
    "sil.": "silk",
    "cot.": "cotton",
    "pol.": "polyester",
}


def load_data():
    """Load the original CSV files without modifying them."""
    train = pd.read_csv(RAW_DIR / "train.csv")
    later = pd.read_csv(RAW_DIR / "later.csv")
    holdout = pd.read_csv(RAW_DIR / "holdout_features.csv")

    return train, later, holdout


def normalize_fabric(df):
    """Standardize fabric names, spaces and letter case."""
    if "fabric" in df.columns:
        df["fabric"] = (
            df["fabric"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace(FABRIC_MAPPING)
        )

    return df


def correct_target_values(train):
    """Correct the identified 10x target-entry errors."""
    if TARGET not in train.columns or "record_id" not in train.columns:
        raise ValueError("Training data must contain record_id and target columns.")

    corrected_count = 0

    for record_id, original_value in TARGET_CORRECTIONS.items():
        mask = train["record_id"] == record_id

        if not mask.any():
            raise ValueError(f"Expected record_id not found: {record_id}")

        current_value = train.loc[mask, TARGET]

        if len(current_value) != 1:
            raise ValueError(f"Expected exactly one row for {record_id}.")

        if float(current_value.iloc[0]) != original_value:
            raise ValueError(
                f"Unexpected target value for {record_id}: "
                f"{current_value.iloc[0]}. Check the original data."
            )

        train.loc[mask, TARGET] = original_value / 10
        corrected_count += 1

    print(f"Corrected target entries: {corrected_count}")
    return train


def clean_datasets(train, later, holdout):
    """Apply consistent cleaning using training data for imputation."""

    # 1. Correct target-entry errors in training data.
    train = correct_target_values(train)

    # 2. Normalize fabric categories in all datasets.
    train = normalize_fabric(train)
    later = normalize_fabric(later)
    holdout = normalize_fabric(holdout)

    # 3. Convert the sentinel value into missing data.
    for df in (train, later, holdout):
        if SENTINEL_COLUMN in df.columns:
            df[SENTINEL_COLUMN] = pd.to_numeric(
                df[SENTINEL_COLUMN], errors="coerce"
            )
            df[SENTINEL_COLUMN] = df[SENTINEL_COLUMN].replace(
                SENTINEL_VALUE, float("nan")
            )

    # 4. Calculate the imputation value from training data only.
    power_cuts_median = train[SENTINEL_COLUMN].median()

    if pd.isna(power_cuts_median):
        raise ValueError("Cannot calculate training median for power_cuts_min.")

    # Apply the same training median to all datasets.
    for df in (train, later, holdout):
        if SENTINEL_COLUMN in df.columns:
            df[SENTINEL_COLUMN] = df[SENTINEL_COLUMN].fillna(
                power_cuts_median
            )
    # 5. Remove the post-production leakage column.
    for df in (train, later, holdout):
        df.drop(columns=[LEAKAGE_COLUMN], errors="ignore", inplace=True)

    print(f"Training median for power_cuts_min: {power_cuts_median}")
    print("Removed leakage column:", LEAKAGE_COLUMN)

    return train, later, holdout


def save_datasets(train, later, holdout):
    """Save cleaned datasets separately from the original CSV files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(PROCESSED_DIR / "train_clean.csv", index=False)
    later.to_csv(PROCESSED_DIR / "later_clean.csv", index=False)
    holdout.to_csv(PROCESSED_DIR / "holdout_clean.csv", index=False)

    print("\nCleaned datasets saved:")
    print("Training:", train.shape)
    print("Later:", later.shape)
    print("Holdout:", holdout.shape)


def main():
    train, later, holdout = load_data()

    print("Original dataset shapes:")
    print("Training:", train.shape)
    print("Later:", later.shape)
    print("Holdout:", holdout.shape)

    train, later, holdout = clean_datasets(train, later, holdout)

    save_datasets(train, later, holdout)


if __name__ == "__main__":
    main()
