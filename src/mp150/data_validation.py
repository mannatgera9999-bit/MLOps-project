
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_FILE = DATA_DIR / "train.csv"
LATER_FILE = DATA_DIR / "later.csv"
HOLDOUT_FILE = DATA_DIR / "holdout_features.csv"


def load_datasets():
    train = pd.read_csv(TRAIN_FILE)
    later = pd.read_csv(LATER_FILE)
    holdout = pd.read_csv(HOLDOUT_FILE)
    return train, later, holdout


def check_leakage(train, holdout):
    print("\n" + "=" * 60)
    print("CHECK 1: POSSIBLE DATA LEAKAGE")
    print("=" * 60)

    train_only = sorted(set(train.columns) - set(holdout.columns))
    print("Columns in training but absent from holdout:")
    print(train_only)


def check_unit_and_range_anomalies(train, later):
    print("\n" + "=" * 60)
    print("CHECK 2: NUMERICAL RANGE AND POSSIBLE UNIT ISSUES")
    print("=" * 60)

    numeric_columns = train.select_dtypes(include="number").columns

    for column in numeric_columns:
        print(f"\nColumn: {column}")
        print("Training range:", train[column].min(), "to", train[column].max())
        if column in later.columns:
            print("Later range:", later[column].min(), "to", later[column].max())

    print("\nLargest target values in training:")
    print(
        train.nlargest(20, "metres_per_shift")[
            ["record_id", "date", "shed", "fabric",
             "metres_checked_at_folding", "metres_per_shift"]
        ].to_string(index=False)
    )


def check_sentinel_values(train):
    print("\n" + "=" * 60)
    print("CHECK 3: POSSIBLE SENTINEL VALUES")
    print("=" * 60)

    for column in train.select_dtypes(include="number").columns:
        unusual = train[train[column] < 0]
        if not unusual.empty:
            print(f"\nNegative values in {column}:")
            print(unusual[["record_id", "date", column]].to_string(index=False))


def check_duplicates(train):
    print("\n" + "=" * 60)
    print("CHECK 4: DUPLICATE ROWS")
    print("=" * 60)

    print("Exact duplicate rows:", train.duplicated().sum())
    print("Duplicate record IDs:", train["record_id"].duplicated().sum())


def check_stuck_sensor(train):
    print("\n" + "=" * 60)
    print("CHECK 5: POSSIBLE STUCK SENSOR")
    print("=" * 60)

    numeric_columns = train.select_dtypes(include="number").columns

    for column in numeric_columns:
        counts = train[column].value_counts(dropna=False)
        print(f"\n{column}:")
        print("Distinct values:", train[column].nunique())
        print("Most frequent readings:")
        print(counts.head(10).to_string())


def check_target_values(train):
    print("\n" + "=" * 60)
    print("CHECK 6: TARGET VALUE INVESTIGATION")
    print("=" * 60)

    target = "metres_per_shift"

    print("Target summary:")
    print(train[target].describe())

    print("\nLargest 20 target values:")
    print(
        train.nlargest(20, target)[
            ["record_id", "date", "shed", target]
        ].to_string(index=False)
    )


def check_spelling_variants(train, later):
    print("\n" + "=" * 60)
    print("CHECK 7: SPELLING VARIANTS")
    print("=" * 60)

    for column in train.select_dtypes(include="object").columns:
        print(f"\nColumn: {column}")
        print("Training values and counts:")
        print(train[column].value_counts(dropna=False).to_string())

        if column in later.columns:
            print("Later dataset values and counts:")
            print(later[column].value_counts(dropna=False).to_string())


def check_unit_mixup_by_shed(train):
    print("\n" + "=" * 60)
    print("TARGETED CHECK: UNIT MIX-UP BY SHED")
    print("=" * 60)

    numeric_columns = train.select_dtypes(include="number").columns

    for column in numeric_columns:
        print(f"\nColumn: {column}")
        print(
            train.groupby("shed")[column]
            .agg(["count", "min", "mean", "median", "max"])
            .round(3)
            .to_string()
        )



def check_stuck_sensor_by_date(train):
    print("\n" + "=" * 60)
    print("TARGETED CHECK: STUCK SENSOR BY SHED")
    print("=" * 60)

    df = train.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["shed", "date", "record_id"])

    numeric_columns = [
        col for col in df.select_dtypes(include="number").columns
        if col not in ["record_id"]
    ]

    for shed, shed_data in df.groupby("shed"):
        print(f"\n{shed}")

        for column in numeric_columns:
            values = shed_data[column]

            # Identify consecutive identical readings
            run_id = values.ne(values.shift()).cumsum()

            runs = (
                shed_data.assign(run_id=run_id)
                .groupby("run_id")
                .agg(
                    reading=(column, "first"),
                    consecutive_records=(column, "size"),
                    start_date=("date", "min"),
                    end_date=("date", "max"),
                )
            )

            suspicious = runs[runs["consecutive_records"] >= 5]

            if not suspicious.empty:
                print(f"\nPossible stuck sensor: {column}")
                print(suspicious.to_string(index=False))



def check_target_typos(train):
    print("\n" + "=" * 60)
    print("TARGETED CHECK: TARGET VALUES 10 TIMES TOO LARGE")
    print("=" * 60)

    target = "metres_per_shift"

    suspicious = train[train[target] >= 100].copy()
    suspicious["possible_corrected_value"] = suspicious[target] / 10

    print("Rows with target >= 100:", len(suspicious))
    print(
        suspicious[
            ["record_id", "date", "shed", target,
             "possible_corrected_value",
             "metres_checked_at_folding"]
        ].sort_values(target, ascending=False).to_string(index=False)
    )


def investigate_unit_patterns(train, later):
    print("\n" + "=" * 60)
    print("UNIT PATTERN INVESTIGATION")
    print("=" * 60)

    numeric_columns = [
        col for col in train.select_dtypes(include="number").columns
        if col in later.columns
    ]

    for column in numeric_columns:
        print(f"\nColumn: {column}")

        print("Training statistics:")
        print(train[column].describe().round(3).to_string())

        print("\nLater statistics:")
        print(later[column].describe().round(3).to_string())

        print("\nTraining statistics by shed:")
        print(
            train.groupby("shed")[column]
            .agg(["min", "median", "mean", "max"])
            .round(3).to_string()
        )


def check_stuck_sensor_by_calendar_day(train):
    print("\n" + "=" * 60)
    print("STUCK SENSOR CHECK: CONSECUTIVE CALENDAR DAYS")
    print("=" * 60)

    df = train.copy()
    df["date"] = pd.to_datetime(df["date"])

    numeric_columns = [
        col for col in df.select_dtypes(include="number").columns
        if col not in ["record_id"]
    ]

    for shed, shed_data in df.groupby("shed"):
        print(f"\n{shed}")

        for column in numeric_columns:
            daily = (
                shed_data.groupby("date")[column]
                .agg(["nunique", "first"])
                .sort_index()
            )

            # A date is considered stable if all its readings
            # for this column have the same value.
            daily["stable"] = daily["nunique"] == 1

            stable = daily[daily["stable"]].copy()
            if stable.empty:
                continue

            # Identify consecutive calendar-day sequences
            # having the same sensor reading.
            stable["reading"] = stable["first"]
            stable["date_gap"] = stable.index.to_series().diff().dt.days
            stable["new_run"] = (
                stable["date_gap"].ne(1)
                | stable["reading"].ne(stable["reading"].shift())
            )
            stable["run_id"] = stable["new_run"].cumsum()

            runs = (
                stable.groupby("run_id")
                .agg(
                    reading=("reading", "first"),
                    start_date=("first", lambda x: x.index.min()),
                    end_date=("first", lambda x: x.index.max()),
                    days=("reading", "size"),
                )
            )

            suspicious = runs[runs["days"] >= 14]

            if not suspicious.empty:
                print(f"\nPossible stuck sensor: {column}")
                print(suspicious.to_string(index=False))


def main():
    train, later, holdout = load_datasets()

    check_leakage(train, holdout)
    check_unit_and_range_anomalies(train, later)
    check_sentinel_values(train)
    check_duplicates(train)
    check_stuck_sensor(train)
    check_target_values(train)
    check_spelling_variants(train, later)
    check_unit_mixup_by_shed(train)
    check_target_typos(train)
    investigate_unit_patterns(train, later)
    check_stuck_sensor_by_calendar_day(train)


if __name__ == "__main__":
    main()
