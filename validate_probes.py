from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/interim/probes_dev.csv")

REQUIRED_COLUMNS = [
    "pair_id",
    "family",
    "source_type",
    "original_en",
    "counterfactual_en",
    "expected_original",
    "expected_counterfactual",
    "critical_change",
]

VALID_LABELS = {-2, -1, 0, 1, 2}


def load_data() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    return pd.read_csv(INPUT_PATH)


def validate_columns(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )


def validate_missing_values(df: pd.DataFrame) -> None:
    missing = df[REQUIRED_COLUMNS].isnull().sum()

    if missing.sum() > 0:
        raise ValueError(
            f"Missing values found:\n{missing[missing > 0]}"
        )


def validate_ids(df: pd.DataFrame) -> None:
    if df["pair_id"].duplicated().any():
        duplicates = df.loc[
            df["pair_id"].duplicated(),
            "pair_id"
        ].tolist()

        raise ValueError(
            f"Duplicate pair IDs: {duplicates}"
        )


def validate_text_pairs(df: pd.DataFrame) -> None:
    identical = (
        df["original_en"].str.strip()
        == df["counterfactual_en"].str.strip()
    )

    if identical.any():
        bad_ids = df.loc[
            identical,
            "pair_id"
        ].tolist()

        raise ValueError(
            f"Identical original/counterfactual text: {bad_ids}"
        )


def validate_labels(df: pd.DataFrame) -> None:
    original_labels = set(df["expected_original"])
    counterfactual_labels = set(df["expected_counterfactual"])

    invalid_original = original_labels - VALID_LABELS
    invalid_counterfactual = counterfactual_labels - VALID_LABELS

    if invalid_original:
        raise ValueError(
            f"Invalid original labels: {invalid_original}"
        )

    if invalid_counterfactual:
        raise ValueError(
            f"Invalid counterfactual labels: {invalid_counterfactual}"
        )


def add_expected_delta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["expected_delta"] = (
        df["expected_counterfactual"]
        - df["expected_original"]
    )

    return df


def print_summary(df: pd.DataFrame) -> None:
    print("VALIDATION PASSED")
    print("-" * 50)
    print(f"Pairs: {len(df)}")
    print(f"Families: {df['family'].nunique()}")
    print()

    print("Pairs per family:")
    print(
        df["family"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Expected counterfactual effects:")

    print(
        df[
            [
                "pair_id",
                "expected_original",
                "expected_counterfactual",
                "expected_delta",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    df = load_data()

    validate_columns(df)
    validate_missing_values(df)
    validate_ids(df)
    validate_text_pairs(df)
    validate_labels(df)

    df = add_expected_delta(df)

    print_summary(df)


if __name__ == "__main__":
    main()