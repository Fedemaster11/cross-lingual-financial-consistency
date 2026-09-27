from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/final/probes_en.csv")

EXPECTED_FAMILY_COUNTS = {
    "profit_direction": 8,
    "loss_direction": 8,
    "cost_direction": 8,
    "revenue_direction": 8,
    "earnings_expectations": 8,
    "guidance": 8,
    "margin": 6,
    "contract_realization": 6,
}


def fail(message: str) -> None:
    raise ValueError(message)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("=" * 70)
    print("FINAL ENGLISH PROBE AUDIT")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Row count
    # ---------------------------------------------------------

    if len(df) != 60:
        fail(
            f"Expected 60 rows, found {len(df)}"
        )

    print("PASS: 60 probes")

    # ---------------------------------------------------------
    # 2. Required columns
    # ---------------------------------------------------------

    required_columns = {
        "pair_id",
        "family",
        "source_type",
        "original_direction",
        "original_en",
        "counterfactual_en",
        "expected_original",
        "expected_counterfactual",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        fail(
            f"Missing columns: {sorted(missing_columns)}"
        )

    print("PASS: required columns")

    # ---------------------------------------------------------
    # 3. Missing values
    # ---------------------------------------------------------

    missing = df[
        list(required_columns)
    ].isna().sum()

    if missing.sum() != 0:
        fail(
            "Missing values found:\n"
            f"{missing[missing > 0]}"
        )

    print("PASS: no missing values")

    # ---------------------------------------------------------
    # 4. Pair IDs
    # ---------------------------------------------------------

    if df["pair_id"].duplicated().any():
        duplicates = df.loc[
            df["pair_id"].duplicated(),
            "pair_id",
        ].tolist()

        fail(
            f"Duplicate pair IDs: {duplicates}"
        )

    expected_ids = {
        f"P{i:03d}"
        for i in range(1, 61)
    }

    actual_ids = set(df["pair_id"])

    if actual_ids != expected_ids:
        fail(
            "Pair IDs are not exactly P001-P060."
        )

    print("PASS: pair IDs P001-P060 unique")

    # ---------------------------------------------------------
    # 5. Original != counterfactual
    # ---------------------------------------------------------

    identical = (
        df["original_en"].str.strip()
        == df["counterfactual_en"].str.strip()
    )

    if identical.any():
        bad_ids = df.loc[
            identical,
            "pair_id",
        ].tolist()

        fail(
            f"Identical text pairs: {bad_ids}"
        )

    print(
        "PASS: original and counterfactual "
        "are different"
    )

    # ---------------------------------------------------------
    # 6. Duplicate complete sentences
    # ---------------------------------------------------------

    all_sentences = pd.concat(
        [
            df["original_en"],
            df["counterfactual_en"],
        ],
        ignore_index=True,
    )

    duplicated_sentences = (
        all_sentences[
            all_sentences.duplicated(
                keep=False
            )
        ]
        .drop_duplicates()
        .tolist()
    )

    if duplicated_sentences:
        fail(
            "Duplicate sentence strings found:\n"
            + "\n".join(duplicated_sentences)
        )

    print("PASS: no duplicate sentences")

    # ---------------------------------------------------------
    # 7. Direction balance
    # ---------------------------------------------------------

    direction_counts = (
        df["original_direction"]
        .value_counts()
        .to_dict()
    )

    if direction_counts != {
        "positive": 30,
        "negative": 30,
    }:
        fail(
            "Direction balance incorrect: "
            f"{direction_counts}"
        )

    print(
        "PASS: 30 positive-original / "
        "30 negative-original"
    )

    # ---------------------------------------------------------
    # 8. Expected labels must agree with direction
    # ---------------------------------------------------------

    for _, row in df.iterrows():

        if row["original_direction"] == "positive":
            expected = (1, -1)

        elif row["original_direction"] == "negative":
            expected = (-1, 1)

        else:
            fail(
                f"{row['pair_id']}: invalid "
                f"original_direction "
                f"{row['original_direction']}"
            )

        actual = (
            int(row["expected_original"]),
            int(row["expected_counterfactual"]),
        )

        if actual != expected:
            fail(
                f"{row['pair_id']}: "
                f"direction={row['original_direction']} "
                f"but labels={actual}, "
                f"expected={expected}"
            )

    print(
        "PASS: expected labels agree "
        "with original_direction"
    )

    # ---------------------------------------------------------
    # 9. Delta balance
    # ---------------------------------------------------------

    delta = (
        df["expected_counterfactual"]
        - df["expected_original"]
    )

    delta_counts = (
        delta.value_counts()
        .sort_index()
        .to_dict()
    )

    if delta_counts != {
        -2: 30,
        2: 30,
    }:
        fail(
            f"Delta distribution incorrect: "
            f"{delta_counts}"
        )

    print("PASS: delta balance -2:30 / +2:30")

    # ---------------------------------------------------------
    # 10. Family counts
    # ---------------------------------------------------------

    family_counts = (
        df["family"]
        .value_counts()
        .to_dict()
    )

    if family_counts != EXPECTED_FAMILY_COUNTS:
        fail(
            "Family distribution incorrect.\n"
            f"Expected: {EXPECTED_FAMILY_COUNTS}\n"
            f"Found: {family_counts}"
        )

    print("PASS: family distribution")

    # ---------------------------------------------------------
    # 11. Source type
    # ---------------------------------------------------------

    source_types = set(
        df["source_type"]
    )

    expected_source = {
        "controlled_synthetic_flame_inspired"
    }

    if source_types != expected_source:
        fail(
            f"Unexpected source types: "
            f"{source_types}"
        )

    print("PASS: source type consistent")

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL FINAL PROBE CHECKS PASSED")
    print("=" * 70)

    print(f"Total pairs: {len(df)}")
    print()

    print("Families:")
    print(
        df["family"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Directions:")
    print(
        df["original_direction"]
        .value_counts()
        .to_string()
    )

    print()
    print("Expected deltas:")
    print(
        delta.value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()