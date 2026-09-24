from pathlib import Path
import pandas as pd


OUTPUT_PATH = Path("data/interim/probes_dev.csv")


PROBES = [
    {
        "pair_id": "DEV001",
        "family": "profit_direction",
        "source_type": "synthetic_development",
        "original_en": (
            "The company reported that operating profit increased "
            "from EUR 10 million to EUR 15 million."
        ),
        "counterfactual_en": (
            "The company reported that operating profit decreased "
            "from EUR 15 million to EUR 10 million."
        ),
        "expected_original": 1,
        "expected_counterfactual": -1,
        "critical_change": "operating profit increased -> decreased",
    },
    {
        "pair_id": "DEV002",
        "family": "loss_direction",
        "source_type": "synthetic_development",
        "original_en": (
            "The company reported that operating loss decreased "
            "from EUR 15 million to EUR 10 million."
        ),
        "counterfactual_en": (
            "The company reported that operating loss increased "
            "from EUR 10 million to EUR 15 million."
        ),
        "expected_original": 1,
        "expected_counterfactual": -1,
        "critical_change": "operating loss decreased -> increased",
    },
    {
        "pair_id": "DEV003",
        "family": "cost_direction",
        "source_type": "synthetic_development",
        "original_en": (
            "The company reported that operating costs decreased "
            "from EUR 15 million to EUR 10 million."
        ),
        "counterfactual_en": (
            "The company reported that operating costs increased "
            "from EUR 10 million to EUR 15 million."
        ),
        "expected_original": 1,
        "expected_counterfactual": -1,
        "critical_change": "operating costs decreased -> increased",
    },
    {
        "pair_id": "DEV004",
        "family": "revenue_direction",
        "source_type": "synthetic_development",
        "original_en": (
            "Quarterly revenue increased by 12 percent year over year."
        ),
        "counterfactual_en": (
            "Quarterly revenue decreased by 12 percent year over year."
        ),
        "expected_original": 1,
        "expected_counterfactual": -1,
        "critical_change": "revenue increased -> decreased",
    },
    {
        "pair_id": "DEV005",
        "family": "earnings_expectations",
        "source_type": "synthetic_development",
        "original_en": (
            "The company reported earnings per share of USD 2.10, "
            "above analyst expectations of USD 1.80."
        ),
        "counterfactual_en": (
            "The company reported earnings per share of USD 1.50, "
            "below analyst expectations of USD 1.80."
        ),
        "expected_original": 1,
        "expected_counterfactual": -1,
        "critical_change": "earnings beat -> earnings miss",
    },
    {
        "pair_id": "DEV006",
        "family": "guidance",
        "source_type": "synthetic_development",
        "original_en": (
            "Management raised its full-year revenue guidance."
        ),
        "counterfactual_en": (
            "Management lowered its full-year revenue guidance."
        ),
        "expected_original": 1,
        "expected_counterfactual": -1,
        "critical_change": "guidance raised -> lowered",
    },
    {
        "pair_id": "DEV007",
        "family": "margin",
        "source_type": "synthetic_development",
        "original_en": (
            "The company's operating margin expanded from "
            "12 percent to 16 percent."
        ),
        "counterfactual_en": (
            "The company's operating margin contracted from "
            "16 percent to 12 percent."
        ),
        "expected_original": 1,
        "expected_counterfactual": -1,
        "critical_change": "margin expanded -> contracted",
    },
    {
        "pair_id": "DEV008",
        "family": "contract_realization",
        "source_type": "synthetic_development",
        "original_en": (
            "The company secured a major five-year customer contract."
        ),
        "counterfactual_en": (
            "The company failed to secure a major five-year customer contract."
        ),
        "expected_original": 1,
        "expected_counterfactual": -1,
        "critical_change": "contract secured -> failed to secure",
    },
]


def validate(df: pd.DataFrame) -> None:
    required_columns = {
        "pair_id",
        "family",
        "source_type",
        "original_en",
        "counterfactual_en",
        "expected_original",
        "expected_counterfactual",
        "critical_change",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair_id values found.")

    if (df["original_en"] == df["counterfactual_en"]).any():
        raise ValueError(
            "At least one pair has identical original and counterfactual text."
        )

    valid_labels = {-2, -1, 0, 1, 2}

    if not set(df["expected_original"]).issubset(valid_labels):
        raise ValueError("Invalid expected_original label.")

    if not set(df["expected_counterfactual"]).issubset(valid_labels):
        raise ValueError("Invalid expected_counterfactual label.")


def main() -> None:
    df = pd.DataFrame(PROBES)

    validate(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Created {len(df)} development counterfactual pairs.")
    print(f"Saved to: {OUTPUT_PATH}")
    print()

    print(
        df[
            ["pair_id", "family", "critical_change"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()