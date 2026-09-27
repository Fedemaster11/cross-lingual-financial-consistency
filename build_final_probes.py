from pathlib import Path
import pandas as pd


OUTPUT_PATH = Path("data/final/probes_en.csv")


def pair(
    pair_id,
    family,
    positive_text,
    negative_text,
    original_direction,
):
    if original_direction == "positive":
        original = positive_text
        counterfactual = negative_text
        expected_original = 1
        expected_counterfactual = -1
    elif original_direction == "negative":
        original = negative_text
        counterfactual = positive_text
        expected_original = -1
        expected_counterfactual = 1
    else:
        raise ValueError(
            f"Invalid direction: {original_direction}"
        )

    return {
        "pair_id": pair_id,
        "family": family,
        "source_type": "controlled_synthetic_flame_inspired",
        "original_direction": original_direction,
        "original_en": original,
        "counterfactual_en": counterfactual,
        "expected_original": expected_original,
        "expected_counterfactual": expected_counterfactual,
    }


PROBES = [

    # =========================================================
    # PROFIT DIRECTION — 8
    # =========================================================

    pair(
        "P001",
        "profit_direction",
        "The company reported that operating profit increased from EUR 40 million to EUR 52 million.",
        "The company reported that operating profit decreased from EUR 52 million to EUR 40 million.",
        "positive",
    ),

    pair(
        "P002",
        "profit_direction",
        "The company reported that net profit rose by 18 percent year over year.",
        "The company reported that net profit fell by 18 percent year over year.",
        "negative",
    ),

    pair(
        "P003",
        "profit_direction",
        "Quarterly profit increased to USD 85 million from USD 70 million a year earlier.",
        "Quarterly profit decreased to USD 70 million from USD 85 million a year earlier.",
        "positive",
    ),

    pair(
        "P004",
        "profit_direction",
        "The company said annual profit grew despite stable revenue.",
        "The company said annual profit declined despite stable revenue.",
        "negative",
    ),

    pair(
        "P005",
        "profit_direction",
        "Operating profit improved by 14 percent compared with the previous quarter.",
        "Operating profit deteriorated by 14 percent compared with the previous quarter.",
        "positive",
    ),

    pair(
        "P006",
        "profit_direction",
        "Net income increased from USD 120 million to USD 145 million.",
        "Net income decreased from USD 145 million to USD 120 million.",
        "negative",
    ),

    pair(
        "P007",
        "profit_direction",
        "The company recorded a 9 percent increase in full-year profit.",
        "The company recorded a 9 percent decrease in full-year profit.",
        "positive",
    ),

    pair(
        "P008",
        "profit_direction",
        "Profit rose to EUR 210 million from EUR 190 million.",
        "Profit fell to EUR 190 million from EUR 210 million.",
        "negative",
    ),

    # =========================================================
    # LOSS DIRECTION — 8
    # =========================================================

    pair(
        "P009",
        "loss_direction",
        "The company narrowed its operating loss from USD 30 million to USD 18 million.",
        "The company widened its operating loss from USD 18 million to USD 30 million.",
        "positive",
    ),

    pair(
        "P010",
        "loss_direction",
        "The company's quarterly net loss decreased by 25 percent.",
        "The company's quarterly net loss increased by 25 percent.",
        "negative",
    ),

    pair(
        "P011",
        "loss_direction",
        "The company reduced its annual loss to EUR 12 million from EUR 20 million.",
        "The company increased its annual loss to EUR 20 million from EUR 12 million.",
        "positive",
    ),

    pair(
        "P012",
        "loss_direction",
        "Operating losses narrowed significantly during the quarter.",
        "Operating losses widened significantly during the quarter.",
        "negative",
    ),

    pair(
        "P013",
        "loss_direction",
        "The net loss fell from USD 44 million to USD 31 million.",
        "The net loss rose from USD 31 million to USD 44 million.",
        "positive",
    ),

    pair(
        "P014",
        "loss_direction",
        "The company's loss decreased for the third consecutive quarter.",
        "The company's loss increased for the third consecutive quarter.",
        "negative",
    ),

    pair(
        "P015",
        "loss_direction",
        "The company reported a smaller operating loss than in the prior year.",
        "The company reported a larger operating loss than in the prior year.",
        "positive",
    ),

    pair(
        "P016",
        "loss_direction",
        "The company's pre-tax loss narrowed to GBP 8 million.",
        "The company's pre-tax loss widened to GBP 18 million.",
        "negative",
    ),

    # =========================================================
    # COST DIRECTION — 8
    # =========================================================

    pair(
        "P017",
        "cost_direction",
        "The company reduced operating costs by 12 percent.",
        "The company increased operating costs by 12 percent.",
        "positive",
    ),

    pair(
        "P018",
        "cost_direction",
        "Quarterly expenses fell from USD 95 million to USD 82 million.",
        "Quarterly expenses rose from USD 82 million to USD 95 million.",
        "negative",
    ),

    pair(
        "P019",
        "cost_direction",
        "The company reported lower production costs during the quarter.",
        "The company reported higher production costs during the quarter.",
        "positive",
    ),

    pair(
        "P020",
        "cost_direction",
        "Operating expenses decreased by 7 percent year over year.",
        "Operating expenses increased by 7 percent year over year.",
        "negative",
    ),

    pair(
        "P021",
        "cost_direction",
        "The company cut logistics costs from EUR 28 million to EUR 21 million.",
        "The company raised logistics costs from EUR 21 million to EUR 28 million.",
        "positive",
    ),

    pair(
        "P022",
        "cost_direction",
        "Administrative expenses declined during the first half of the year.",
        "Administrative expenses increased during the first half of the year.",
        "negative",
    ),

    pair(
        "P023",
        "cost_direction",
        "The company reduced its cost base by USD 50 million.",
        "The company increased its cost base by USD 50 million.",
        "positive",
    ),

    pair(
        "P024",
        "cost_direction",
        "Manufacturing costs fell by 10 percent compared with the previous quarter.",
        "Manufacturing costs rose by 10 percent compared with the previous quarter.",
        "negative",
    ),

    # =========================================================
    # REVENUE DIRECTION — 8
    # =========================================================

    pair(
        "P025",
        "revenue_direction",
        "Quarterly revenue increased by 12 percent year over year.",
        "Quarterly revenue decreased by 12 percent year over year.",
        "positive",
    ),

    pair(
        "P026",
        "revenue_direction",
        "Sales rose from EUR 600 million to EUR 680 million.",
        "Sales fell from EUR 680 million to EUR 600 million.",
        "negative",
    ),

    pair(
        "P027",
        "revenue_direction",
        "The company reported higher revenue in all major business segments.",
        "The company reported lower revenue in all major business segments.",
        "positive",
    ),

    pair(
        "P028",
        "revenue_direction",
        "Full-year revenue grew by 9 percent.",
        "Full-year revenue declined by 9 percent.",
        "negative",
    ),

    pair(
        "P029",
        "revenue_direction",
        "Quarterly sales increased to USD 1.2 billion from USD 1.0 billion.",
        "Quarterly sales decreased to USD 1.0 billion from USD 1.2 billion.",
        "positive",
    ),

    pair(
        "P030",
        "revenue_direction",
        "Revenue rose for the fourth consecutive quarter.",
        "Revenue fell for the fourth consecutive quarter.",
        "negative",
    ),

    pair(
        "P031",
        "revenue_direction",
        "The company recorded a 15 percent increase in annual sales.",
        "The company recorded a 15 percent decrease in annual sales.",
        "positive",
    ),

    pair(
        "P032",
        "revenue_direction",
        "Revenue improved to EUR 950 million from EUR 870 million.",
        "Revenue declined to EUR 870 million from EUR 950 million.",
        "negative",
    ),

    # =========================================================
    # EARNINGS EXPECTATIONS — 8
    # =========================================================

    pair(
        "P033",
        "earnings_expectations",
        "The company reported earnings per share of USD 2.10, above analyst expectations of USD 1.80.",
        "The company reported earnings per share of USD 1.50, below analyst expectations of USD 1.80.",
        "positive",
    ),

    pair(
        "P034",
        "earnings_expectations",
        "Quarterly revenue exceeded analyst estimates by 8 percent.",
        "Quarterly revenue missed analyst estimates by 8 percent.",
        "negative",
    ),

    pair(
        "P035",
        "earnings_expectations",
        "The company reported profit above the market consensus.",
        "The company reported profit below the market consensus.",
        "positive",
    ),

    pair(
        "P036",
        "earnings_expectations",
        "Earnings per share beat analysts' forecasts.",
        "Earnings per share missed analysts' forecasts.",
        "negative",
    ),

    pair(
        "P037",
        "earnings_expectations",
        "Revenue came in higher than analysts had expected.",
        "Revenue came in lower than analysts had expected.",
        "positive",
    ),

    pair(
        "P038",
        "earnings_expectations",
        "Quarterly earnings surpassed the consensus estimate.",
        "Quarterly earnings fell short of the consensus estimate.",
        "negative",
    ),

    pair(
        "P039",
        "earnings_expectations",
        "The company exceeded analysts' earnings expectations for the quarter.",
        "The company failed to meet analysts' earnings expectations for the quarter.",
        "positive",
    ),

    pair(
        "P040",
        "earnings_expectations",
        "Reported revenue was above the consensus forecast.",
        "Reported revenue was below the consensus forecast.",
        "negative",
    ),

    # =========================================================
    # GUIDANCE — 8
    # =========================================================

    pair(
        "P041",
        "guidance",
        "Management raised its full-year revenue guidance.",
        "Management lowered its full-year revenue guidance.",
        "positive",
    ),

    pair(
        "P042",
        "guidance",
        "The company increased its annual earnings outlook.",
        "The company reduced its annual earnings outlook.",
        "negative",
    ),

    pair(
        "P043",
        "guidance",
        "Management upgraded its forecast for full-year operating profit.",
        "Management downgraded its forecast for full-year operating profit.",
        "positive",
    ),

    pair(
        "P044",
        "guidance",
        "The company lifted its expected revenue range for the year.",
        "The company cut its expected revenue range for the year.",
        "negative",
    ),

    pair(
        "P045",
        "guidance",
        "Management raised its expected earnings per share for the full year.",
        "Management lowered its expected earnings per share for the full year.",
        "positive",
    ),

    pair(
        "P046",
        "guidance",
        "The company improved its forecast for annual sales growth.",
        "The company reduced its forecast for annual sales growth.",
        "negative",
    ),

    pair(
        "P047",
        "guidance",
        "Management increased its profitability outlook for the coming year.",
        "Management decreased its profitability outlook for the coming year.",
        "positive",
    ),

    pair(
        "P048",
        "guidance",
        "The company raised the lower end of its full-year earnings forecast.",
        "The company lowered the lower end of its full-year earnings forecast.",
        "negative",
    ),

    # =========================================================
    # MARGIN — 6
    # =========================================================

    pair(
        "P049",
        "margin",
        "The company's operating margin expanded from 12 percent to 16 percent.",
        "The company's operating margin contracted from 16 percent to 12 percent.",
        "positive",
    ),

    pair(
        "P050",
        "margin",
        "Gross margin increased by three percentage points.",
        "Gross margin decreased by three percentage points.",
        "negative",
    ),

    pair(
        "P051",
        "margin",
        "The company reported wider operating margins during the quarter.",
        "The company reported narrower operating margins during the quarter.",
        "positive",
    ),

    pair(
        "P052",
        "margin",
        "Operating margin improved to 18 percent from 15 percent.",
        "Operating margin deteriorated to 15 percent from 18 percent.",
        "negative",
    ),

    pair(
        "P053",
        "margin",
        "The company's gross margin expanded despite stable revenue.",
        "The company's gross margin contracted despite stable revenue.",
        "positive",
    ),

    pair(
        "P054",
        "margin",
        "Profit margin increased from 9 percent to 11 percent.",
        "Profit margin decreased from 11 percent to 9 percent.",
        "negative",
    ),

    # =========================================================
    # CONTRACT / BUSINESS REALIZATION — 6
    # =========================================================

    pair(
        "P055",
        "contract_realization",
        "The company secured a major five-year customer contract.",
        "The company failed to secure a major five-year customer contract.",
        "positive",
    ),

    pair(
        "P056",
        "contract_realization",
        "The company won a large government supply contract.",
        "The company lost a large government supply contract.",
        "negative",
    ),

    pair(
        "P057",
        "contract_realization",
        "The company was awarded a multi-year infrastructure contract.",
        "The company's bid for a multi-year infrastructure contract was rejected.",
        "positive",
    ),

    pair(
        "P058",
        "contract_realization",
        "The company signed a major new customer agreement.",
        "The company lost a major customer agreement.",
        "negative",
    ),

    pair(
        "P059",
        "contract_realization",
        "The company secured a contract worth USD 300 million.",
        "The company failed to secure a contract worth USD 300 million.",
        "positive",
    ),

    pair(
        "P060",
        "contract_realization",
        "The company won the tender for a large industrial project.",
        "The company failed to win the tender for a large industrial project.",
        "negative",
    ),
]


def validate(df):
    if len(df) != 60:
        raise ValueError(
            f"Expected 60 probes, found {len(df)}"
        )

    if df["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair IDs found.")

    if (
        df["original_en"].str.strip()
        == df["counterfactual_en"].str.strip()
    ).any():
        raise ValueError(
            "Identical original/counterfactual text found."
        )

    direction_counts = (
        df["original_direction"]
        .value_counts()
        .to_dict()
    )

    if direction_counts.get("positive", 0) != 30:
        raise ValueError(
            "Expected 30 positive-original probes."
        )

    if direction_counts.get("negative", 0) != 30:
        raise ValueError(
            "Expected 30 negative-original probes."
        )

    valid_labels = {-1, 1}

    if not set(
        df["expected_original"]
    ).issubset(valid_labels):
        raise ValueError(
            "Invalid original expected label."
        )

    if not set(
        df["expected_counterfactual"]
    ).issubset(valid_labels):
        raise ValueError(
            "Invalid counterfactual expected label."
        )


def print_summary(df):
    print("=" * 70)
    print("FINAL ENGLISH PROBE SET")
    print("=" * 70)

    print(f"Total pairs: {len(df)}")

    print("\nOriginal direction:")
    print(
        df["original_direction"]
        .value_counts()
        .to_string()
    )

    print("\nPairs by family:")
    print(
        df["family"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nExpected delta distribution:")

    delta = (
        df["expected_counterfactual"]
        - df["expected_original"]
    )

    print(
        delta.value_counts()
        .sort_index()
        .to_string()
    )

    print(f"\nSaved to: {OUTPUT_PATH}")


def main():
    df = pd.DataFrame(PROBES)

    validate(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print_summary(df)


if __name__ == "__main__":
    main()