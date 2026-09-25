from pathlib import Path

import pandas as pd


INPUT_PATH = Path(
    "data/interim/flame_candidate_review.csv"
)

EXAMPLES_PER_FAMILY = 6


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    families = sorted(
        df["family"].unique()
    )

    for family in families:
        subset = (
            df[df["family"] == family]
            .head(EXAMPLES_PER_FAMILY)
        )

        print("\n" + "=" * 80)
        print(family.upper())
        print("=" * 80)

        for _, row in subset.iterrows():
            print(
                f"\n{row['candidate_id']} | "
                f"label={row['label']} | "
                f"score={row['match_score']}"
            )

            print(row["sentence"])

            print(
                f"Source: {row['source']}"
            )


if __name__ == "__main__":
    main()