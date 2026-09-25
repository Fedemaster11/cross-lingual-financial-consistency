from pathlib import Path
import re

import pandas as pd


INPUT_PATH = Path("data/raw/flame_v2.parquet")

ALL_OUTPUT = Path(
    "data/interim/flame_candidate_pool.csv"
)

REVIEW_OUTPUT = Path(
    "data/interim/flame_candidate_review.csv"
)

TARGET_PER_FAMILY = 40


FAMILIES = {
    "profit_direction": {
        "metric": re.compile(
            r"\b("
            r"profit|profits|operating profit|"
            r"net income|income"
            r")\b",
            re.IGNORECASE,
        ),
        "change": re.compile(
            r"\b("
            r"increase[ds]?|decrease[ds]?|"
            r"rose|rise[sn]?|fell|fall[sn]?|"
            r"grew|grow[sn]?|decline[ds]?|"
            r"jumped|dropped|surged|slumped"
            r")\b",
            re.IGNORECASE,
        ),
    },

    "loss_direction": {
        "metric": re.compile(
            r"\b(loss|losses|net loss|operating loss)\b",
            re.IGNORECASE,
        ),
        "change": re.compile(
            r"\b("
            r"increase[ds]?|decrease[ds]?|"
            r"widened|narrowed|grew|shr[a|u]nk|"
            r"rose|fell|higher|lower"
            r")\b",
            re.IGNORECASE,
        ),
    },

    "cost_direction": {
        "metric": re.compile(
            r"\b("
            r"cost|costs|expense|expenses|"
            r"spending|expenditure"
            r")\b",
            re.IGNORECASE,
        ),
        "change": re.compile(
            r"\b("
            r"increase[ds]?|decrease[ds]?|"
            r"rose|fell|higher|lower|"
            r"cut|cuts|reduced|grew"
            r")\b",
            re.IGNORECASE,
        ),
    },

    "revenue_direction": {
        "metric": re.compile(
            r"\b("
            r"revenue|revenues|sales|turnover"
            r")\b",
            re.IGNORECASE,
        ),
        "change": re.compile(
            r"\b("
            r"increase[ds]?|decrease[ds]?|"
            r"rose|fell|grew|decline[ds]?|"
            r"higher|lower|jumped|dropped|"
            r"surged|slumped"
            r")\b",
            re.IGNORECASE,
        ),
    },

    "earnings_expectations": {
        "metric": re.compile(
            r"\b("
            r"earnings|eps|profit|revenue|sales"
            r")\b",
            re.IGNORECASE,
        ),
        "change": re.compile(
            r"\b("
            r"beat|beats|beating|"
            r"miss|misses|missed|"
            r"above|below|exceeded|"
            r"surpassed|short of"
            r")\b",
            re.IGNORECASE,
        ),
        "context": re.compile(
            r"\b("
            r"expectation|expectations|"
            r"estimate|estimates|"
            r"forecast|forecasts|"
            r"consensus|analyst|analysts"
            r")\b",
            re.IGNORECASE,
        ),
    },

    "guidance": {
        "metric": re.compile(
            r"\b("
            r"guidance|outlook|forecast|"
            r"full-year outlook|full year outlook"
            r")\b",
            re.IGNORECASE,
        ),
        "change": re.compile(
            r"\b("
            r"raise[ds]?|raised|"
            r"lower[eds]*|cut|cuts|"
            r"lifted|boosted|slashed|"
            r"upgraded|downgraded|"
            r"increased|decreased"
            r")\b",
            re.IGNORECASE,
        ),
    },

    "margin": {
        "metric": re.compile(
            r"\b("
            r"margin|margins|"
            r"operating margin|gross margin"
            r")\b",
            re.IGNORECASE,
        ),
        "change": re.compile(
            r"\b("
            r"expanded|contracted|"
            r"widened|narrowed|"
            r"increased|decreased|"
            r"rose|fell|higher|lower"
            r")\b",
            re.IGNORECASE,
        ),
    },

    "contract_realization": {
        "metric": re.compile(
            r"\b("
            r"contract|contracts|deal|deals|"
            r"order|orders|tender"
            r")\b",
            re.IGNORECASE,
        ),
        "change": re.compile(
            r"\b("
            r"won|win|wins|secured|"
            r"awarded|signed|landed|"
            r"lost|cancelled|canceled|"
            r"terminated|rejected|failed"
            r")\b",
            re.IGNORECASE,
        ),
    },
}


def normalize_label(label: str) -> str:
    return str(label).strip().lower()


def match_family(
    sentence: str,
    rules: dict,
) -> tuple[bool, int]:
    metric_matches = rules["metric"].findall(sentence)
    change_matches = rules["change"].findall(sentence)

    if not metric_matches or not change_matches:
        return False, 0

    score = len(metric_matches) + len(change_matches)

    if "context" in rules:
        context_matches = rules["context"].findall(sentence)

        if not context_matches:
            return False, 0

        score += len(context_matches)

    return True, score


def build_candidate_pool(
    df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    english = df[
        df["language"].eq("en")
    ].copy()

    for _, row in english.iterrows():
        sentence = str(row["sentence"]).strip()

        if not sentence:
            continue

        for family, rules in FAMILIES.items():
            matched, score = match_family(
                sentence,
                rules,
            )

            if not matched:
                continue

            rows.append(
                {
                    "family": family,
                    "sentence": sentence,
                    "label": normalize_label(
                        row["label"]
                    ),
                    "source": row["source"],
                    "language": row["language"],
                    "match_score": score,
                    "word_count": len(
                        sentence.split()
                    ),
                }
            )

    candidates = pd.DataFrame(rows)

    if candidates.empty:
        raise ValueError(
            "No candidate headlines were found."
        )

    candidates = candidates.drop_duplicates(
        subset=["family", "sentence"]
    )

    candidates = candidates.sort_values(
        by=[
            "family",
            "match_score",
            "word_count",
            "sentence",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    candidates.insert(
        0,
        "candidate_id",
        [
            f"CAND{i:05d}"
            for i in range(
                1,
                len(candidates) + 1,
            )
        ],
    )

    return candidates


def build_review_set(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    review = (
        candidates
        .groupby(
            "family",
            group_keys=False,
        )
        .head(TARGET_PER_FAMILY)
        .copy()
    )

    review["keep"] = ""
    review["notes"] = ""
    review["counterfactual_plan"] = ""

    return review


def print_summary(
    candidates: pd.DataFrame,
    review: pd.DataFrame,
) -> None:
    print("=" * 70)
    print("FLAME CANDIDATE MINING")
    print("=" * 70)

    print(
        f"Total candidate family-headline matches: "
        f"{len(candidates):,}"
    )

    print("\nCandidates by family:")

    counts = (
        candidates["family"]
        .value_counts()
        .sort_index()
    )

    for family, count in counts.items():
        print(
            f"  {family}: {count:,}"
        )

    print(
        f"\nManual review set: "
        f"{len(review):,} rows"
    )

    print("\nReview rows by family:")

    review_counts = (
        review["family"]
        .value_counts()
        .sort_index()
    )

    for family, count in review_counts.items():
        print(
            f"  {family}: {count:,}"
        )

    print()
    print(
        f"Full candidate pool: {ALL_OUTPUT}"
    )
    print(
        f"Manual review file: {REVIEW_OUTPUT}"
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing FLAME dataset: {INPUT_PATH}"
        )

    df = pd.read_parquet(
        INPUT_PATH
    )

    candidates = build_candidate_pool(
        df
    )

    review = build_review_set(
        candidates
    )

    ALL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        ALL_OUTPUT,
        index=False,
    )

    review.to_csv(
        REVIEW_OUTPUT,
        index=False,
    )

    print_summary(
        candidates,
        review,
    )


if __name__ == "__main__":
    main()