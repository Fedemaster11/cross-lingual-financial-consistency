from pathlib import Path
import json

import pandas as pd


RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FLAME_PATH = RAW_DIR / "flame_v2.parquet"
EVAL_PATH = RAW_DIR / "financial_sentiment_eval_7lang.parquet"

OUTPUT_PATH = RESULTS_DIR / "dataset_audit.json"

VALID_LABELS = {"positive", "neutral", "negative"}

TARGET_LANGUAGES = {
    "ar",
    "de",
    "en",
    "es",
    "fr",
    "ja",
    "zh",
}


def normalize_labels(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
    )


def value_counts_dict(series: pd.Series) -> dict:
    counts = series.value_counts(dropna=False)

    return {
        str(key): int(value)
        for key, value in counts.items()
    }


def audit_common(
    name: str,
    df: pd.DataFrame,
) -> dict:
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    missing = {
        column: int(df[column].isna().sum())
        for column in df.columns
    }

    print("\nMissing values:")
    for column, count in missing.items():
        print(f"  {column}: {count:,}")

    duplicate_rows = int(df.duplicated().sum())

    duplicate_sentences = int(
        df["sentence"].duplicated().sum()
    )

    duplicate_language_sentences = int(
        df.duplicated(
            subset=["language", "sentence"]
        ).sum()
    )

    print("\nDuplicates:")
    print(f"  Exact duplicate rows: {duplicate_rows:,}")
    print(
        "  Duplicate sentence strings: "
        f"{duplicate_sentences:,}"
    )
    print(
        "  Duplicate language + sentence pairs: "
        f"{duplicate_language_sentences:,}"
    )

    languages = value_counts_dict(
        df["language"]
    )

    print("\nLanguage counts:")
    for language in sorted(languages):
        print(
            f"  {language}: "
            f"{languages[language]:,}"
        )

    normalized_labels = normalize_labels(
        df["label"]
    )

    invalid_labels = sorted(
        set(normalized_labels.unique())
        - VALID_LABELS
    )

    if invalid_labels:
        raise ValueError(
            f"{name}: invalid normalized labels: "
            f"{invalid_labels}"
        )

    labels = value_counts_dict(
        normalized_labels
    )

    print("\nNormalized label counts:")
    for label in sorted(labels):
        print(
            f"  {label}: "
            f"{labels[label]:,}"
        )

    sources = value_counts_dict(
        df["source"]
    )

    print("\nTop 10 sources:")
    source_series = (
        df["source"]
        .value_counts(dropna=False)
        .head(10)
    )

    for source, count in source_series.items():
        print(
            f"  {source}: {count:,}"
        )

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "missing_values": missing,
        "exact_duplicate_rows": duplicate_rows,
        "duplicate_sentences": duplicate_sentences,
        "duplicate_language_sentence_pairs": (
            duplicate_language_sentences
        ),
        "language_counts": languages,
        "normalized_label_counts": labels,
        "source_counts": sources,
    }


def audit_flame(
    df: pd.DataFrame,
) -> dict:
    audit = audit_common(
        "FLAME v2",
        df,
    )

    raw_labels = sorted(
        str(x)
        for x in df["label"]
        .dropna()
        .unique()
    )

    print("\nRaw FLAME labels:")
    for label in raw_labels:
        print(f"  {label}")

    target_subset = df[
        df["language"].isin(
            TARGET_LANGUAGES
        )
    ]

    print(
        "\nRows in our seven target languages: "
        f"{len(target_subset):,}"
    )

    audit["raw_labels"] = raw_labels
    audit["target_language_rows"] = (
        len(target_subset)
    )

    return audit


def audit_eval(
    df: pd.DataFrame,
) -> dict:
    audit = audit_common(
        "Financial Sentiment Evaluation Set",
        df,
    )

    unique_groups = int(
        df["group_id"].nunique()
    )

    group_sizes = (
        df.groupby("group_id")
        .size()
    )

    print("\nGroup structure:")
    print(
        f"  Unique group_id values: "
        f"{unique_groups:,}"
    )

    print(
        f"  Minimum group size: "
        f"{group_sizes.min()}"
    )

    print(
        f"  Maximum group size: "
        f"{group_sizes.max()}"
    )

    print(
        f"  Mean group size: "
        f"{group_sizes.mean():.2f}"
    )

    group_size_counts = value_counts_dict(
        group_sizes
    )

    print("\nDistribution of group sizes:")
    for size in sorted(
        group_size_counts,
        key=lambda x: int(x),
    ):
        print(
            f"  size {size}: "
            f"{group_size_counts[size]:,} groups"
        )

    observed_languages = set(
        df["language"]
        .dropna()
        .unique()
    )

    missing_target_languages = (
        TARGET_LANGUAGES
        - observed_languages
    )

    if missing_target_languages:
        raise ValueError(
            "Evaluation dataset is missing target "
            f"languages: {sorted(missing_target_languages)}"
        )

    audit["unique_group_ids"] = unique_groups
    audit["group_size_min"] = int(
        group_sizes.min()
    )
    audit["group_size_max"] = int(
        group_sizes.max()
    )
    audit["group_size_mean"] = float(
        group_sizes.mean()
    )
    audit["group_size_distribution"] = (
        group_size_counts
    )

    return audit


def main() -> None:
    if not FLAME_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {FLAME_PATH}"
        )

    if not EVAL_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {EVAL_PATH}"
        )

    flame = pd.read_parquet(
        FLAME_PATH
    )

    evaluation = pd.read_parquet(
        EVAL_PATH
    )

    report = {
        "flame_v2": audit_flame(
            flame
        ),
        "financial_eval_7lang": audit_eval(
            evaluation
        ),
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print(f"Report saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()