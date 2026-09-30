import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results" / "main_inference_raw.csv"
OUT = ROOT / "results" / "direct_label_agreement"
OUT.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["en", "es", "de", "fr", "zh", "ja", "ar"]
DISPLAY = {x: x.upper() for x in LANGUAGES}
LABELS = {"A", "B", "C", "D", "E"}

LEADING_RE = re.compile(r"^\s*([A-Ea-e])(?:\b|\s|[|:;,.\-–—])")


def as_bool(x):
    return str(x).strip().lower() == "true"


def parse_strict(row):
    if as_bool(row["valid_format"]):
        lab = str(row["label"]).strip().upper()
        if lab in LABELS:
            return lab
    return np.nan


def parse_sensitivity(row):
    strict = parse_strict(row)
    if isinstance(strict, str):
        return strict

    raw = str(row["raw_output"]).strip()
    m = LEADING_RE.match(raw)
    return m.group(1).upper() if m else np.nan


def agreement_matrix(df, label_col):
    agreement = pd.DataFrame(
        np.nan, index=LANGUAGES, columns=LANGUAGES, dtype=float
    )
    counts = pd.DataFrame(
        0, index=LANGUAGES, columns=LANGUAGES, dtype=int
    )

    for a in LANGUAGES:
        da = (
            df[df["language"] == a]
            [["pair_id", "condition", label_col]]
            .rename(columns={label_col: "label_a"})
        )

        for b in LANGUAGES:
            db = (
                df[df["language"] == b]
                [["pair_id", "condition", label_col]]
                .rename(columns={label_col: "label_b"})
            )

            m = da.merge(
                db,
                on=["pair_id", "condition"],
                how="inner",
            ).dropna(subset=["label_a", "label_b"])

            counts.loc[a, b] = len(m)

            if len(m):
                agreement.loc[a, b] = (
                    m["label_a"] == m["label_b"]
                ).mean()

    return agreement, counts


def plot_heatmap(matrix, title, path):
    values = matrix.values * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(values, vmin=0, vmax=100)

    ax.set_xticks(
        range(len(LANGUAGES)),
        [DISPLAY[x] for x in LANGUAGES],
    )
    ax.set_yticks(
        range(len(LANGUAGES)),
        [DISPLAY[x] for x in LANGUAGES],
    )

    ax.set_xlabel("Language")
    ax.set_ylabel("Language")
    ax.set_title(title)

    for i in range(len(LANGUAGES)):
        for j in range(len(LANGUAGES)):
            value = values[i, j]
            if not np.isnan(value):
                ax.text(
                    j, i, f"{value:.1f}%",
                    ha="center", va="center"
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Exact same A-E label (%)")

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pair_table(matrix, counts):
    rows = []

    for i, a in enumerate(LANGUAGES):
        for b in LANGUAGES[i + 1:]:
            rows.append({
                "language_a": a,
                "language_b": b,
                "n_compared": int(counts.loc[a, b]),
                "exact_label_agreement": float(matrix.loc[a, b]),
                "agreement_percent": float(matrix.loc[a, b] * 100),
            })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "exact_label_agreement",
            ascending=False
        )
        .reset_index(drop=True)
    )


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing primary results file: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    if len(df) != 840:
        raise RuntimeError(
            f"Expected 840 Primary rows, found {len(df)}"
        )

    required = {
        "language",
        "pair_id",
        "condition",
        "raw_output",
        "label",
        "valid_format",
    }

    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["strict_label"] = df.apply(parse_strict, axis=1)
    df["sensitivity_label"] = df.apply(
        parse_sensitivity, axis=1
    )

    strict_matrix, strict_counts = agreement_matrix(
        df, "strict_label"
    )
    sens_matrix, sens_counts = agreement_matrix(
        df, "sensitivity_label"
    )

    strict_matrix.mul(100).to_csv(
        OUT / "direct_label_agreement_strict_percent.csv"
    )
    strict_counts.to_csv(
        OUT / "direct_label_agreement_strict_n.csv"
    )
    sens_matrix.mul(100).to_csv(
        OUT / "direct_label_agreement_sensitivity_percent.csv"
    )
    sens_counts.to_csv(
        OUT / "direct_label_agreement_sensitivity_n.csv"
    )

    strict_pairs = pair_table(
        strict_matrix, strict_counts
    )
    sens_pairs = pair_table(
        sens_matrix, sens_counts
    )

    strict_pairs.to_csv(
        OUT / "direct_label_agreement_strict_pairs.csv",
        index=False,
    )
    sens_pairs.to_csv(
        OUT / "direct_label_agreement_sensitivity_pairs.csv",
        index=False,
    )

    plot_heatmap(
        strict_matrix,
        "Primary — Direct Cross-Lingual Label Agreement (Strict)",
        OUT / "direct_label_agreement_strict.png",
    )

    plot_heatmap(
        sens_matrix,
        "Primary — Direct Cross-Lingual Label Agreement (Sensitivity)",
        OUT / "direct_label_agreement_sensitivity.png",
    )

    print("=" * 72)
    print("DIRECT CROSS-LINGUAL LABEL AGREEMENT — PRIMARY")
    print("=" * 72)

    print()
    print("Strict analysis — highest agreement:")
    print(
        strict_pairs.head(7).to_string(
            index=False,
            formatters={
                "agreement_percent": lambda x: f"{x:.1f}%"
            },
        )
    )

    print()
    print("Strict analysis — lowest agreement:")
    print(
        strict_pairs.tail(7).to_string(
            index=False,
            formatters={
                "agreement_percent": lambda x: f"{x:.1f}%"
            },
        )
    )

    print()
    print("Sensitivity analysis — highest agreement:")
    print(
        sens_pairs.head(7).to_string(
            index=False,
            formatters={
                "agreement_percent": lambda x: f"{x:.1f}%"
            },
        )
    )

    print()
    print("Sensitivity analysis — lowest agreement:")
    print(
        sens_pairs.tail(7).to_string(
            index=False,
            formatters={
                "agreement_percent": lambda x: f"{x:.1f}%"
            },
        )
    )

    print()
    print("Outputs saved to:")
    print(OUT)


if __name__ == "__main__":
    main()
