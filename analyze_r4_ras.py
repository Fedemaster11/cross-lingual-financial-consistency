from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results" / "robustness" / "r4_qualitative_manual" / "r4_qualitative_224_coded.csv"
OUT = ROOT / "results" / "robustness" / "r4_rationale_alignment"
OUT.mkdir(parents=True, exist_ok=True)

LANGS = ["en", "es", "de", "fr", "zh", "ja", "ar"]

REQUIRED = {
    "family", "pair_id", "case_type", "condition", "language",
    "recovered_label", "direction", "magnitude",
    "causal_structure", "uncertainty", "label_alignment",
    "prompt_leakage",
}

df = pd.read_csv(INPUT)

missing = REQUIRED - set(df.columns)
if missing:
    raise RuntimeError(f"Missing required columns: {sorted(missing)}")

if len(df) != 224:
    raise RuntimeError(f"Expected 224 coded rationales, found {len(df)}")

rows = []

for (family, pair_id, case_type, condition), g in df.groupby(
    ["family", "pair_id", "case_type", "condition"]
):
    by_lang = g.set_index("language")

    if set(by_lang.index) != set(LANGS):
        raise RuntimeError(
            f"{pair_id}/{condition}: expected all 7 languages"
        )

    for a, b in combinations(LANGS, 2):
        A = by_lang.loc[a]
        B = by_lang.loc[b]

        direction_match = int(A["direction"] == B["direction"])
        magnitude_match = int(
            direction_match == 1
            and A["magnitude"] == B["magnitude"]
        )
        causal_match = int(
            int(A["causal_structure"]) == int(B["causal_structure"])
        )

        ras = (
            direction_match
            + magnitude_match
            + causal_match
        ) / 3

        same_label = int(
            A["recovered_label"] == B["recovered_label"]
        )

        if same_label and ras >= 2 / 3:
            cell = "same_label_high_RAS"
        elif same_label:
            cell = "same_label_low_RAS"
        elif ras >= 2 / 3:
            cell = "different_label_high_RAS"
        else:
            cell = "different_label_low_RAS"

        rows.append({
            "family": family,
            "pair_id": pair_id,
            "case_type": case_type,
            "condition": condition,
            "language_a": a,
            "language_b": b,
            "label_a": A["recovered_label"],
            "label_b": B["recovered_label"],
            "same_label": same_label,
            "direction_match": direction_match,
            "magnitude_match": magnitude_match,
            "causal_match": causal_match,
            "RAS": ras,
            "alignment_cell": cell,
        })

pairwise = pd.DataFrame(rows)

if len(pairwise) != 672:
    raise RuntimeError(
        f"Expected 672 statement × language-pair observations, found {len(pairwise)}"
    )

by_pair = pairwise.groupby(
    ["language_a", "language_b"], as_index=False
).agg(
    n=("RAS", "size"),
    direction_match_rate=("direction_match", "mean"),
    magnitude_match_rate=("magnitude_match", "mean"),
    causal_match_rate=("causal_match", "mean"),
    mean_RAS=("RAS", "mean"),
    exact_label_match_rate=("same_label", "mean"),
)

by_family = pairwise.groupby(
    "family", as_index=False
).agg(
    n_pairwise=("RAS", "size"),
    direction_match_rate=("direction_match", "mean"),
    magnitude_match_rate=("magnitude_match", "mean"),
    causal_match_rate=("causal_match", "mean"),
    mean_RAS=("RAS", "mean"),
    exact_label_match_rate=("same_label", "mean"),
).sort_values("mean_RAS", ascending=False)

by_language = df.groupby(
    "language", as_index=False
).agg(
    n=("pair_id", "size"),
    mean_uncertainty=("uncertainty", "mean"),
    mean_causal=("causal_structure", "mean"),
    prompt_leakage_rate=(
        "prompt_leakage",
        lambda s: (s == "yes").mean(),
    ),
    aligned_rate=(
        "label_alignment",
        lambda s: (s == "aligned").mean(),
    ),
    partial_rate=(
        "label_alignment",
        lambda s: (s == "partial").mean(),
    ),
    contradicted_rate=(
        "label_alignment",
        lambda s: (s == "contradicted").mean(),
    ),
)

cells = pairwise.groupby(
    "alignment_cell"
).size().reset_index(name="n")
cells["pct"] = cells["n"] / len(pairwise)

pairwise.to_csv(
    OUT / "r4_rationale_alignment_pairwise_672.csv",
    index=False,
)
by_pair.to_csv(
    OUT / "r4_ras_by_language_pair.csv",
    index=False,
)
by_family.to_csv(
    OUT / "r4_ras_by_family.csv",
    index=False,
)
by_language.to_csv(
    OUT / "r4_rationale_behavior_by_language.csv",
    index=False,
)
cells.to_csv(
    OUT / "r4_alignment_2x2_summary.csv",
    index=False,
)

print("=" * 72)
print("R4 RATIONALE ALIGNMENT")
print("=" * 72)
print(f"Coded rationales:              {len(df)}")
print(f"Pairwise RAS observations:     {len(pairwise)}")
print()
print("Language-pair summary:")
print(
    by_pair.sort_values("mean_RAS", ascending=False)
    .round(3)
    .to_string(index=False)
)
print()
print("Family summary:")
print(
    by_family.round(3).to_string(index=False)
)
print()
print("2x2:")
print(
    cells.assign(
        pct=(cells["pct"] * 100).round(1)
    ).to_string(index=False)
)
print()
print("Saved to:", OUT)
