import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
R4_PATH = ROOT / "results" / "robustness" / "r4_rationale_raw.csv"
OUT = ROOT / "results" / "robustness" / "r4_qualitative_manual"
OUT.mkdir(parents=True, exist_ok=True)

# Frozen purposive sample: one relatively consistent and one divergent case per family.
SELECTION = [
    ("profit_direction",      "P005", "consistent"),
    ("profit_direction",      "P008", "divergent"),
    ("loss_direction",        "P013", "consistent"),
    ("loss_direction",        "P009", "divergent"),
    ("cost_direction",        "P023", "consistent"),
    ("cost_direction",        "P019", "divergent"),
    ("revenue_direction",     "P025", "consistent"),
    ("revenue_direction",     "P028", "divergent"),
    ("earnings_expectations", "P034", "consistent"),
    ("earnings_expectations", "P038", "divergent"),
    ("guidance",              "P045", "consistent"),
    ("guidance",              "P048", "divergent"),
    ("margin",                "P052", "consistent"),
    ("margin",                "P051", "divergent"),
    ("contract_realization",  "P057", "consistent"),
    ("contract_realization",  "P060", "divergent"),
]

LANGS = ["en", "es", "de", "fr", "zh", "ja", "ar"]
LABELS = {"A": -2, "B": -1, "C": 0, "D": 1, "E": 2}

LEAD_RE = re.compile(r"^\s*([A-Ea-e])(?:\b|\s|[|=:;,.+\-–—])")
RATIONALE_RE = re.compile(
    r"^\s*([A-Ea-e])\s*(?:\||=|:|;|-|–|—)?\s*(.*)$",
    flags=re.DOTALL,
)

def recover_label(row):
    label = str(row.get("label", "")).strip().upper()
    if label in LABELS:
        return label
    raw = str(row.get("raw_output", "")).strip()
    m = LEAD_RE.match(raw)
    return m.group(1).upper() if m else np.nan

def recover_rationale(row):
    raw = str(row.get("raw_output", "")).strip()
    m = RATIONALE_RE.match(raw)
    if not m:
        return np.nan
    text = m.group(2).strip()
    return text if text else np.nan

def main():
    df = pd.read_csv(R4_PATH)

    if len(df) != 840:
        raise RuntimeError(f"Expected 840 R4 rows, found {len(df)}")

    sel = pd.DataFrame(SELECTION, columns=["family", "pair_id", "case_type"])

    # Validate frozen IDs against the raw file.
    available = set(zip(df["family"], df["pair_id"]))
    missing = [x for x in SELECTION if (x[0], x[1]) not in available]
    if missing:
        raise RuntimeError(f"Selected family/pair combinations not found: {missing}")

    df["recovered_label"] = df.apply(recover_label, axis=1)
    df["recovered_value"] = df["recovered_label"].map(LABELS)
    df["rationale"] = df.apply(recover_rationale, axis=1)

    sample = df.merge(sel, on=["family", "pair_id"], how="inner")

    sample["language_order"] = sample["language"].map(
        {lang: i for i, lang in enumerate(LANGS)}
    )
    sample["condition_order"] = sample["condition"].map(
        {"original": 0, "counterfactual": 1}
    )

    sample = sample.sort_values(
        ["family", "case_type", "pair_id", "condition_order", "language_order"]
    ).reset_index(drop=True)

    expected_n = 16 * 2 * 7
    if len(sample) != expected_n:
        raise RuntimeError(f"Expected {expected_n} rows, found {len(sample)}")

    # Minimal qualitative codebook. Keep coding simple and interpretable.
    sample["translation_en"] = ""
    sample["financial_direction"] = ""   # positive / negative / neutral / unclear
    sample["uncertainty"] = ""           # yes / no
    sample["label_alignment"] = ""       # aligned / partial / contradicted / unclear
    sample["reasoning_pattern"] = ""     # profitability / expectations / revenue-demand /
                                         # cost-margin / contract-business / investor-reaction /
                                         # uncertainty-only / other
    sample["notes"] = ""

    keep = [
        "family", "case_type", "pair_id", "condition", "language",
        "statement", "expected_label_value", "recovered_label",
        "recovered_value", "rationale", "raw_output",
        "translation_en", "financial_direction", "uncertainty",
        "label_alignment", "reasoning_pattern", "notes",
    ]

    sample[keep].to_csv(OUT / "r4_qualitative_224.csv", index=False, encoding="utf-8-sig")

    # One-row-per-pair manifest for the README/method section.
    manifest = sel.copy()
    manifest.to_csv(OUT / "sample_manifest.csv", index=False, encoding="utf-8-sig")

    codebook = """# R4 qualitative rationale coding protocol

## Sample
16 frozen financial pairs:
- 8 financial families
- 1 relatively consistent case + 1 divergent case per family
- 2 conditions (original/counterfactual)
- 7 languages
- Total: 224 rationales

## Coding unit
One generated rationale for one statement in one language.

## Fields

### financial_direction
What direction does the rationale itself imply for the stock?
- positive
- negative
- neutral
- unclear

Do not infer from the A-E label. Code the rationale text only.

### uncertainty
Does the rationale explicitly hedge or qualify the market effect?
- yes
- no

Examples include equivalents of may, could, likely, uncertain, unclear, depends.

### label_alignment
Does the rationale support the emitted A-E label?
- aligned
- partial
- contradicted
- unclear

A/B = negative; C = neutral/unclear; D/E = positive.

### reasoning_pattern
Primary mechanism used in the rationale:
- profitability
- expectations
- revenue-demand
- cost-margin
- contract-business
- investor-reaction
- uncertainty-only
- other

Choose the dominant mechanism. Use notes for secondary mechanisms.

## Language procedure
Keep the original rationale unchanged.
For languages the human reviewer does not read confidently, create a literal English
translation in translation_en for annotation support. Do not code from label or expected
target. Preserve the original text so ambiguous translations can be checked.

## Interpretation
This is exploratory qualitative analysis. It is not a confirmatory statistical test and
does not establish internal reasoning or causality.
"""
    (OUT / "CODEBOOK.md").write_text(codebook, encoding="utf-8")

    print("=" * 72)
    print("R4 QUALITATIVE SAMPLE FROZEN")
    print("=" * 72)
    print(sel.to_string(index=False))
    print()
    print(f"Rows exported: {len(sample)}")
    print(f"Expected:      {expected_n}")
    print()
    print("Saved:")
    print(OUT / "r4_qualitative_224.csv")
    print(OUT / "sample_manifest.csv")
    print(OUT / "CODEBOOK.md")

if __name__ == "__main__":
    main()
