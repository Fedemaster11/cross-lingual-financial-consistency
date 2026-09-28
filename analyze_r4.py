\
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

R1_PATH = ROOT / "results" / "robustness" / "r1_max12_raw.csv"
R4_PATH = ROOT / "results" / "robustness" / "r4_rationale_raw.csv"
OUT_DIR = ROOT / "results" / "robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = ["en", "es", "de", "fr", "zh", "ja", "ar"]
LABEL_TO_VALUE = {"A": -2, "B": -1, "C": 0, "D": 1, "E": 2}


def bool_series(s):
    return s.astype(str).str.lower().eq("true")


def load_run(path):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df["task_id"].duplicated().any():
        dup = df.loc[df["task_id"].duplicated(), "task_id"].tolist()[:10]
        raise RuntimeError(f"Duplicate task_id values in {path}: {dup}")
    return df


def strict_labels(df):
    out = df.copy()
    ok = bool_series(out["valid_format"])
    out["analysis_label"] = np.where(ok, out["label"].astype(str), "")
    out["analysis_value"] = pd.to_numeric(
        np.where(ok, out["label_value"], np.nan),
        errors="coerce",
    )
    return out


def sensitivity_labels_r4(df):
    out = df.copy()
    strict_ok = bool_series(out["valid_format"])

    strict_label = out["label"].fillna("").astype(str).str.upper()
    leading_label = out["leading_label"].fillna("").astype(str).str.upper()

    chosen = np.where(
        strict_ok,
        strict_label,
        np.where(leading_label.isin(LABEL_TO_VALUE), leading_label, ""),
    )
    out["analysis_label"] = chosen
    out["analysis_value"] = pd.Series(chosen).map(LABEL_TO_VALUE)
    return out


def make_deltas(df):
    x = df.copy()
    x = x[pd.notna(x["analysis_value"])].copy()

    piv = x.pivot_table(
        index=["language", "pair_id", "family"],
        columns="condition",
        values=["analysis_value", "expected_label_value"],
        aggfunc="first",
    )

    piv.columns = ["_".join(c) for c in piv.columns]
    piv = piv.reset_index()

    needed = [
        "analysis_value_original",
        "analysis_value_counterfactual",
        "expected_label_value_original",
        "expected_label_value_counterfactual",
    ]
    for c in needed:
        if c not in piv:
            piv[c] = np.nan

    piv = piv.dropna(
        subset=["analysis_value_original", "analysis_value_counterfactual"]
    ).copy()

    piv["delta"] = (
        piv["analysis_value_counterfactual"]
        - piv["analysis_value_original"]
    )
    piv["expected_delta"] = (
        piv["expected_label_value_counterfactual"]
        - piv["expected_label_value_original"]
    )

    piv["direction_correct"] = (
        np.sign(piv["delta"]) == np.sign(piv["expected_delta"])
    )
    piv["exact_expected_delta"] = (
        piv["delta"] == piv["expected_delta"]
    )
    piv["abs_delta"] = piv["delta"].abs()

    return piv


def pairwise_disagreement(deltas):
    rows = []

    for pair_id, g in deltas.groupby("pair_id"):
        by_lang = g.set_index("language")
        langs = [x for x in LANGUAGES if x in by_lang.index]

        for a, b in itertools.combinations(langs, 2):
            da = float(by_lang.loc[a, "delta"])
            db = float(by_lang.loc[b, "delta"])

            rows.append({
                "pair_id": pair_id,
                "language_a": a,
                "language_b": b,
                "delta_a": da,
                "delta_b": db,
                "D": abs(da - db),
            })

    return pd.DataFrame(rows)


def label_agreement(r1, r4):
    a = strict_labels(r1)[["task_id", "analysis_label"]].rename(
        columns={"analysis_label": "r1_label"}
    )
    b = strict_labels(r4)[["task_id", "analysis_label"]].rename(
        columns={"analysis_label": "r4_label"}
    )

    m = a.merge(b, on="task_id", how="inner")
    m = m[(m["r1_label"] != "") & (m["r4_label"] != "")].copy()
    m["agree"] = m["r1_label"] == m["r4_label"]
    return m


def run_summary(name, analyzed):
    valid = analyzed["analysis_value"].notna().sum()
    deltas = make_deltas(analyzed)

    return {
        "run": name,
        "valid_classifications": int(valid),
        "complete_deltas": int(len(deltas)),
        "direction_correct_rate": float(deltas["direction_correct"].mean()),
        "exact_expected_delta_rate": float(deltas["exact_expected_delta"].mean()),
        "mean_abs_delta": float(deltas["abs_delta"].mean()),
    }, deltas


def main():
    r1 = load_run(R1_PATH)
    r4 = load_run(R4_PATH)

    if len(r4) != 840:
        print(f"WARNING: R4 has {len(r4)}/840 rows. Analysis will use current checkpoint.")

    r1s = strict_labels(r1)
    r4s = strict_labels(r4)
    r4sens = sensitivity_labels_r4(r4)

    s1, d1 = run_summary("R1_strict", r1s)
    s4, d4 = run_summary("R4_strict", r4s)
    s4b, d4b = run_summary("R4_sensitivity", r4sens)

    summary = pd.DataFrame([s1, s4, s4b])
    summary.to_csv(OUT_DIR / "r4_summary.csv", index=False)

    agree = label_agreement(r1, r4)
    agree.to_csv(OUT_DIR / "r4_label_agreement_tasklevel.csv", index=False)

    overall_agreement = agree["agree"].mean() if len(agree) else float("nan")

    by_language = (
        r4s[r4s["analysis_label"] != ""]
        .groupby(["language", "analysis_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=LANGUAGES, fill_value=0)
        .reindex(columns=["A", "B", "C", "D", "E"], fill_value=0)
    )
    by_language.to_csv(OUT_DIR / "r4_label_distribution_by_language.csv")

    fam = (
        d4.groupby("family")
        .agg(
            n=("pair_id", "size"),
            direction_correct_rate=("direction_correct", "mean"),
            exact_expected_delta_rate=("exact_expected_delta", "mean"),
            mean_abs_delta=("abs_delta", "mean"),
        )
        .reset_index()
        .sort_values("direction_correct_rate", ascending=False)
    )
    fam.to_csv(OUT_DIR / "r4_family_summary.csv", index=False)

    pw = pairwise_disagreement(d4)
    pw.to_csv(OUT_DIR / "r4_pairwise_disagreement.csv", index=False)

    pair_means = (
        pw.groupby(["language_a", "language_b"], as_index=False)
        .agg(n=("D", "size"), mean_D=("D", "mean"), exact_match=("D", lambda x: (x == 0).mean()))
    )
    pair_means.to_csv(OUT_DIR / "r4_pairwise_mean_disagreement.csv", index=False)

    # Case-selection rules frozen before qualitative review:
    # 1) 4 items with highest mean cross-lingual disagreement in R4
    by_item_D = (
        pw.groupby("pair_id", as_index=False)
        .agg(mean_crosslingual_D=("D", "mean"))
    )

    top_disagreement = by_item_D.nlargest(4, "mean_crosslingual_D").copy()
    top_disagreement["selection_reason"] = "highest_r4_crosslingual_disagreement"

    # 2) 4 items with largest average absolute label change R1 -> R4
    r1v = strict_labels(r1)[["language", "pair_id", "condition", "analysis_value"]].rename(
        columns={"analysis_value": "r1_value"}
    )
    r4v = strict_labels(r4)[["language", "pair_id", "condition", "analysis_value"]].rename(
        columns={"analysis_value": "r4_value"}
    )
    ch = r1v.merge(r4v, on=["language", "pair_id", "condition"], how="inner").dropna()
    ch["abs_change"] = (ch["r4_value"] - ch["r1_value"]).abs()

    changed = (
        ch.groupby("pair_id", as_index=False)
        .agg(mean_abs_r1_r4_change=("abs_change", "mean"))
        .nlargest(4, "mean_abs_r1_r4_change")
    )
    changed["selection_reason"] = "largest_r1_to_r4_label_change"

    # 3) 4 items with lowest mean cross-lingual disagreement.
    low_disagreement = by_item_D.nsmallest(4, "mean_crosslingual_D").copy()
    low_disagreement["selection_reason"] = "lowest_r4_crosslingual_disagreement"

    cases = pd.concat(
        [
            top_disagreement[["pair_id", "selection_reason"]],
            changed[["pair_id", "selection_reason"]],
            low_disagreement[["pair_id", "selection_reason"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    cases.to_csv(OUT_DIR / "r4_case_selection.csv", index=False)

    print("=" * 72)
    print("R4 EXPLORATORY ANALYSIS")
    print("=" * 72)
    print()
    print(summary.to_string(index=False))
    print()
    print(f"R1 vs R4 strict label agreement: {overall_agreement:.4f} over {len(agree)} mutually valid tasks")
    print()
    print("R4 strict label distribution by language")
    print(by_language)
    print()
    print("R4 family summary")
    print(fam.to_string(index=False))
    print()
    print("Selected qualitative cases")
    print(cases.to_string(index=False))
    print()
    print("Saved outputs under results/robustness/")


if __name__ == "__main__":
    main()
