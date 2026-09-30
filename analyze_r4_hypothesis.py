import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent
R4_PATH = ROOT / "results" / "robustness" / "r4_rationale_raw.csv"
OVERLAP_PATH = ROOT / "results" / "lexical_overlap.csv"
OUT = ROOT / "results" / "robustness" / "r4_hypothesis"
OUT.mkdir(parents=True, exist_ok=True)

LANGS = ["en", "es", "de", "fr", "zh", "ja", "ar"]
LABEL_TO_VALUE = {"A": -2, "B": -1, "C": 0, "D": 1, "E": 2}
LEAD = re.compile(r"^\s*([A-Ea-e])(?:\b|\s|[|=:;,.+\-–—])")


def recover_label(row):
    if str(row.get("valid_format", "")).strip().lower() == "true":
        lab = str(row.get("label", "")).strip().upper()
        if lab in LABEL_TO_VALUE:
            return lab

    raw = str(row.get("raw_output", "")).strip()
    m = LEAD.match(raw)
    return m.group(1).upper() if m else np.nan


def load_r4():
    df = pd.read_csv(R4_PATH)

    if len(df) != 840:
        raise RuntimeError(f"Expected 840 R4 rows, found {len(df)}")

    if df["task_id"].duplicated().any():
        raise RuntimeError("Duplicate R4 task_id values detected.")

    df["rec_label"] = df.apply(recover_label, axis=1)
    df["rec_value"] = df["rec_label"].map(LABEL_TO_VALUE)

    if df["rec_value"].notna().sum() != 840:
        raise RuntimeError(
            "Expected all 840 R4 labels to be recoverable."
        )

    return df


def make_deltas(df):
    piv = df.pivot_table(
        index=["language", "pair_id", "family"],
        columns="condition",
        values=["rec_value", "expected_label_value"],
        aggfunc="first",
    )

    piv.columns = ["_".join(c) for c in piv.columns]
    piv = piv.reset_index()

    piv = piv.dropna(
        subset=["rec_value_original", "rec_value_counterfactual"]
    ).copy()

    piv["delta"] = (
        piv["rec_value_counterfactual"]
        - piv["rec_value_original"]
    )

    return piv


def make_pairwise(deltas):
    rows = []

    for pair_id, g in deltas.groupby("pair_id"):
        by_lang = g.set_index("language")

        for a, b in itertools.combinations(LANGS, 2):
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


def load_overlap():
    ov = pd.read_csv(OVERLAP_PATH)

    # Flexible column handling.
    colmap = {c.lower(): c for c in ov.columns}

    a_col = None
    b_col = None

    for candidate in ["language_a", "lang_a", "language1", "lang1"]:
        if candidate in colmap:
            a_col = colmap[candidate]
            break

    for candidate in ["language_b", "lang_b", "language2", "lang2"]:
        if candidate in colmap:
            b_col = colmap[candidate]
            break

    if a_col is None or b_col is None:
        raise RuntimeError(
            f"Could not identify language columns in {OVERLAP_PATH}. "
            f"Columns: {list(ov.columns)}"
        )

    overlap_col = None
    for candidate in [
        "jaccard",
        "jaccard_overlap",
        "lexical_overlap",
        "overlap",
        "similarity",
    ]:
        if candidate in colmap:
            overlap_col = colmap[candidate]
            break

    if overlap_col is None:
        numeric = [
            c for c in ov.columns
            if c not in [a_col, b_col]
            and pd.api.types.is_numeric_dtype(ov[c])
        ]
        if len(numeric) == 1:
            overlap_col = numeric[0]
        else:
            raise RuntimeError(
                f"Could not identify overlap column. Columns: {list(ov.columns)}"
            )

    out = ov[[a_col, b_col, overlap_col]].copy()
    out.columns = ["language_a", "language_b", "overlap"]

    # Canonicalize pair order.
    def canon(row):
        a, b = row["language_a"], row["language_b"]
        ia, ib = LANGS.index(a), LANGS.index(b)
        return (a, b) if ia < ib else (b, a)

    pairs = out.apply(canon, axis=1, result_type="expand")
    out["language_a"] = pairs[0]
    out["language_b"] = pairs[1]

    return out


def exact_qap(overlap_matrix, d_matrix):
    iu = np.triu_indices(len(LANGS), k=1)

    x = overlap_matrix[iu]
    y = d_matrix[iu]
    observed = spearmanr(x, y).statistic

    perm_rhos = []

    for perm in itertools.permutations(range(len(LANGS))):
        perm = np.array(perm)
        yp = d_matrix[np.ix_(perm, perm)][iu]
        rho = spearmanr(x, yp).statistic
        perm_rhos.append(rho)

    perm_rhos = np.array(perm_rhos, dtype=float)

    one_sided = np.mean(perm_rhos <= observed)
    two_sided = np.mean(np.abs(perm_rhos) >= abs(observed))

    return observed, one_sided, two_sided, perm_rhos


def bootstrap_items(pairwise, overlap_df, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    pair_ids = sorted(pairwise["pair_id"].unique())
    rhos = []

    for _ in range(n_boot):
        sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)

        chunks = []
        for new_i, pid in enumerate(sampled):
            x = pairwise[pairwise["pair_id"] == pid].copy()
            x["boot_item"] = new_i
            chunks.append(x)

        boot = pd.concat(chunks, ignore_index=True)

        means = (
            boot.groupby(["language_a", "language_b"], as_index=False)
            .agg(mean_D=("D", "mean"))
        )

        m = overlap_df.merge(
            means,
            on=["language_a", "language_b"],
            how="inner",
        )

        rho = spearmanr(m["overlap"], m["mean_D"]).statistic

        if np.isfinite(rho):
            rhos.append(rho)

    lo, hi = np.percentile(rhos, [2.5, 97.5])

    return np.array(rhos), float(lo), float(hi)


def main():
    r4 = load_r4()
    deltas = make_deltas(r4)

    if len(deltas) != 420:
        raise RuntimeError(
            f"Expected 420 complete R4 deltas, found {len(deltas)}"
        )

    pairwise = make_pairwise(deltas)
    pairwise.to_csv(
        OUT / "r4_pairwise_item_disagreement.csv",
        index=False,
    )

    means = (
        pairwise.groupby(["language_a", "language_b"], as_index=False)
        .agg(
            n=("D", "size"),
            mean_D=("D", "mean"),
            exact_delta_match=("D", lambda x: (x == 0).mean()),
        )
    )

    overlap = load_overlap()

    merged = overlap.merge(
        means,
        on=["language_a", "language_b"],
        how="inner",
    )

    if len(merged) != 21:
        raise RuntimeError(
            f"Expected 21 language pairs after merge, found {len(merged)}"
        )

    merged.to_csv(
        OUT / "r4_overlap_vs_disagreement.csv",
        index=False,
    )

    sp = spearmanr(merged["overlap"], merged["mean_D"])
    pe = pearsonr(merged["overlap"], merged["mean_D"])

    # Build symmetric matrices in fixed language order.
    overlap_mat = np.eye(len(LANGS))
    d_mat = np.zeros((len(LANGS), len(LANGS)))

    for _, row in merged.iterrows():
        i = LANGS.index(row["language_a"])
        j = LANGS.index(row["language_b"])

        overlap_mat[i, j] = row["overlap"]
        overlap_mat[j, i] = row["overlap"]

        d_mat[i, j] = row["mean_D"]
        d_mat[j, i] = row["mean_D"]

    qap_rho, qap_one, qap_two, perm_rhos = exact_qap(
        overlap_mat,
        d_mat,
    )

    boot_rhos, ci_lo, ci_hi = bootstrap_items(
        pairwise,
        overlap,
        n_boot=10000,
        seed=42,
    )

    pd.DataFrame({
        "permutation_rho": perm_rhos
    }).to_csv(
        OUT / "r4_qap_permutation_distribution.csv",
        index=False,
    )

    pd.DataFrame({
        "bootstrap_rho": boot_rhos
    }).to_csv(
        OUT / "r4_bootstrap_distribution.csv",
        index=False,
    )

    summary = pd.DataFrame([{
        "n_language_pairs": 21,
        "n_pairwise_item_observations": len(pairwise),
        "spearman_rho": sp.statistic,
        "naive_spearman_p": sp.pvalue,
        "pearson_r": pe.statistic,
        "naive_pearson_p": pe.pvalue,
        "qap_rho": qap_rho,
        "qap_one_sided_p": qap_one,
        "qap_two_sided_p": qap_two,
        "qap_permutations": len(perm_rhos),
        "bootstrap_valid": len(boot_rhos),
        "bootstrap_ci_low": ci_lo,
        "bootstrap_ci_high": ci_hi,
    }])

    summary.to_csv(
        OUT / "r4_hypothesis_summary.csv",
        index=False,
    )

    # Format compliance by language.
    fmt = (
        r4.assign(
            strict_format=r4["valid_format"]
            .astype(str)
            .str.lower()
            .eq("true")
        )
        .groupby("language", as_index=False)
        .agg(
            n=("task_id", "size"),
            strict_valid=("strict_format", "sum"),
        )
    )
    fmt["strict_rate"] = fmt["strict_valid"] / fmt["n"]
    fmt.to_csv(
        OUT / "r4_format_compliance_by_language.csv",
        index=False,
    )

    print("=" * 72)
    print("R4 — LEXICAL OVERLAP VS COUNTERFACTUAL DISAGREEMENT")
    print("=" * 72)
    print(f"Language pairs:            21")
    print(f"Pairwise item observations:{len(pairwise)}")
    print(f"Spearman rho:              {sp.statistic:.6f}")
    print(f"Naive Spearman p:          {sp.pvalue:.8f}")
    print(f"Pearson r:                 {pe.statistic:.6f}")
    print(f"Naive Pearson p:           {pe.pvalue:.8f}")
    print(f"Exact QAP one-sided p:     {qap_one:.8f}")
    print(f"Exact QAP two-sided p:     {qap_two:.8f}")
    print(f"Bootstrap 95% CI:          [{ci_lo:.6f}, {ci_hi:.6f}]")
    print()

    print("Mean D by language pair:")
    print(
        merged[
            ["language_a", "language_b", "overlap", "mean_D", "exact_delta_match"]
        ]
        .sort_values("mean_D")
        .to_string(index=False)
    )
    print()

    print("R4 format compliance by language:")
    print(
        fmt.to_string(
            index=False,
            formatters={"strict_rate": lambda x: f"{x:.1%}"}
        )
    )
    print()
    print("Saved to:", OUT)


if __name__ == "__main__":
    main()
