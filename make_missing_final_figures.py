"""
Generate the three final figures that were skipped by make_final_figures.py.

Run from the repository root:

    python .\make_missing_final_figures.py

Outputs:
    results\figures_final\01_primary_overlap_vs_disagreement.png/.pdf
    results\figures_final\02_primary_counterfactual_disagreement_heatmap.png/.pdf
    results\figures_final\04_robustness_spearman_rho.png/.pdf

This script does not modify experimental results.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "figures_final"
OUT.mkdir(parents=True, exist_ok=True)

LANGS = ["en", "es", "de", "fr", "zh", "ja", "ar"]
DISPLAY = {x: x.upper() for x in LANGS}


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def canonical_pair(a, b):
    ia, ib = LANGS.index(a), LANGS.index(b)
    return (a, b) if ia < ib else (b, a)


def find_overlap_file():
    preferred = [
        RESULTS / "lexical_overlap.csv",
        RESULTS / "lexical_overlap_pairwise.csv",
        RESULTS / "pairwise_lexical_overlap.csv",
    ]
    for p in preferred:
        if p.exists():
            df = pd.read_csv(p)
            if {"language_a", "language_b"}.issubset(df.columns):
                overlap_col = next(
                    (c for c in ["overlap", "jaccard", "jaccard_overlap", "lexical_overlap"]
                     if c in df.columns),
                    None,
                )
                if overlap_col:
                    return p, overlap_col

    for p in RESULTS.rglob("*.csv"):
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        if not {"language_a", "language_b"}.issubset(df.columns):
            continue
        overlap_col = next(
            (c for c in ["overlap", "jaccard", "jaccard_overlap", "lexical_overlap"]
             if c in df.columns),
            None,
        )
        if overlap_col:
            return p, overlap_col

    raise FileNotFoundError(
        "Could not find a lexical-overlap CSV with language_a, language_b, "
        "and overlap/jaccard columns under results/."
    )


def load_primary_pair_means():
    path = RESULTS / "pairwise_disagreement_strict.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    # Your actual Primary file uses "disagreement", not "D".
    d_col = "D" if "D" in df.columns else "disagreement"

    required = {"language_a", "language_b", d_col}
    if not required.issubset(df.columns):
        raise RuntimeError(
            f"{path} must contain language_a, language_b and "
            f"either D or disagreement. Found: {list(df.columns)}"
        )

    df = df[df["language_a"] != df["language_b"]].copy()
    means = (
        df.groupby(["language_a", "language_b"], as_index=False)
        .agg(n=(d_col, "size"), mean_D=(d_col, "mean"))
    )
    return means, path


def load_overlap():
    path, overlap_col = find_overlap_file()
    df = pd.read_csv(path)[["language_a", "language_b", overlap_col]].copy()
    df.columns = ["language_a", "language_b", "overlap"]

    pairs = df.apply(
        lambda r: canonical_pair(str(r["language_a"]), str(r["language_b"])),
        axis=1,
    )
    df["language_a"] = [p[0] for p in pairs]
    df["language_b"] = [p[1] for p in pairs]
    df = df.drop_duplicates(["language_a", "language_b"])
    return df, path


def fig01():
    overlap, _ = load_overlap()
    means, _ = load_primary_pair_means()
    df = overlap.merge(means, on=["language_a", "language_b"], how="inner")

    if len(df) != 21:
        raise RuntimeError(f"Expected 21 language pairs, got {len(df)}")

    rho, _ = spearmanr(df["overlap"], df["mean_D"])

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.scatter(df["overlap"] * 100, df["mean_D"], s=55)

    for _, r in df.iterrows():
        ax.annotate(
            f"{DISPLAY[r['language_a']]}–{DISPLAY[r['language_b']]}",
            (r["overlap"] * 100, r["mean_D"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Tokenizer lexical overlap (Jaccard, %)")
    ax.set_ylabel("Mean counterfactual disagreement D")
    ax.set_title(
        "Primary: lexical overlap vs counterfactual disagreement\n"
        f"Spearman ρ = {rho:.3f}"
    )
    ax.grid(alpha=0.25)

    save(fig, "01_primary_overlap_vs_disagreement")
    df.to_csv(OUT / "01_primary_overlap_vs_disagreement_data.csv", index=False)


def fig02():
    means, _ = load_primary_pair_means()

    mat = pd.DataFrame(np.nan, index=LANGS, columns=LANGS, dtype=float)
    for lang in LANGS:
        mat.loc[lang, lang] = 0.0

    for _, r in means.iterrows():
        a, b = canonical_pair(str(r["language_a"]), str(r["language_b"]))
        mat.loc[a, b] = float(r["mean_D"])
        mat.loc[b, a] = float(r["mean_D"])

    values = mat.loc[LANGS, LANGS].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(values, vmin=0)

    ax.set_xticks(range(len(LANGS)), [DISPLAY[x] for x in LANGS])
    ax.set_yticks(range(len(LANGS)), [DISPLAY[x] for x in LANGS])
    ax.set_xlabel("Language")
    ax.set_ylabel("Language")
    ax.set_title("Primary: mean counterfactual disagreement by language pair")

    for i in range(len(LANGS)):
        for j in range(len(LANGS)):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean D (lower = more consistent)")

    save(fig, "02_primary_counterfactual_disagreement_heatmap")


def normalize_run(value):
    s = str(value).strip().lower()
    if s.startswith("primary"):
        return "Primary"
    if s.startswith("r1"):
        return "R1"
    if s.startswith("r2"):
        return "R2"
    if s.startswith("r3"):
        return "R3"
    if s.startswith("r4"):
        return "R4"
    return None


def get_value(row, names):
    for name in names:
        if name in row.index and pd.notna(row[name]) and str(row[name]) != "":
            return float(row[name])
    return np.nan


def load_robustness_rows():
    path = RESULTS / "robustness" / "robustness_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    if "run" not in df.columns or "spearman_rho" not in df.columns:
        raise RuntimeError(
            f"{path} must contain run and spearman_rho. Found: {list(df.columns)}"
        )

    df["run_norm"] = df["run"].map(normalize_run)
    df = df[df["run_norm"].notna()].copy()

    # If strict/sensitivity rows coexist, prefer strict.
    if "analysis" in df.columns:
        selected = []
        for run, g in df.groupby("run_norm", sort=False):
            strict = g[g["analysis"].astype(str).str.lower() == "strict"]
            selected.append(strict.iloc[0] if len(strict) else g.iloc[0])
        df = pd.DataFrame(selected)

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "run": r["run_norm"],
            "spearman_rho": float(r["spearman_rho"]),
            "qap_p": get_value(r, ["qap_p_one_sided", "qap_one_sided_p"]),
            "ci_low": get_value(r, ["bootstrap_ci_low", "bootstrap_rho_ci_low"]),
            "ci_high": get_value(r, ["bootstrap_ci_high", "bootstrap_rho_ci_high"]),
        })

    # R4 is stored separately.
    r4_path = (
        RESULTS
        / "robustness"
        / "r4_hypothesis"
        / "r4_hypothesis_summary.csv"
    )
    if r4_path.exists():
        r4 = pd.read_csv(r4_path).iloc[0]
        rows = [x for x in rows if x["run"] != "R4"]
        rows.append({
            "run": "R4",
            "spearman_rho": float(r4["spearman_rho"]),
            "qap_p": get_value(r4, ["qap_one_sided_p", "qap_p_one_sided"]),
            "ci_low": get_value(r4, ["bootstrap_ci_low", "bootstrap_rho_ci_low"]),
            "ci_high": get_value(r4, ["bootstrap_ci_high", "bootstrap_rho_ci_high"]),
        })

    out = pd.DataFrame(rows).drop_duplicates("run", keep="first")
    return out, path


def fig04():
    df, _ = load_robustness_rows()
    order = ["Primary", "R1", "R2", "R3", "R4"]

    missing = [x for x in order if x not in set(df["run"])]
    if missing:
        raise RuntimeError(
            "robustness_summary.csv does not contain all expected runs. "
            f"Missing: {missing}. Found: {df['run'].tolist()}"
        )

    df["run"] = pd.Categorical(df["run"], categories=order, ordered=True)
    df = df.sort_values("run").reset_index(drop=True)

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.4, 5.8))

    ax.scatter(x, df["spearman_rho"], s=70, zorder=3)

    for i, r in df.iterrows():
        if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]):
            lower = r["spearman_rho"] - r["ci_low"]
            upper = r["ci_high"] - r["spearman_rho"]
            ax.errorbar(
                i,
                r["spearman_rho"],
                yerr=np.array([[lower], [upper]]),
                capsize=5,
            )

        if np.isfinite(r["qap_p"]):
            ax.annotate(
                f"QAP p={r['qap_p']:.3g}",
                (i, r["spearman_rho"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x, order)
    ax.set_ylabel("Spearman ρ: lexical overlap vs mean D")
    ax.set_title("Prompt robustness of the lexical-overlap association")
    ax.grid(axis="y", alpha=0.25)

    save(fig, "04_robustness_spearman_rho")
    df.to_csv(OUT / "04_robustness_spearman_rho_data.csv", index=False)


def main():
    print("=" * 72)
    print("GENERATING THE 3 MISSING FINAL FIGURES")
    print("=" * 72)

    jobs = [
        ("01_primary_overlap_vs_disagreement", fig01),
        ("02_primary_counterfactual_disagreement_heatmap", fig02),
        ("04_robustness_spearman_rho", fig04),
    ]

    ok = 0
    for name, fn in jobs:
        try:
            fn()
            ok += 1
            print(f"[OK]   {name}")
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")

    print()
    print(f"Generated: {ok}/3")
    print("Saved to:", OUT)


if __name__ == "__main__":
    main()
