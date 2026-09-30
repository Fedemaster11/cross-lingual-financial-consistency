r"""
Generate the final figures for the Knowledge Conflicts seminar project.

Run from the repository root:

    python .\\make_final_figures.py

Outputs:
    results\figures_final\*.png
    results\figures_final\*.pdf
    results\figures_final\figure_manifest.csv

The script never changes experimental result files. It only reads existing CSVs
and writes figures / small figure-source tables.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "figures_final"
OUT.mkdir(parents=True, exist_ok=True)

LANGS = ["en", "es", "de", "fr", "zh", "ja", "ar"]
DISPLAY = {x: x.upper() for x in LANGS}
LABELS = ["A", "B", "C", "D", "E"]
LEADING_RE = re.compile(r"^\s*([A-Ea-e])(?:\b|\s|[|=:;,.\-+–—])")

manifest_rows: list[dict] = []
generated: list[str] = []
skipped: list[tuple[str, str]] = []


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def find_file(basename: str, preferred: list[Path] | None = None) -> Path:
    """Return a unique/best existing file by basename."""
    if preferred:
        for p in preferred:
            if p.exists():
                return p

    matches = sorted(RESULTS.rglob(basename))
    if not matches:
        raise FileNotFoundError(f"Could not find {basename} under {RESULTS}")

    # Prefer the shortest path; it usually corresponds to the canonical output.
    matches.sort(key=lambda p: (len(p.parts), str(p)))
    return matches[0]


def record_figure(stem: str, purpose: str, sources: list[Path]) -> None:
    manifest_rows.append({
        "figure": stem,
        "purpose": purpose,
        "sources": "; ".join(str(p.relative_to(ROOT)) for p in sources),
    })


def save_figure(fig, stem: str) -> None:
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    generated.append(stem)


def run_plot(stem: str, func) -> None:
    try:
        func()
        print(f"[OK]   {stem}")
    except Exception as exc:
        skipped.append((stem, str(exc)))
        print(f"[SKIP] {stem}: {exc}")


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    ia, ib = LANGS.index(a), LANGS.index(b)
    return (a, b) if ia < ib else (b, a)


def load_overlap() -> tuple[pd.DataFrame, Path]:
    path = find_file(
        "lexical_overlap.csv",
        [RESULTS / "lexical_overlap.csv"],
    )
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}

    a_col = next(
        (lower[x] for x in ["language_a", "lang_a", "language1", "lang1"] if x in lower),
        None,
    )
    b_col = next(
        (lower[x] for x in ["language_b", "lang_b", "language2", "lang2"] if x in lower),
        None,
    )

    if a_col is None or b_col is None:
        raise RuntimeError(f"Cannot identify language columns in {path}: {list(df.columns)}")

    overlap_col = next(
        (
            lower[x]
            for x in ["jaccard", "jaccard_overlap", "lexical_overlap", "overlap", "similarity"]
            if x in lower
        ),
        None,
    )

    if overlap_col is None:
        numeric = [
            c for c in df.columns
            if c not in [a_col, b_col]
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(numeric) != 1:
            raise RuntimeError(f"Cannot identify overlap column in {path}: {list(df.columns)}")
        overlap_col = numeric[0]

    out = df[[a_col, b_col, overlap_col]].copy()
    out.columns = ["language_a", "language_b", "overlap"]
    out[["language_a", "language_b"]] = out.apply(
        lambda r: pd.Series(canonical_pair(str(r["language_a"]), str(r["language_b"]))),
        axis=1,
    )
    out = out.drop_duplicates(["language_a", "language_b"])
    return out, path


def load_primary_pairwise_strict() -> tuple[pd.DataFrame, Path]:
    path = find_file(
        "pairwise_disagreement_strict.csv",
        [
            RESULTS / "pairwise_disagreement_strict.csv",
            RESULTS / "analysis" / "pairwise_disagreement_strict.csv",
        ],
    )
    df = pd.read_csv(path)
    required = {"language_a", "language_b", "D"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"{path} must contain {sorted(required)}")
    return df, path


def primary_pair_means() -> tuple[pd.DataFrame, list[Path]]:
    pairwise, p = load_primary_pairwise_strict()
    means = (
        pairwise.groupby(["language_a", "language_b"], as_index=False)
        .agg(n=("D", "size"), mean_D=("D", "mean"))
    )
    return means, [p]


def matrix_from_pairs(
    df: pd.DataFrame,
    value_col: str,
    diagonal: float | None = 0.0,
) -> pd.DataFrame:
    mat = pd.DataFrame(np.nan, index=LANGS, columns=LANGS, dtype=float)
    if diagonal is not None:
        for lang in LANGS:
            mat.loc[lang, lang] = diagonal

    for _, row in df.iterrows():
        a, b = canonical_pair(str(row["language_a"]), str(row["language_b"]))
        v = float(row[value_col])
        mat.loc[a, b] = v
        mat.loc[b, a] = v
    return mat


def heatmap(
    matrix: pd.DataFrame,
    title: str,
    colorbar_label: str,
    stem: str,
    fmt: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    values = matrix.loc[LANGS, LANGS].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 7.0))
    im = ax.imshow(values, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(LANGS)), [DISPLAY[x] for x in LANGS])
    ax.set_yticks(range(len(LANGS)), [DISPLAY[x] for x in LANGS])
    ax.set_xlabel("Language")
    ax.set_ylabel("Language")
    ax.set_title(title)

    for i in range(len(LANGS)):
        for j in range(len(LANGS)):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    save_figure(fig, stem)


def parse_strict_label(row) -> str | float:
    valid = str(row.get("valid_format", "")).strip().lower() == "true"
    label = str(row.get("label", "")).strip().upper()
    if valid and label in LABELS:
        return label
    return np.nan


def recover_leading_label(row) -> str | float:
    strict = parse_strict_label(row)
    if isinstance(strict, str):
        return strict
    raw = str(row.get("raw_output", "")).strip()
    m = LEADING_RE.match(raw)
    return m.group(1).upper() if m else np.nan


def compute_label_agreement_matrix(
    raw: pd.DataFrame,
    label_col: str,
) -> pd.DataFrame:
    mat = pd.DataFrame(np.nan, index=LANGS, columns=LANGS, dtype=float)

    for a in LANGS:
        da = (
            raw[raw["language"] == a][["pair_id", "condition", label_col]]
            .rename(columns={label_col: "a"})
        )
        for b in LANGS:
            db = (
                raw[raw["language"] == b][["pair_id", "condition", label_col]]
                .rename(columns={label_col: "b"})
            )
            z = da.merge(db, on=["pair_id", "condition"]).dropna()
            if len(z):
                mat.loc[a, b] = (z["a"] == z["b"]).mean() * 100
    return mat


def load_matrix_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = [str(x).lower() for x in df.index]
    df.columns = [str(x).lower() for x in df.columns]
    df = df.loc[LANGS, LANGS].astype(float)
    return df


def load_primary_direct_agreement() -> tuple[pd.DataFrame, list[Path]]:
    candidates = [
        RESULTS / "direct_label_agreement" / "direct_label_agreement_strict_percent.csv",
    ]
    existing = next((p for p in candidates if p.exists()), None)
    if existing is None:
        matches = list(RESULTS.rglob("direct_label_agreement_strict_percent.csv"))
        existing = matches[0] if matches else None

    if existing:
        return load_matrix_csv(existing), [existing]

    raw_path = find_file("main_inference_raw.csv", [RESULTS / "main_inference_raw.csv"])
    raw = pd.read_csv(raw_path)
    raw["strict_label"] = raw.apply(parse_strict_label, axis=1)
    return compute_label_agreement_matrix(raw, "strict_label"), [raw_path]


def load_r4_direct_agreement() -> tuple[pd.DataFrame, list[Path]]:
    preferred = (
        RESULTS / "robustness" / "r4_recovered" / "r4_direct_label_agreement_percent.csv"
    )
    if preferred.exists():
        return load_matrix_csv(preferred), [preferred]

    matches = list(RESULTS.rglob("r4_direct_label_agreement_percent.csv"))
    if matches:
        return load_matrix_csv(matches[0]), [matches[0]]

    raw_path = find_file(
        "r4_rationale_raw.csv",
        [RESULTS / "robustness" / "r4_rationale_raw.csv"],
    )
    raw = pd.read_csv(raw_path)
    raw["rec_label"] = raw.apply(recover_leading_label, axis=1)
    return compute_label_agreement_matrix(raw, "rec_label"), [raw_path]


# ---------------------------------------------------------------------
# Figure 01: Primary lexical overlap vs disagreement
# ---------------------------------------------------------------------

def fig01_primary_overlap_scatter():
    overlap, op = load_overlap()
    means, ps = primary_pair_means()
    df = overlap.merge(means, on=["language_a", "language_b"], how="inner")
    if len(df) != 21:
        raise RuntimeError(f"Expected 21 language pairs, found {len(df)}")

    sp = spearmanr(df["overlap"], df["mean_D"])

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.scatter(df["overlap"] * 100, df["mean_D"], s=55)

    for _, r in df.iterrows():
        label = f"{DISPLAY[r['language_a']]}–{DISPLAY[r['language_b']]}"
        ax.annotate(
            label,
            (r["overlap"] * 100, r["mean_D"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Tokenizer lexical overlap (Jaccard, %)")
    ax.set_ylabel("Mean counterfactual disagreement D")
    ax.set_title(
        f"Primary: lexical overlap vs counterfactual disagreement\n"
        f"Spearman ρ = {sp.statistic:.3f}"
    )
    ax.grid(alpha=0.25)

    save_figure(fig, "01_primary_overlap_vs_disagreement")
    df.to_csv(OUT / "01_primary_overlap_vs_disagreement_data.csv", index=False)
    record_figure(
        "01_primary_overlap_vs_disagreement",
        "Main H1 figure: higher lexical overlap corresponds to lower mean disagreement.",
        [op, *ps],
    )


# ---------------------------------------------------------------------
# Figure 02: Primary mean-D heatmap
# ---------------------------------------------------------------------

def fig02_primary_d_heatmap():
    means, sources = primary_pair_means()
    mat = matrix_from_pairs(means, "mean_D", diagonal=0.0)
    heatmap(
        mat,
        "Primary: mean counterfactual disagreement by language pair",
        "Mean D (lower = more consistent)",
        "02_primary_counterfactual_disagreement_heatmap",
        ".2f",
        vmin=0,
    )
    record_figure(
        "02_primary_counterfactual_disagreement_heatmap",
        "Main cross-lingual counterfactual-consistency heatmap.",
        sources,
    )


# ---------------------------------------------------------------------
# Figure 03: Primary direct label agreement heatmap
# ---------------------------------------------------------------------

def fig03_primary_label_heatmap():
    mat, sources = load_primary_direct_agreement()
    heatmap(
        mat,
        "Primary: direct cross-lingual label agreement",
        "Exact same A–E label (%)",
        "03_primary_direct_label_agreement_heatmap",
        ".1f",
        vmin=0,
        vmax=100,
    )
    record_figure(
        "03_primary_direct_label_agreement_heatmap",
        "Static same-label agreement, complementary to counterfactual D.",
        sources,
    )


# ---------------------------------------------------------------------
# Figure 04: Robustness of lexical-overlap association
# ---------------------------------------------------------------------

def classify_run(path: Path) -> str | None:
    s = str(path).lower().replace("\\", "/")
    if "r4" in s:
        return "R4"
    if "r3" in s:
        return "R3"
    if "r2" in s:
        return "R2"
    if "r1" in s:
        return "R1"
    if "robust" not in s:
        return "Primary"
    return None


def discover_hypothesis_summaries() -> tuple[pd.DataFrame, list[Path]]:
    rows = []
    sources = []

    for path in sorted(RESULTS.rglob("*.csv")):
        try:
            d = pd.read_csv(path)
        except Exception:
            continue

        if "spearman_rho" not in d.columns:
            continue

        run = classify_run(path)
        if run is None:
            continue

        row = d.copy()

        # Prefer a strict row if the file contains multiple analysis variants.
        selector_col = next(
            (c for c in ["analysis", "mode", "parser", "variant"] if c in row.columns),
            None,
        )
        if selector_col is not None:
            strict = row[row[selector_col].astype(str).str.lower().str.contains("strict")]
            if len(strict):
                row = strict

        r = row.iloc[0]

        def maybe(*names):
            for name in names:
                if name in row.columns:
                    return r[name]
            return np.nan

        rows.append({
            "run": run,
            "spearman_rho": float(r["spearman_rho"]),
            "qap_p": float(maybe("qap_one_sided_p", "qap_p")),
            "ci_low": float(maybe("bootstrap_ci_low", "ci_low")),
            "ci_high": float(maybe("bootstrap_ci_high", "ci_high")),
            "source": str(path),
        })
        sources.append(path)

    if not rows:
        raise FileNotFoundError("No hypothesis-summary CSVs with spearman_rho found.")

    out = pd.DataFrame(rows)

    # If multiple candidates exist for a run, prefer the canonical-looking one.
    priority = {
        "Primary": ["hypothesis_test.csv", "hypothesis_summary.csv"],
        "R1": ["r1", "hypothesis"],
        "R2": ["r2", "hypothesis"],
        "R3": ["r3", "hypothesis"],
        "R4": ["r4_hypothesis_summary.csv"],
    }

    selected = []
    for run in ["Primary", "R1", "R2", "R3", "R4"]:
        g = out[out["run"] == run].copy()
        if not len(g):
            continue

        def score(p):
            s = str(p).lower()
            prefs = priority[run]
            return sum(x in s for x in prefs)

        g["priority"] = g["source"].map(score)
        selected.append(g.sort_values(["priority", "source"], ascending=[False, True]).iloc[0])

    final = pd.DataFrame(selected)
    if len(final) < 3:
        raise RuntimeError(
            f"Only found {len(final)} robustness conditions. "
            "Run the robustness analysis scripts first."
        )

    source_paths = [Path(x) for x in final["source"]]
    return final, source_paths


def fig04_robustness_rho():
    df, sources = discover_hypothesis_summaries()
    order = [x for x in ["Primary", "R1", "R2", "R3", "R4"] if x in set(df["run"])]
    df["run"] = pd.Categorical(df["run"], categories=order, ordered=True)
    df = df.sort_values("run").reset_index(drop=True)

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.scatter(x, df["spearman_rho"], s=70, zorder=3)

    for i, r in df.iterrows():
        if np.isfinite(r["ci_low"]) and np.isfinite(r["ci_high"]):
            yerr = np.array([
                [r["spearman_rho"] - r["ci_low"]],
                [r["ci_high"] - r["spearman_rho"]],
            ])
            ax.errorbar(i, r["spearman_rho"], yerr=yerr, capsize=5)

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
    ax.set_xticks(x, [str(v) for v in df["run"]])
    ax.set_ylabel("Spearman ρ: lexical overlap vs mean D")
    ax.set_title("Prompt robustness of the lexical-overlap association")
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, "04_robustness_spearman_rho")
    df.drop(columns=["priority"], errors="ignore").to_csv(
        OUT / "04_robustness_spearman_rho_data.csv",
        index=False,
    )
    record_figure(
        "04_robustness_spearman_rho",
        "Shows that the overlap association is strong in Primary/R1 but prompt-sensitive.",
        sources,
    )


# ---------------------------------------------------------------------
# Figure 05: R1 vs R4 label agreement by language
# ---------------------------------------------------------------------

def load_r1_r4_by_language() -> tuple[pd.DataFrame, Path]:
    preferred = RESULTS / "robustness" / "r4_recovered" / "r1_vs_r4_agreement_by_language.csv"
    if preferred.exists():
        return pd.read_csv(preferred), preferred
    path = find_file("r1_vs_r4_agreement_by_language.csv")
    return pd.read_csv(path), path


def fig05_r1_r4_by_language():
    df, path = load_r1_r4_by_language()
    if "agreement_percent" not in df.columns:
        if "agreement" not in df.columns:
            raise RuntimeError(f"{path} needs agreement or agreement_percent")
        df["agreement_percent"] = df["agreement"] * 100

    df["language"] = pd.Categorical(df["language"], LANGS, ordered=True)
    df = df.sort_values("language")

    fig, ax = plt.subplots(figsize=(8.0, 5.3))
    ax.bar([DISPLAY[str(x)] for x in df["language"]], df["agreement_percent"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("R1–R4 exact label agreement (%)")
    ax.set_xlabel("Language")
    ax.set_title("Rationale elicitation changes classifications unevenly across languages")
    ax.grid(axis="y", alpha=0.25)

    for i, v in enumerate(df["agreement_percent"]):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8)

    save_figure(fig, "05_r1_vs_r4_label_agreement_by_language")
    record_figure(
        "05_r1_vs_r4_label_agreement_by_language",
        "Language-specific behavioral change induced by rationale elicitation.",
        [path],
    )


# ---------------------------------------------------------------------
# Figure 06: Direction-correct by financial family across robustness runs
# ---------------------------------------------------------------------

def discover_family_summaries() -> tuple[dict[str, tuple[pd.DataFrame, Path]], list[Path]]:
    found: dict[str, tuple[pd.DataFrame, Path]] = {}
    sources = []

    for path in sorted(RESULTS.rglob("*.csv")):
        try:
            d = pd.read_csv(path)
        except Exception:
            continue

        if "family" not in d.columns:
            continue

        col = next(
            (
                c for c in [
                    "direction_correct",
                    "direction_correct_rate",
                    "direction_accuracy",
                ]
                if c in d.columns
            ),
            None,
        )
        if col is None:
            continue

        run = classify_run(path)
        if run not in ["R1", "R2", "R3", "R4"]:
            continue

        x = d[["family", col]].copy()
        x.columns = ["family", "direction_correct"]
        if x["direction_correct"].max() > 1.5:
            x["direction_correct"] = x["direction_correct"] / 100

        # Prefer explicit family-summary files.
        score = int("family_summary" in path.name.lower()) + int("recovered" in str(path).lower())
        if run not in found:
            found[run] = (x, path, score)
        elif score > found[run][2]:
            found[run] = (x, path, score)

    clean = {k: (v[0], v[1]) for k, v in found.items()}
    sources = [v[1] for v in clean.values()]
    return clean, sources


def fig06_family_direction_correct():
    runs, sources = discover_family_summaries()
    if not runs:
        raise FileNotFoundError("No family direction-correct summaries found.")

    order_runs = [r for r in ["R1", "R2", "R3", "R4"] if r in runs]
    families = []
    for run in order_runs:
        for f in runs[run][0]["family"]:
            if f not in families:
                families.append(f)

    # Sort using R4 if available, otherwise first run.
    sorter = "R4" if "R4" in runs else order_runs[0]
    rank = (
        runs[sorter][0]
        .set_index("family")["direction_correct"]
        .sort_values(ascending=True)
    )
    families = list(rank.index)

    y = np.arange(len(families))
    nrun = len(order_runs)
    width = 0.8 / max(nrun, 1)

    fig, ax = plt.subplots(figsize=(9.5, 6.7))
    for k, run in enumerate(order_runs):
        d = runs[run][0].set_index("family").reindex(families)
        offset = (k - (nrun - 1) / 2) * width
        ax.barh(
            y + offset,
            d["direction_correct"] * 100,
            height=width * 0.9,
            label=run,
        )

    ax.set_yticks(y, [f.replace("_", " ") for f in families])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Counterfactual direction correct (%)")
    ax.set_title("Financial-family behavior across robustness conditions")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)

    save_figure(fig, "06_family_direction_correct_robustness")
    record_figure(
        "06_family_direction_correct_robustness",
        "Shows which financial relation types are easier/harder across robustness runs.",
        sources,
    )


# ---------------------------------------------------------------------
# Figure 07: R4 label distribution by language
# ---------------------------------------------------------------------

def load_r4_label_distribution() -> tuple[pd.DataFrame, Path]:
    preferred = RESULTS / "robustness" / "r4_recovered" / "r4_label_distribution.csv"
    path = preferred if preferred.exists() else find_file("r4_label_distribution.csv")
    df = pd.read_csv(path, index_col=0)
    df.index = [str(x).lower() for x in df.index]
    return df.reindex(index=LANGS, columns=LABELS, fill_value=0), path


def fig07_r4_label_distribution():
    df, path = load_r4_label_distribution()

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    bottom = np.zeros(len(df))

    for lab in LABELS:
        vals = df[lab].to_numpy()
        ax.bar([DISPLAY[x] for x in df.index], vals, bottom=bottom, label=lab)
        bottom += vals

    ax.set_ylabel("Number of recovered labels")
    ax.set_xlabel("Language")
    ax.set_title("R4 recovered A–E label distribution by language")
    ax.legend(title="Label", ncol=5)
    ax.grid(axis="y", alpha=0.2)

    save_figure(fig, "07_r4_label_distribution_by_language")
    record_figure(
        "07_r4_label_distribution_by_language",
        "Shows scale asymmetry and language-specific use of A–E under rationale elicitation.",
        [path],
    )


# ---------------------------------------------------------------------
# Figure 08: R4 direct label agreement heatmap
# ---------------------------------------------------------------------

def fig08_r4_direct_label_heatmap():
    mat, sources = load_r4_direct_agreement()
    heatmap(
        mat,
        "R4: direct cross-lingual label agreement",
        "Exact same recovered A–E label (%)",
        "08_r4_direct_label_agreement_heatmap",
        ".1f",
        vmin=0,
        vmax=100,
    )
    record_figure(
        "08_r4_direct_label_agreement_heatmap",
        "Static label-consistency structure under rationale elicitation.",
        sources,
    )


# ---------------------------------------------------------------------
# Figure 09: R4 lexical overlap vs disagreement
# ---------------------------------------------------------------------

def fig09_r4_overlap_scatter():
    preferred = RESULTS / "robustness" / "r4_hypothesis" / "r4_overlap_vs_disagreement.csv"
    path = preferred if preferred.exists() else find_file("r4_overlap_vs_disagreement.csv")
    df = pd.read_csv(path)

    sp = spearmanr(df["overlap"], df["mean_D"])

    summary_path = RESULTS / "robustness" / "r4_hypothesis" / "r4_hypothesis_summary.csv"
    qap_text = ""
    sources = [path]
    if summary_path.exists():
        s = pd.read_csv(summary_path).iloc[0]
        if "qap_one_sided_p" in s:
            qap_text = f", QAP p={float(s['qap_one_sided_p']):.3g}"
        sources.append(summary_path)

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
    ax.set_ylabel("Mean R4 counterfactual disagreement D")
    ax.set_title(
        f"R4: lexical overlap vs disagreement\n"
        f"Spearman ρ={sp.statistic:.3f}{qap_text}"
    )
    ax.grid(alpha=0.25)

    save_figure(fig, "09_r4_overlap_vs_disagreement")
    record_figure(
        "09_r4_overlap_vs_disagreement",
        "R4 version of the H1 plot; illustrates weaker prompt-dependent association.",
        sources,
    )


# ---------------------------------------------------------------------
# Figures 10–12: RAS analysis
# ---------------------------------------------------------------------

def load_ras_by_pair() -> tuple[pd.DataFrame, Path]:
    preferred = RESULTS / "robustness" / "r4_rationale_alignment" / "r4_ras_by_language_pair.csv"
    path = preferred if preferred.exists() else find_file("r4_ras_by_language_pair.csv")
    return pd.read_csv(path), path


def load_ras_by_family() -> tuple[pd.DataFrame, Path]:
    preferred = RESULTS / "robustness" / "r4_rationale_alignment" / "r4_ras_by_family.csv"
    path = preferred if preferred.exists() else find_file("r4_ras_by_family.csv")
    return pd.read_csv(path), path


def load_ras_2x2() -> tuple[pd.DataFrame, Path]:
    preferred = RESULTS / "robustness" / "r4_rationale_alignment" / "r4_alignment_2x2_summary.csv"
    path = preferred if preferred.exists() else find_file("r4_alignment_2x2_summary.csv")
    return pd.read_csv(path), path


def fig10_ras_by_language_pair():
    df, path = load_ras_by_pair()
    df["pair"] = df["language_a"].str.upper() + "–" + df["language_b"].str.upper()
    df = df.sort_values("mean_RAS", ascending=True)

    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    ax.barh(df["pair"], df["mean_RAS"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean Rationale Alignment Score (RAS)")
    ax.set_ylabel("Language pair")
    ax.set_title("R4 rationale alignment by language pair")
    ax.grid(axis="x", alpha=0.25)

    for i, v in enumerate(df["mean_RAS"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

    save_figure(fig, "10_r4_ras_by_language_pair")
    record_figure(
        "10_r4_ras_by_language_pair",
        "Cross-lingual rationale alignment over the 32 selected R4 statements.",
        [path],
    )


def fig11_ras_by_family():
    df, path = load_ras_by_family()
    df = df.sort_values("mean_RAS", ascending=True)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.barh(df["family"].str.replace("_", " ", regex=False), df["mean_RAS"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean Rationale Alignment Score (RAS)")
    ax.set_ylabel("Financial family")
    ax.set_title("R4 rationale alignment by financial relation type")
    ax.grid(axis="x", alpha=0.25)

    for i, v in enumerate(df["mean_RAS"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

    save_figure(fig, "11_r4_ras_by_family")
    record_figure(
        "11_r4_ras_by_family",
        "Shows where generated rationales are more/less cross-lingually aligned.",
        [path],
    )


def fig12_ras_2x2():
    df, path = load_ras_2x2()
    mapping = {
        "same_label_high_RAS": "Same label\nHigh RAS",
        "same_label_low_RAS": "Same label\nLow RAS",
        "different_label_high_RAS": "Different label\nHigh RAS",
        "different_label_low_RAS": "Different label\nLow RAS",
    }
    df["display"] = df["alignment_cell"].map(mapping).fillna(df["alignment_cell"])

    if "pct" not in df.columns:
        df["pct"] = df["n"] / df["n"].sum()
    pct = df["pct"] * 100

    fig, ax = plt.subplots(figsize=(8.3, 5.7))
    bars = ax.bar(df["display"], pct)
    ax.set_ylim(0, max(70, pct.max() + 8))
    ax.set_ylabel("Share of 672 comparisons (%)")
    ax.set_title("R4 output agreement × rationale alignment")
    ax.grid(axis="y", alpha=0.25)

    for bar, n, p in zip(bars, df["n"], pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            p + 1.2,
            f"{int(n)}\n({p:.1f}%)",
            ha="center",
            fontsize=8,
        )

    save_figure(fig, "12_r4_label_vs_rationale_2x2")
    record_figure(
        "12_r4_label_vs_rationale_2x2",
        "Separates aligned behavior, calibration differences, and deeper rationale divergence.",
        [path],
    )


# ---------------------------------------------------------------------
# Figure 13: Inter-annotator reliability
# ---------------------------------------------------------------------

def load_annotator_agreement() -> tuple[pd.DataFrame, Path]:
    preferred = (
        RESULTS
        / "robustness"
        / "r4_second_annotator"
        / "analysis"
        / "r4_second_annotator_agreement.csv"
    )
    path = preferred if preferred.exists() else find_file("r4_second_annotator_agreement.csv")
    return pd.read_csv(path), path


def fig13_second_annotator_agreement():
    df, path = load_annotator_agreement()
    df["reported_kappa"] = df["weighted_kappa_quadratic"].where(
        df["weighted_kappa_quadratic"].notna(),
        df["cohen_kappa"],
    )
    df["exact_agreement_fraction"] = df["exact_agreement"]
    if df["exact_agreement_fraction"].max() > 1.5:
        df["exact_agreement_fraction"] /= 100

    labels = df["dimension"].str.replace("_", " ", regex=False)
    x = np.arange(len(df))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.bar(
        x - width / 2,
        df["exact_agreement_fraction"],
        width=width,
        label="Exact agreement",
    )
    ax.bar(
        x + width / 2,
        df["reported_kappa"],
        width=width,
        label="κ (weighted for ordinal)",
    )

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(min(-0.15, df["reported_kappa"].min() - 0.05), 1.05)
    ax.set_ylabel("Agreement coefficient")
    ax.set_title("Independent second-annotator validation of R4 coding")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, "13_r4_second_annotator_reliability")
    record_figure(
        "13_r4_second_annotator_reliability",
        "Reliability check for semantic coding, causal structure, uncertainty, and label alignment.",
        [path],
    )


# ---------------------------------------------------------------------
# Figures 14–15: RAS stability across annotators
# ---------------------------------------------------------------------

def load_ras_comparison() -> tuple[pd.DataFrame, Path]:
    preferred = (
        RESULTS
        / "robustness"
        / "r4_second_annotator"
        / "analysis"
        / "r4_second_annotator_ras_comparison.csv"
    )
    path = preferred if preferred.exists() else find_file("r4_second_annotator_ras_comparison.csv")
    return pd.read_csv(path), path


def annotator_scatter(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    title: str,
    stem: str,
    purpose: str,
    source: Path,
):
    sp = spearmanr(df[xcol], df[ycol])
    pe = pearsonr(df[xcol], df[ycol])

    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    ax.scatter(df[xcol], df[ycol], s=55)

    lo = min(df[xcol].min(), df[ycol].min()) - 0.03
    hi = max(df[xcol].max(), df[ycol].max()) + 0.03
    lo = max(0, lo)
    hi = min(1, hi)
    ax.plot([lo, hi], [lo, hi], linewidth=1)

    for _, r in df.iterrows():
        ax.annotate(
            f"{DISPLAY[r['language_a']]}–{DISPLAY[r['language_b']]}",
            (r[xcol], r[ycol]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("First annotation")
    ax.set_ylabel("Independent second annotation")
    ax.set_title(
        f"{title}\nSpearman ρ={sp.statistic:.3f}; Pearson r={pe.statistic:.3f}"
    )
    ax.grid(alpha=0.25)

    save_figure(fig, stem)
    record_figure(stem, purpose, [source])


def fig14_ras_annotator_stability():
    df, path = load_ras_comparison()
    annotator_scatter(
        df,
        "RAS_first",
        "RAS_second",
        "RAS stability across independent annotations",
        "14_r4_ras_annotator_stability",
        "Shows aggregate RAS ranking is stable despite lower causal-structure agreement.",
        path,
    )


def fig15_semantic_ras_annotator_stability():
    df, path = load_ras_comparison()
    annotator_scatter(
        df,
        "semantic_RAS_first",
        "semantic_RAS_second",
        "Semantic-only RAS stability across independent annotations",
        "15_r4_semantic_ras_annotator_stability",
        "Sensitivity using direction + magnitude only; excludes causal-structure coding.",
        path,
    )


# ---------------------------------------------------------------------
# Figure 16: R4 format compliance by language
# ---------------------------------------------------------------------

def fig16_r4_format_compliance():
    preferred = (
        RESULTS / "robustness" / "r4_hypothesis" / "r4_format_compliance_by_language.csv"
    )
    path = preferred if preferred.exists() else find_file("r4_format_compliance_by_language.csv")
    df = pd.read_csv(path)
    if "strict_rate" not in df.columns:
        df["strict_rate"] = df["strict_valid"] / df["n"]

    df["language"] = pd.Categorical(df["language"], LANGS, ordered=True)
    df = df.sort_values("language")

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    vals = df["strict_rate"] * 100
    bars = ax.bar([DISPLAY[str(x)] for x in df["language"]], vals)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Exact `LETTER | rationale` compliance (%)")
    ax.set_xlabel("Language")
    ax.set_title("R4 strict output-format compliance by language")
    ax.grid(axis="y", alpha=0.25)

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1.5,
            f"{v:.1f}%",
            ha="center",
            fontsize=8,
        )

    save_figure(fig, "16_r4_format_compliance_by_language")
    record_figure(
        "16_r4_format_compliance_by_language",
        "Instruction-following differences under rationale elicitation.",
        [path],
    )


# ---------------------------------------------------------------------
# Figure 17: Change in direct label agreement from Primary to R4
# ---------------------------------------------------------------------

def fig17_primary_to_r4_agreement_change():
    pmat, psources = load_primary_direct_agreement()
    rmat, rsources = load_r4_direct_agreement()
    delta = rmat - pmat

    # Diagonal is mechanically zero and not substantively interesting.
    for lang in LANGS:
        delta.loc[lang, lang] = 0.0

    values = delta.loc[LANGS, LANGS].to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values))

    heatmap(
        delta,
        "Change in direct label agreement: R4 − Primary",
        "Percentage-point change",
        "17_primary_to_r4_label_agreement_change",
        "+.1f",
        vmin=-vmax,
        vmax=vmax,
    )
    record_figure(
        "17_primary_to_r4_label_agreement_change",
        "Shows that rationale elicitation changes the cross-lingual similarity structure non-uniformly.",
        [*psources, *rsources],
    )


# ---------------------------------------------------------------------
# Figure 18: Pair-level relationship between label agreement and RAS
# ---------------------------------------------------------------------

def fig18_label_agreement_vs_ras():
    df, path = load_ras_by_pair()

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.scatter(df["exact_label_match_rate"] * 100, df["mean_RAS"], s=55)

    for _, r in df.iterrows():
        ax.annotate(
            f"{DISPLAY[r['language_a']]}–{DISPLAY[r['language_b']]}",
            (r["exact_label_match_rate"] * 100, r["mean_RAS"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    sp = spearmanr(df["exact_label_match_rate"], df["mean_RAS"])
    ax.set_xlabel("Exact label agreement in selected R4 sample (%)")
    ax.set_ylabel("Mean Rationale Alignment Score (RAS)")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Label agreement vs rationale alignment by language pair\n"
        f"Spearman ρ={sp.statistic:.3f}"
    )
    ax.grid(alpha=0.25)

    save_figure(fig, "18_r4_label_agreement_vs_ras")
    record_figure(
        "18_r4_label_agreement_vs_ras",
        "Shows how static label similarity relates to rationale similarity.",
        [path],
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

PLOTS = [
    ("01_primary_overlap_vs_disagreement", fig01_primary_overlap_scatter),
    ("02_primary_counterfactual_disagreement_heatmap", fig02_primary_d_heatmap),
    ("03_primary_direct_label_agreement_heatmap", fig03_primary_label_heatmap),
    ("04_robustness_spearman_rho", fig04_robustness_rho),
    ("05_r1_vs_r4_label_agreement_by_language", fig05_r1_r4_by_language),
    ("06_family_direction_correct_robustness", fig06_family_direction_correct),
    ("07_r4_label_distribution_by_language", fig07_r4_label_distribution),
    ("08_r4_direct_label_agreement_heatmap", fig08_r4_direct_label_heatmap),
    ("09_r4_overlap_vs_disagreement", fig09_r4_overlap_scatter),
    ("10_r4_ras_by_language_pair", fig10_ras_by_language_pair),
    ("11_r4_ras_by_family", fig11_ras_by_family),
    ("12_r4_label_vs_rationale_2x2", fig12_ras_2x2),
    ("13_r4_second_annotator_reliability", fig13_second_annotator_agreement),
    ("14_r4_ras_annotator_stability", fig14_ras_annotator_stability),
    ("15_r4_semantic_ras_annotator_stability", fig15_semantic_ras_annotator_stability),
    ("16_r4_format_compliance_by_language", fig16_r4_format_compliance),
    ("17_primary_to_r4_label_agreement_change", fig17_primary_to_r4_agreement_change),
    ("18_r4_label_agreement_vs_ras", fig18_label_agreement_vs_ras),
]


def main():
    print("=" * 78)
    print("FINAL FIGURE GENERATION")
    print("=" * 78)
    print("Repository:", ROOT)
    print("Output:", OUT)
    print()

    for stem, func in PLOTS:
        run_plot(stem, func)

    pd.DataFrame(manifest_rows).to_csv(
        OUT / "figure_manifest.csv",
        index=False,
    )

    print()
    print("=" * 78)
    print(f"Generated: {len(generated)} figures")
    print(f"Skipped:   {len(skipped)} figures")
    print("=" * 78)

    if skipped:
        print("\nSkipped figures:")
        for stem, reason in skipped:
            print(f"- {stem}: {reason}")

    print("\nRecommended README figures:")
    print("  01  Primary overlap vs disagreement")
    print("  02  Primary mean-D heatmap")
    print("  04  Robustness Spearman rho")
    print("  12  R4 label agreement × rationale alignment (if space permits)")

    print("\nRecommended presentation extras:")
    print("  03, 05, 07, 09, 10, 11, 13, 14, 16, 17, 18")


if __name__ == "__main__":
    main()
