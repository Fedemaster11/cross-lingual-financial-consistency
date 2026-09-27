import csv
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent

OVERLAP_PATH = ROOT / "results" / "lexical_overlap.csv"

ANALYSES = {
    "strict": ROOT / "results" / "pairwise_disagreement_strict.csv",
    "sensitivity": ROOT / "results" / "pairwise_disagreement_sensitivity.csv",
}

OUT_PATH = ROOT / "results" / "hypothesis_test.csv"

LANGUAGES = ["en", "es", "de", "fr", "zh", "ja", "ar"]
LANG_INDEX = {lang: i for i, lang in enumerate(LANGUAGES)}

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260927


def canonical_pair(a, b):
    if LANG_INDEX[a] < LANG_INDEX[b]:
        return a, b
    return b, a


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_overlap():
    rows = read_csv(OVERLAP_PATH)

    if len(rows) != 21:
        raise RuntimeError(
            f"Expected 21 overlap rows, found {len(rows)}"
        )

    overlap = {}

    for row in rows:
        key = canonical_pair(
            row["language_a"],
            row["language_b"],
        )

        overlap[key] = float(row["jaccard_overlap"])

    return overlap


def load_disagreement(path):
    rows = read_csv(path)

    if len(rows) != 1260:
        raise RuntimeError(
            f"{path.name}: expected 1260 rows, found {len(rows)}"
        )

    return rows


def pair_means(rows):
    grouped = {}

    for a, b in itertools.combinations(LANGUAGES, 2):
        grouped[(a, b)] = []

    for row in rows:
        complete = row["complete"].strip().lower() == "true"

        if not complete:
            continue

        key = canonical_pair(
            row["language_a"],
            row["language_b"],
        )

        grouped[key].append(
            float(row["disagreement"])
        )

    means = {}

    for key, values in grouped.items():
        if not values:
            raise RuntimeError(
                f"No complete disagreement observations for {key}"
            )

        means[key] = float(np.mean(values))

    return means


def aligned_vectors(overlap, means):
    pairs = list(
        itertools.combinations(LANGUAGES, 2)
    )

    x = np.array(
        [overlap[p] for p in pairs],
        dtype=float,
    )

    y = np.array(
        [means[p] for p in pairs],
        dtype=float,
    )

    return pairs, x, y


def exact_qap(overlap, means):
    pairs, _, observed_y = aligned_vectors(
        overlap,
        means,
    )

    observed_x = np.array(
        [overlap[p] for p in pairs],
        dtype=float,
    )

    observed_rho = spearmanr(
        observed_x,
        observed_y
    ).statistic

    permuted_rhos = []

    for permutation in itertools.permutations(LANGUAGES):
        mapping = dict(
            zip(LANGUAGES, permutation)
        )

        permuted_x = []

        for a, b in pairs:
            mapped_pair = canonical_pair(
                mapping[a],
                mapping[b],
            )

            permuted_x.append(
                overlap[mapped_pair]
            )

        rho = spearmanr(
            permuted_x,
            observed_y
        ).statistic

        permuted_rhos.append(rho)

    permuted_rhos = np.array(
        permuted_rhos,
        dtype=float,
    )

    # Direction specified by H1:
    # higher overlap -> lower disagreement.
    p_one_sided = np.mean(
        permuted_rhos <= observed_rho
    )

    # Conservative two-sided result.
    p_two_sided = np.mean(
        np.abs(permuted_rhos)
        >= abs(observed_rho)
    )

    return (
        float(observed_rho),
        float(p_one_sided),
        float(p_two_sided),
        len(permuted_rhos),
    )


def bootstrap_items(rows, overlap):
    pair_ids = sorted(
        {row["pair_id"] for row in rows}
    )

    if len(pair_ids) != 60:
        raise RuntimeError(
            f"Expected 60 pair IDs, found {len(pair_ids)}"
        )

    pair_rows = {}

    for pair_id in pair_ids:
        pair_rows[pair_id] = [
            row
            for row in rows
            if row["pair_id"] == pair_id
        ]

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    boot_rhos = []

    for _ in range(BOOTSTRAP_REPS):
        sampled_ids = rng.choice(
            pair_ids,
            size=len(pair_ids),
            replace=True,
        )

        grouped = {
            pair: []
            for pair in itertools.combinations(
                LANGUAGES,
                2
            )
        }

        for pair_id in sampled_ids:
            for row in pair_rows[pair_id]:
                complete = (
                    row["complete"]
                    .strip()
                    .lower()
                    == "true"
                )

                if not complete:
                    continue

                key = canonical_pair(
                    row["language_a"],
                    row["language_b"],
                )

                grouped[key].append(
                    float(row["disagreement"])
                )

        means = {}

        valid = True

        for key, values in grouped.items():
            if not values:
                valid = False
                break

            means[key] = float(
                np.mean(values)
            )

        if not valid:
            continue

        _, x, y = aligned_vectors(
            overlap,
            means,
        )

        rho = spearmanr(
            x,
            y
        ).statistic

        boot_rhos.append(rho)

    boot_rhos = np.array(
        boot_rhos,
        dtype=float,
    )

    lower = np.percentile(
        boot_rhos,
        2.5
    )

    upper = np.percentile(
        boot_rhos,
        97.5
    )

    return (
        float(lower),
        float(upper),
        len(boot_rhos),
    )


def run_analysis(name, path, overlap):
    rows = load_disagreement(path)

    means = pair_means(rows)

    _, x, y = aligned_vectors(
        overlap,
        means,
    )

    spearman = spearmanr(x, y)
    pearson = pearsonr(x, y)

    (
        qap_rho,
        qap_p_one,
        qap_p_two,
        qap_permutations,
    ) = exact_qap(
        overlap,
        means,
    )

    (
        bootstrap_low,
        bootstrap_high,
        bootstrap_valid,
    ) = bootstrap_items(
        rows,
        overlap,
    )

    n_complete = sum(
        row["complete"].strip().lower() == "true"
        for row in rows
    )

    return {
        "analysis": name,
        "n_language_pairs": 21,
        "n_pairwise_item_observations": n_complete,
        "spearman_rho": round(
            float(spearman.statistic),
            8,
        ),
        "naive_spearman_p": round(
            float(spearman.pvalue),
            10,
        ),
        "pearson_r": round(
            float(pearson.statistic),
            8,
        ),
        "naive_pearson_p": round(
            float(pearson.pvalue),
            10,
        ),
        "qap_spearman_rho": round(
            qap_rho,
            8,
        ),
        "qap_p_one_sided": round(
            qap_p_one,
            10,
        ),
        "qap_p_two_sided": round(
            qap_p_two,
            10,
        ),
        "qap_permutations": qap_permutations,
        "bootstrap_reps_requested":
            BOOTSTRAP_REPS,
        "bootstrap_reps_valid":
            bootstrap_valid,
        "bootstrap_rho_ci_low": round(
            bootstrap_low,
            8,
        ),
        "bootstrap_rho_ci_high": round(
            bootstrap_high,
            8,
        ),
    }


def main():
    print("=" * 72)
    print("H1 TEST: LEXICAL OVERLAP vs CROSS-LINGUAL DISAGREEMENT")
    print("=" * 72)

    overlap = load_overlap()

    results = []

    for name, path in ANALYSES.items():
        print()
        print(f"{name.upper()} ANALYSIS")
        print("-" * 72)

        result = run_analysis(
            name,
            path,
            overlap,
        )

        results.append(result)

        print(
            f"Pairwise item observations: "
            f"{result['n_pairwise_item_observations']}"
        )

        print(
            f"Spearman rho: "
            f"{result['spearman_rho']}"
        )

        print(
            f"Pearson r:    "
            f"{result['pearson_r']}"
        )

        print(
            f"Exact QAP permutations: "
            f"{result['qap_permutations']}"
        )

        print(
            f"QAP p (one-sided H1): "
            f"{result['qap_p_one_sided']}"
        )

        print(
            f"QAP p (two-sided):    "
            f"{result['qap_p_two_sided']}"
        )

        print(
            "Item-bootstrap 95% CI "
            f"for Spearman rho: "
            f"[{result['bootstrap_rho_ci_low']}, "
            f"{result['bootstrap_rho_ci_high']}]"
        )

    with OUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 72)
    print("SAVED")
    print("=" * 72)
    print(
        OUT_PATH.relative_to(ROOT)
    )


if __name__ == "__main__":
    main()