import csv
import itertools
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr


ROOT = Path(__file__).resolve().parent

OVERLAP_PATH = ROOT / "results" / "lexical_overlap.csv"

RUNS = {
    "Primary": ROOT / "results" / "main_inference_raw.csv",
    "R1": ROOT / "results" / "robustness" / "r1_max12_raw.csv",
    "R2": ROOT / "results" / "robustness" / "r2_strict_prompt_raw.csv",
    "R3": ROOT / "results" / "robustness" / "r3_paraphrase_raw.csv",
}

OUT = ROOT / "results" / "robustness" / "robustness_summary.csv"

LANGUAGES = ["en", "es", "de", "fr", "zh", "ja", "ar"]
LANG_INDEX = {lang: i for i, lang in enumerate(LANGUAGES)}

LABEL_TO_VALUE = {
    "A": -2,
    "B": -1,
    "C": 0,
    "D": 1,
    "E": 2,
}

BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20260928


def canonical_pair(a, b):
    if LANG_INDEX[a] < LANG_INDEX[b]:
        return a, b
    return b, a


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def strict_label(row):
    if row["valid_format"].strip().lower() != "true":
        return None

    label = row["label"].strip().upper()

    if label not in LABEL_TO_VALUE:
        return None

    return label


def sensitivity_label(row):
    strict = strict_label(row)

    if strict is not None:
        return strict

    raw = row["raw_output"]

    match = re.match(
        r"^\s*([A-E])(?:\s|[-–—:.)]|$)",
        raw,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1).upper()

    return None


def load_overlap():
    rows = read_csv(OVERLAP_PATH)

    overlap = {}

    for row in rows:
        key = canonical_pair(
            row["language_a"],
            row["language_b"],
        )

        overlap[key] = float(
            row["jaccard_overlap"]
        )

    if len(overlap) != 21:
        raise RuntimeError(
            f"Expected 21 overlaps, found {len(overlap)}"
        )

    return overlap


def build_deltas(rows, parser):
    grouped = defaultdict(dict)

    for row in rows:
        key = (
            row["language"],
            row["pair_id"],
        )

        grouped[key][row["condition"]] = row

    output = {}

    for language in LANGUAGES:
        for n in range(1, 61):
            pair_id = f"P{n:03d}"

            pair = grouped[
                (language, pair_id)
            ]

            orig = parser(
                pair["original"]
            )

            cf = parser(
                pair["counterfactual"]
            )

            if orig is None or cf is None:
                delta = None
            else:
                delta = (
                    LABEL_TO_VALUE[cf]
                    - LABEL_TO_VALUE[orig]
                )

            output[
                (pair_id, language)
            ] = delta

    return output


def build_pairwise(deltas):
    output = []

    for n in range(1, 61):
        pair_id = f"P{n:03d}"

        for a, b in itertools.combinations(
            LANGUAGES,
            2
        ):
            da = deltas[(pair_id, a)]
            db = deltas[(pair_id, b)]

            if da is None or db is None:
                continue

            output.append({
                "pair_id": pair_id,
                "language_a": a,
                "language_b": b,
                "disagreement": abs(da - db),
            })

    return output


def pair_means(pairwise):
    grouped = defaultdict(list)

    for row in pairwise:
        key = canonical_pair(
            row["language_a"],
            row["language_b"],
        )

        grouped[key].append(
            row["disagreement"]
        )

    return {
        key: float(np.mean(values))
        for key, values in grouped.items()
    }


def vectors(overlap, means):
    pairs = list(
        itertools.combinations(
            LANGUAGES,
            2
        )
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
    pairs, x, y = vectors(
        overlap,
        means
    )

    observed = float(
        spearmanr(x, y).statistic
    )

    permuted = []

    for permutation in itertools.permutations(
        LANGUAGES
    ):
        mapping = dict(
            zip(
                LANGUAGES,
                permutation,
            )
        )

        px = []

        for a, b in pairs:
            mapped = canonical_pair(
                mapping[a],
                mapping[b],
            )

            px.append(
                overlap[mapped]
            )

        rho = spearmanr(
            px,
            y
        ).statistic

        permuted.append(rho)

    permuted = np.array(
        permuted,
        dtype=float,
    )

    p_one = float(
        np.mean(
            permuted <= observed
        )
    )

    p_two = float(
        np.mean(
            np.abs(permuted)
            >= abs(observed)
        )
    )

    return (
        observed,
        p_one,
        p_two,
        len(permuted),
    )


def item_bootstrap(
    pairwise,
    overlap,
):
    by_item = defaultdict(list)

    for row in pairwise:
        by_item[row["pair_id"]].append(row)

    pair_ids = [
        f"P{i:03d}"
        for i in range(1, 61)
    ]

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    rhos = []

    for _ in range(
        BOOTSTRAP_REPS
    ):
        sampled = rng.choice(
            pair_ids,
            size=60,
            replace=True,
        )

        grouped = defaultdict(list)

        for pair_id in sampled:
            for row in by_item[pair_id]:
                key = canonical_pair(
                    row["language_a"],
                    row["language_b"],
                )

                grouped[key].append(
                    row["disagreement"]
                )

        # A bootstrap sample must contain all 21
        # language pairs.
        if len(grouped) != 21:
            continue

        means = {
            key: float(np.mean(values))
            for key, values in grouped.items()
        }

        _, x, y = vectors(
            overlap,
            means
        )

        rhos.append(
            spearmanr(
                x,
                y
            ).statistic
        )

    rhos = np.array(
        rhos,
        dtype=float,
    )

    return (
        float(np.percentile(rhos, 2.5)),
        float(np.percentile(rhos, 97.5)),
        len(rhos),
    )


def label_map(rows, parser):
    output = {}

    for row in rows:
        output[row["task_id"]] = parser(row)

    return output


def agreement(a, b):
    shared = []

    for task_id in a:
        if (
            a[task_id] is not None
            and b.get(task_id) is not None
        ):
            shared.append(
                a[task_id]
                == b[task_id]
            )

    if not shared:
        return "", 0

    return (
        float(np.mean(shared)),
        len(shared),
    )


def analyze_run(
    name,
    rows,
    overlap,
    parser,
    analysis_name,
):
    if len(rows) != 840:
        raise RuntimeError(
            f"{name}: expected 840 rows, "
            f"found {len(rows)}"
        )

    deltas = build_deltas(
        rows,
        parser,
    )

    complete_deltas = sum(
        value is not None
        for value in deltas.values()
    )

    pairwise = build_pairwise(
        deltas
    )

    means = pair_means(
        pairwise
    )

    _, x, y = vectors(
        overlap,
        means
    )

    spearman = spearmanr(
        x,
        y
    )

    pearson = pearsonr(
        x,
        y
    )

    (
        qap_rho,
        qap_p_one,
        qap_p_two,
        qap_n,
    ) = exact_qap(
        overlap,
        means,
    )

    (
        boot_low,
        boot_high,
        boot_n,
    ) = item_bootstrap(
        pairwise,
        overlap,
    )

    valid_outputs = sum(
        parser(row) is not None
        for row in rows
    )

    return {
        "run": name,
        "analysis": analysis_name,
        "valid_outputs": valid_outputs,
        "invalid_outputs": 840 - valid_outputs,
        "complete_deltas": complete_deltas,
        "pairwise_observations": len(pairwise),
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
        "qap_p_one_sided": round(
            qap_p_one,
            10,
        ),
        "qap_p_two_sided": round(
            qap_p_two,
            10,
        ),
        "qap_permutations": qap_n,
        "bootstrap_ci_low": round(
            boot_low,
            8,
        ),
        "bootstrap_ci_high": round(
            boot_high,
            8,
        ),
        "bootstrap_reps": boot_n,
    }


def main():
    print("=" * 78)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 78)

    overlap = load_overlap()

    run_rows = {
        name: read_csv(path)
        for name, path in RUNS.items()
    }

    results = []

    for name, rows in run_rows.items():

        print()
        print(name)
        print("-" * 78)

        strict = analyze_run(
            name,
            rows,
            overlap,
            strict_label,
            "strict",
        )

        sensitivity = analyze_run(
            name,
            rows,
            overlap,
            sensitivity_label,
            "sensitivity",
        )

        results.extend([
            strict,
            sensitivity,
        ])

        print(
            f"Strict: "
            f"valid={strict['valid_outputs']}/840 | "
            f"deltas={strict['complete_deltas']}/420 | "
            f"rho={strict['spearman_rho']} | "
            f"QAP p={strict['qap_p_one_sided']} | "
            f"CI=[{strict['bootstrap_ci_low']}, "
            f"{strict['bootstrap_ci_high']}]"
        )

        print(
            f"Sensitivity: "
            f"valid={sensitivity['valid_outputs']}/840 | "
            f"deltas={sensitivity['complete_deltas']}/420 | "
            f"rho={sensitivity['spearman_rho']} | "
            f"QAP p={sensitivity['qap_p_one_sided']} | "
            f"CI=[{sensitivity['bootstrap_ci_low']}, "
            f"{sensitivity['bootstrap_ci_high']}]"
        )

    # ---------------------------------------------------------
    # Classification agreement with Primary
    # ---------------------------------------------------------

    print()
    print("=" * 78)
    print("LABEL AGREEMENT WITH PRIMARY — STRICT")
    print("=" * 78)

    primary_labels = label_map(
        run_rows["Primary"],
        strict_label,
    )

    for name in ["R1", "R2", "R3"]:
        labels = label_map(
            run_rows[name],
            strict_label,
        )

        rate, n = agreement(
            primary_labels,
            labels,
        )

        print(
            f"Primary vs {name}: "
            f"{rate:.4f} "
            f"over {n} mutually valid classifications"
        )

    # ---------------------------------------------------------
    # Save final summary
    # ---------------------------------------------------------

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUT.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 78)
    print("SAVED")
    print("=" * 78)
    print(
        OUT.relative_to(ROOT)
    )


if __name__ == "__main__":
    main()