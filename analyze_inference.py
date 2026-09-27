import csv
import itertools
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent

INPUT = ROOT / "results" / "main_inference_raw.csv"

STRICT_DELTAS = ROOT / "results" / "pair_language_deltas_strict.csv"
SENSITIVITY_DELTAS = ROOT / "results" / "pair_language_deltas_sensitivity.csv"

INVALID_OUTPUTS = ROOT / "results" / "invalid_outputs.csv"

STRICT_PAIRWISE = ROOT / "results" / "pairwise_disagreement_strict.csv"
SENSITIVITY_PAIRWISE = ROOT / "results" / "pairwise_disagreement_sensitivity.csv"

LANGUAGE_SUMMARY = ROOT / "results" / "classification_summary_by_language.csv"

LANGUAGES = ["en", "es", "de", "fr", "zh", "ja", "ar"]

LABEL_TO_VALUE = {
    "A": -2,
    "B": -1,
    "C": 0,
    "D": 1,
    "E": 2,
}


def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def read_rows():
    with INPUT.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 840:
        raise RuntimeError(
            f"Expected 840 raw observations, found {len(rows)}"
        )

    task_ids = [row["task_id"] for row in rows]

    if len(set(task_ids)) != 840:
        raise RuntimeError("Duplicate task_id detected")

    return rows


def strict_label(row):
    """
    Primary analysis:
    accept only outputs that were valid under the
    pre-specified exact A-E parser.
    """
    if row["valid_format"].strip().lower() != "true":
        return None

    label = row["label"].strip().upper()

    if label not in LABEL_TO_VALUE:
        return None

    return label


def sensitivity_label(row):
    """
    Sensitivity analysis:
    first use strict label if available.

    Otherwise recover only outputs whose FIRST meaningful
    character is unambiguously A-E, followed by whitespace,
    punctuation, dash, or end of string.
    """
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


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)


def build_invalid_outputs(rows):
    invalid = [
        row for row in rows
        if strict_label(row) is None
    ]

    write_csv(
        INVALID_OUTPUTS,
        invalid,
        rows[0].keys(),
    )

    return invalid


def build_classification_summary(rows):
    summary = []

    for language in LANGUAGES:
        lang_rows = [
            r for r in rows
            if r["language"] == language
        ]

        strict_valid = 0
        sensitivity_valid = 0

        strict_direction_correct = 0
        sensitivity_direction_correct = 0

        for row in lang_rows:
            expected = int(row["expected_label_value"])

            s_label = strict_label(row)

            if s_label is not None:
                strict_valid += 1

                value = LABEL_TO_VALUE[s_label]

                if sign(value) == sign(expected):
                    strict_direction_correct += 1

            sens_label = sensitivity_label(row)

            if sens_label is not None:
                sensitivity_valid += 1

                value = LABEL_TO_VALUE[sens_label]

                if sign(value) == sign(expected):
                    sensitivity_direction_correct += 1

        summary.append({
            "language": language,
            "n_total": len(lang_rows),
            "strict_valid": strict_valid,
            "strict_invalid": len(lang_rows) - strict_valid,
            "strict_direction_correct": strict_direction_correct,
            "strict_direction_accuracy_valid_only": (
                round(
                    strict_direction_correct / strict_valid,
                    6
                )
                if strict_valid
                else ""
            ),
            "sensitivity_valid": sensitivity_valid,
            "sensitivity_direction_correct":
                sensitivity_direction_correct,
            "sensitivity_direction_accuracy": (
                round(
                    sensitivity_direction_correct /
                    sensitivity_valid,
                    6
                )
                if sensitivity_valid
                else ""
            ),
        })

    write_csv(
        LANGUAGE_SUMMARY,
        summary,
        summary[0].keys(),
    )

    return summary


def build_deltas(rows, parser, output_path):
    grouped = defaultdict(dict)

    for row in rows:
        key = (
            row["language"],
            row["pair_id"],
        )

        grouped[key][row["condition"]] = row

    output = []

    for language in LANGUAGES:
        for n in range(1, 61):
            pair_id = f"P{n:03d}"
            key = (language, pair_id)

            pair = grouped.get(key)

            if pair is None:
                raise RuntimeError(
                    f"Missing pair {language} {pair_id}"
                )

            if (
                "original" not in pair
                or "counterfactual" not in pair
            ):
                raise RuntimeError(
                    f"Incomplete raw pair {language} {pair_id}"
                )

            original_row = pair["original"]
            cf_row = pair["counterfactual"]

            original_label = parser(original_row)
            cf_label = parser(cf_row)

            expected_original = int(
                original_row["expected_label_value"]
            )
            expected_cf = int(
                cf_row["expected_label_value"]
            )

            expected_delta = (
                expected_cf - expected_original
            )

            if (
                original_label is None
                or cf_label is None
            ):
                original_value = (
                    LABEL_TO_VALUE[original_label]
                    if original_label is not None
                    else ""
                )

                cf_value = (
                    LABEL_TO_VALUE[cf_label]
                    if cf_label is not None
                    else ""
                )

                delta = ""
                delta_direction_correct = ""
                complete = False

            else:
                original_value = LABEL_TO_VALUE[
                    original_label
                ]

                cf_value = LABEL_TO_VALUE[
                    cf_label
                ]

                delta = (
                    cf_value - original_value
                )

                delta_direction_correct = (
                    sign(delta)
                    == sign(expected_delta)
                )

                complete = True

            output.append({
                "language": language,
                "pair_id": pair_id,
                "family": original_row["family"],
                "original_label": (
                    original_label or ""
                ),
                "original_value": original_value,
                "counterfactual_label": (
                    cf_label or ""
                ),
                "counterfactual_value": cf_value,
                "delta": delta,
                "expected_original": expected_original,
                "expected_counterfactual": expected_cf,
                "expected_delta": expected_delta,
                "delta_direction_correct":
                    delta_direction_correct,
                "complete": complete,
            })

    write_csv(
        output_path,
        output,
        output[0].keys(),
    )

    return output


def build_pairwise_disagreement(
    deltas,
    output_path
):
    lookup = {
        (row["pair_id"], row["language"]): row
        for row in deltas
    }

    output = []

    for pair_id in [
        f"P{i:03d}" for i in range(1, 61)
    ]:
        for lang_a, lang_b in itertools.combinations(
            LANGUAGES,
            2
        ):
            row_a = lookup[(pair_id, lang_a)]
            row_b = lookup[(pair_id, lang_b)]

            complete = (
                row_a["complete"]
                and row_b["complete"]
            )

            if complete:
                delta_a = int(row_a["delta"])
                delta_b = int(row_b["delta"])

                disagreement = abs(
                    delta_a - delta_b
                )

                exact_match = (
                    delta_a == delta_b
                )
            else:
                delta_a = ""
                delta_b = ""
                disagreement = ""
                exact_match = ""

            output.append({
                "pair_id": pair_id,
                "family": row_a["family"],
                "language_a": lang_a,
                "language_b": lang_b,
                "delta_a": delta_a,
                "delta_b": delta_b,
                "disagreement": disagreement,
                "exact_delta_match": exact_match,
                "complete": complete,
            })

    write_csv(
        output_path,
        output,
        output[0].keys(),
    )

    return output


def summarize_pairwise(rows):
    grouped = defaultdict(list)

    for row in rows:
        if not row["complete"]:
            continue

        key = (
            row["language_a"],
            row["language_b"],
        )

        grouped[key].append(row)

    summaries = []

    for key in itertools.combinations(
        LANGUAGES,
        2
    ):
        values = grouped[key]

        disagreements = [
            int(r["disagreement"])
            for r in values
        ]

        exact_matches = [
            r["exact_delta_match"]
            for r in values
        ]

        mean_d = (
            sum(disagreements)
            / len(disagreements)
            if disagreements
            else None
        )

        exact_rate = (
            sum(bool(x) for x in exact_matches)
            / len(exact_matches)
            if exact_matches
            else None
        )

        summaries.append({
            "language_a": key[0],
            "language_b": key[1],
            "n_complete_pairs": len(values),
            "mean_disagreement": (
                round(mean_d, 6)
                if mean_d is not None
                else ""
            ),
            "exact_delta_match_rate": (
                round(exact_rate, 6)
                if exact_rate is not None
                else ""
            ),
        })

    return summaries


def main():
    print("=" * 72)
    print("ANALYZE FROZEN INFERENCE RESULTS")
    print("=" * 72)

    rows = read_rows()

    print(f"Raw observations:       {len(rows)}")

    invalid = build_invalid_outputs(rows)

    print(f"Strict invalid outputs: {len(invalid)}")

    summary = build_classification_summary(rows)

    strict_deltas = build_deltas(
        rows,
        strict_label,
        STRICT_DELTAS,
    )

    sensitivity_deltas = build_deltas(
        rows,
        sensitivity_label,
        SENSITIVITY_DELTAS,
    )

    strict_complete = sum(
        row["complete"]
        for row in strict_deltas
    )

    sensitivity_complete = sum(
        row["complete"]
        for row in sensitivity_deltas
    )

    print(
        f"Strict complete deltas: {strict_complete}/420"
    )

    print(
        "Sensitivity complete "
        f"deltas: {sensitivity_complete}/420"
    )

    strict_pairwise = build_pairwise_disagreement(
        strict_deltas,
        STRICT_PAIRWISE,
    )

    sensitivity_pairwise = (
        build_pairwise_disagreement(
            sensitivity_deltas,
            SENSITIVITY_PAIRWISE,
        )
    )

    strict_pairwise_complete = sum(
        row["complete"]
        for row in strict_pairwise
    )

    sensitivity_pairwise_complete = sum(
        row["complete"]
        for row in sensitivity_pairwise
    )

    print(
        "Strict pairwise observations: "
        f"{strict_pairwise_complete}/1260"
    )

    print(
        "Sensitivity pairwise observations: "
        f"{sensitivity_pairwise_complete}/1260"
    )

    print()
    print("STRICT LANGUAGE SUMMARY")
    print("-" * 72)

    for row in summary:
        print(
            f"{row['language'].upper():2} | "
            f"valid={row['strict_valid']:3}/120 | "
            f"direction_correct="
            f"{row['strict_direction_correct']:3} | "
            f"accuracy="
            f"{row['strict_direction_accuracy_valid_only']}"
        )

    print()
    print("STRICT PAIRWISE SUMMARY")
    print("-" * 72)

    for row in summarize_pairwise(
        strict_pairwise
    ):
        print(
            f"{row['language_a'].upper()}-"
            f"{row['language_b'].upper()} | "
            f"n={row['n_complete_pairs']:2} | "
            f"mean_D="
            f"{row['mean_disagreement']} | "
            f"exact_match="
            f"{row['exact_delta_match_rate']}"
        )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    for path in [
        INVALID_OUTPUTS,
        LANGUAGE_SUMMARY,
        STRICT_DELTAS,
        SENSITIVITY_DELTAS,
        STRICT_PAIRWISE,
        SENSITIVITY_PAIRWISE,
    ]:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()