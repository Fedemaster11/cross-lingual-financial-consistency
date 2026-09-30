import argparse
import csv
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent

URL = "http://127.0.0.1:8080/v1/chat/completions"
HEALTH_URL = "http://127.0.0.1:8080/health"

MODEL = "Qwen/Qwen3-4B-GGUF:Q4_K_M"

LANGUAGES = ["en", "es", "de", "fr", "zh", "ja", "ar"]

LABEL_TO_VALUE = {
    "A": -2,
    "B": -1,
    "C": 0,
    "D": 1,
    "E": 2,
}

EXPECTED_PROMPTS_SHA256 = (
    "0ef49cea5cda449ac32f613eb72ce944d31cbb479f422bfb90bab8bc774fcaae"
)

PROMPTS_PATH = ROOT / "config" / "robustness" / "prompts_r2.json"
OUTPUT_PATH = ROOT / "results" / "robustness" / "r2_strict_prompt_raw.csv"


FIELDNAMES = [
    "task_id",
    "language",
    "pair_id",
    "family",
    "condition",
    "statement",
    "expected_label_value",
    "raw_output",
    "label",
    "label_value",
    "valid_format",
    "seconds",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "model",
    "prompts_sha256",
]


def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_prompts():
    actual_hash = sha256_file(PROMPTS_PATH)

    if actual_hash != EXPECTED_PROMPTS_SHA256:
        raise RuntimeError(
            "Frozen prompts hash does not match.\n"
            f"Expected: {EXPECTED_PROMPTS_SHA256}\n"
            f"Actual:   {actual_hash}"
        )

    with PROMPTS_PATH.open("r", encoding="utf-8") as f:
        prompts = json.load(f)

    if set(prompts.keys()) != set(LANGUAGES):
        raise RuntimeError(
            f"Unexpected prompt languages: {sorted(prompts.keys())}"
        )

    return prompts


def load_tasks():
    tasks = []
    reference_pair_ids = None

    for language in LANGUAGES:
        path = ROOT / "data" / "frozen" / f"probes_{language}.csv"

        if not path.exists():
            raise RuntimeError(f"Missing dataset: {path}")

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) != 60:
            raise RuntimeError(
                f"{language}: expected 60 pairs, found {len(rows)}"
            )

        original_column = f"original_{language}"
        counterfactual_column = f"counterfactual_{language}"

        required = {
            "pair_id",
            "family",
            original_column,
            counterfactual_column,
            "expected_original",
            "expected_counterfactual",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(
                f"{language}: missing columns: {sorted(missing)}"
            )

        pair_ids = []

        for row in rows:
            # Detect malformed CSV rows with unexpected extra fields.
            if None in row:
                raise RuntimeError(
                    f"{language}: malformed CSV row near "
                    f"{row.get('pair_id', 'UNKNOWN')}"
                )

            pair_id = row["pair_id"].strip()
            pair_ids.append(pair_id)

            original = row[original_column].strip()
            counterfactual = row[counterfactual_column].strip()

            if not original or not counterfactual:
                raise RuntimeError(
                    f"{language} {pair_id}: empty statement"
                )

            tasks.append({
                "task_id": f"{language}:{pair_id}:original",
                "language": language,
                "pair_id": pair_id,
                "family": row["family"].strip(),
                "condition": "original",
                "statement": original,
                "expected_label_value": int(
                    row["expected_original"]
                ),
            })

            tasks.append({
                "task_id": f"{language}:{pair_id}:counterfactual",
                "language": language,
                "pair_id": pair_id,
                "family": row["family"].strip(),
                "condition": "counterfactual",
                "statement": counterfactual,
                "expected_label_value": int(
                    row["expected_counterfactual"]
                ),
            })

        if len(set(pair_ids)) != 60:
            raise RuntimeError(
                f"{language}: duplicate pair_id detected"
            )

        if reference_pair_ids is None:
            reference_pair_ids = pair_ids
        elif pair_ids != reference_pair_ids:
            raise RuntimeError(
                f"{language}: pair order/IDs differ from English"
            )

    if len(tasks) != 840:
        raise RuntimeError(
            f"Expected exactly 840 tasks, found {len(tasks)}"
        )

    return tasks


def check_server():
    try:
        with urllib.request.urlopen(
            HEALTH_URL,
            timeout=10
        ) as response:
            data = json.loads(
                response.read().decode("utf-8")
            )
    except Exception as exc:
        raise RuntimeError(
            "llama-server is not reachable at "
            "http://127.0.0.1:8080"
        ) from exc

    if data.get("status") != "ok":
        raise RuntimeError(
            f"Unexpected server status: {data}"
        )


def classify(prompt, statement):
    body = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": statement,
            },
        ],
        "temperature": 0,
        "max_tokens": 12,
        "stream": False,
    }

    request = urllib.request.Request(
        URL,
        data=json.dumps(
            body,
            ensure_ascii=False
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )

    start = time.perf_counter()

    with urllib.request.urlopen(
        request,
        timeout=120
    ) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    elapsed = time.perf_counter() - start

    choice = result["choices"][0]
    raw = choice["message"]["content"].strip()

    # Strict parsing:
    # only an exact A/B/C/D/E counts as valid.
    normalized = raw.upper()

    if normalized in LABEL_TO_VALUE:
        label = normalized
        label_value = LABEL_TO_VALUE[label]
        valid_format = True
    else:
        label = ""
        label_value = ""
        valid_format = False

    usage = result.get("usage", {})

    return {
        "raw_output": raw,
        "label": label,
        "label_value": label_value,
        "valid_format": valid_format,
        "seconds": round(elapsed, 4),
        "finish_reason": choice.get(
            "finish_reason",
            ""
        ),
        "prompt_tokens": usage.get(
            "prompt_tokens",
            ""
        ),
        "completion_tokens": usage.get(
            "completion_tokens",
            ""
        ),
    }


def load_completed():
    if not OUTPUT_PATH.exists():
        return set()

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        reader = csv.DictReader(f)
        return {
            row["task_id"]
            for row in reader
            if row.get("task_id")
        }


def count_results():
    if not OUTPUT_PATH.exists():
        return 0, 0

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    valid = sum(
        row["valid_format"].lower() == "true"
        for row in rows
    )

    return len(rows), valid


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate everything without calling the model.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the next N unfinished tasks.",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("ROBUSTNESS R2 — STRICT FORMAT PROMPT, MAX_TOKENS=12")
    print("=" * 72)

    prompts = load_prompts()

    print("Prompts SHA256: PASS")

    tasks = load_tasks()

    print("Frozen datasets: PASS")
    print(f"Languages:       {len(LANGUAGES)}")
    print("Pairs/language:  60")
    print(f"Total tasks:     {len(tasks)}")

    for language in LANGUAGES:
        n = sum(
            t["language"] == language
            for t in tasks
        )
        print(f"  {language.upper()}: {n}")

    completed = load_completed()

    pending = [
        task
        for task in tasks
        if task["task_id"] not in completed
    ]

    print()
    print(f"Already completed: {len(completed)}")
    print(f"Pending:           {len(pending)}")

    if args.dry_run:
        print()
        print("DRY RUN COMPLETE — NO MODEL CALLS MADE")
        return

    check_server()
    print("llama-server:     PASS")

    if args.limit is not None:
        pending = pending[:args.limit]

    if not pending:
        print()
        print("Nothing left to run.")
        return

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    new_file = not OUTPUT_PATH.exists()

    with OUTPUT_PATH.open(
        "a",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES
        )

        if new_file:
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

        for i, task in enumerate(
            pending,
            start=1
        ):
            print(
                f"[{i}/{len(pending)}] "
                f"{task['language'].upper()} "
                f"{task['pair_id']} "
                f"{task['condition']}...",
                end=" ",
                flush=True,
            )

            result = classify(
                prompts[task["language"]],
                task["statement"],
            )

            output_row = {
                **task,
                **result,
                "model": MODEL,
                "prompts_sha256":
                    EXPECTED_PROMPTS_SHA256,
            }

            writer.writerow(output_row)

            # Durable checkpoint after every inference.
            f.flush()
            os.fsync(f.fileno())

            if result["valid_format"]:
                print(
                    f"{result['label']} "
                    f"({result['seconds']:.2f}s)"
                )
            else:
                print(
                    "INVALID FORMAT "
                    f"{result['raw_output']!r}"
                )

    total_rows, valid_rows = count_results()

    print()
    print("=" * 72)
    print("CHECKPOINT SUMMARY")
    print("=" * 72)
    print(f"Saved observations: {total_rows}/840")
    print(f"Valid A-E outputs:  {valid_rows}/{total_rows}")

    if total_rows == 840:
        print()

        if valid_rows == 840:
            print("MAIN INFERENCE COMPLETE")
        else:
            print(
                "MAIN INFERENCE COMPLETE WITH "
                "INVALID OUTPUTS — INSPECT BEFORE ANALYSIS"
            )
    else:
        print(
            f"Remaining observations: "
            f"{840 - total_rows}"
        )


if __name__ == "__main__":
    main()