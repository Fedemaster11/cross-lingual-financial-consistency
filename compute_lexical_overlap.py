import csv
import itertools
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent

TOKENIZE_URL = "http://127.0.0.1:8080/tokenize"

LANGUAGES = ["en", "es", "de", "fr", "zh", "ja", "ar"]

OUT_OVERLAP = ROOT / "results" / "lexical_overlap.csv"
OUT_VOCABS = ROOT / "results" / "token_vocabulary_summary.csv"


def tokenize(text):
    body = {
        "content": text,
        "add_special": False,
        "parse_special": False,
        "with_pieces": False,
    }

    request = urllib.request.Request(
        TOKENIZE_URL,
        data=json.dumps(
            body,
            ensure_ascii=False
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(
        request,
        timeout=30
    ) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    tokens = result["tokens"]

    if not isinstance(tokens, list):
        raise RuntimeError(
            f"Unexpected tokenization result: {result}"
        )

    return tokens


def load_statements(language):
    path = (
        ROOT
        / "data"
        / "frozen"
        / f"probes_{language}.csv"
    )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 60:
        raise RuntimeError(
            f"{language}: expected 60 rows, found {len(rows)}"
        )

    original_column = f"original_{language}"
    counterfactual_column = f"counterfactual_{language}"

    statements = []

    for row in rows:
        original = row[original_column].strip()
        counterfactual = row[counterfactual_column].strip()

        if not original or not counterfactual:
            raise RuntimeError(
                f"{language} {row['pair_id']}: empty statement"
            )

        statements.append(original)
        statements.append(counterfactual)

    if len(statements) != 120:
        raise RuntimeError(
            f"{language}: expected 120 statements"
        )

    return statements


def build_vocabulary(language):
    statements = load_statements(language)

    vocab = set()
    total_tokens = 0

    for i, statement in enumerate(
        statements,
        start=1
    ):
        tokens = tokenize(statement)

        vocab.update(tokens)
        total_tokens += len(tokens)

    return {
        "language": language,
        "statements": len(statements),
        "total_tokens": total_tokens,
        "unique_tokens": len(vocab),
        "vocab": vocab,
    }


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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


def main():
    print("=" * 72)
    print("QWEN3-4B SUBWORD VOCABULARY OVERLAP")
    print("=" * 72)

    vocabularies = {}
    summaries = []

    for language in LANGUAGES:
        print(
            f"Tokenizing {language.upper()}...",
            end=" ",
            flush=True,
        )

        info = build_vocabulary(language)

        vocabularies[language] = info["vocab"]

        summaries.append({
            "language": language,
            "statements": info["statements"],
            "total_tokens": info["total_tokens"],
            "unique_tokens": info["unique_tokens"],
        })

        print(
            f"{info['total_tokens']} tokens, "
            f"{info['unique_tokens']} unique"
        )

    overlap_rows = []

    print()
    print("PAIRWISE JACCARD OVERLAP")
    print("-" * 72)

    for lang_a, lang_b in itertools.combinations(
        LANGUAGES,
        2
    ):
        vocab_a = vocabularies[lang_a]
        vocab_b = vocabularies[lang_b]

        intersection = vocab_a & vocab_b
        union = vocab_a | vocab_b

        overlap = (
            len(intersection) / len(union)
            if union
            else 0.0
        )

        row = {
            "language_a": lang_a,
            "language_b": lang_b,
            "vocab_a": len(vocab_a),
            "vocab_b": len(vocab_b),
            "intersection": len(intersection),
            "union": len(union),
            "jaccard_overlap": round(
                overlap,
                8
            ),
        }

        overlap_rows.append(row)

        print(
            f"{lang_a.upper()}-"
            f"{lang_b.upper()} | "
            f"intersection={len(intersection):4} | "
            f"union={len(union):4} | "
            f"overlap={overlap:.6f}"
        )

    if len(overlap_rows) != 21:
        raise RuntimeError(
            f"Expected 21 language pairs, "
            f"found {len(overlap_rows)}"
        )

    write_csv(
        OUT_VOCABS,
        summaries,
        summaries[0].keys(),
    )

    write_csv(
        OUT_OVERLAP,
        overlap_rows,
        overlap_rows[0].keys(),
    )

    print()
    print("=" * 72)
    print("COMPLETE")
    print("=" * 72)
    print(
        "Language vocabularies: "
        f"{OUT_VOCABS.relative_to(ROOT)}"
    )
    print(
        "Pairwise overlap:      "
        f"{OUT_OVERLAP.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()