from pathlib import Path
import json

import pandas as pd
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_ID = "facebook/nllb-200-distilled-600M"

INPUT_PATH = Path("data/final/probes_en.csv")

OUTPUT_PATH = Path(
    "data/interim/translations_es_nllb_raw.csv"
)

MANIFEST_PATH = Path(
    "data/interim/translation_manifest_es.json"
)

SOURCE_LANG = "eng_Latn"
TARGET_LANG = "spa_Latn"

DEVICE = "cpu"


def load_model():
    print("=" * 70)
    print("LOADING NLLB-200")
    print("=" * 70)

    api = HfApi()
    info = api.model_info(MODEL_ID)

    revision = info.sha

    print(f"Model: {MODEL_ID}")
    print(f"Revision: {revision}")
    print(f"Source: {SOURCE_LANG}")
    print(f"Target: {TARGET_LANG}")
    print(f"Device: {DEVICE}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=revision,
        src_lang=SOURCE_LANG,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        revision=revision,
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model, revision


def translate_text(
    text,
    tokenizer,
    model,
):
    tokenizer.src_lang = SOURCE_LANG

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    target_token_id = tokenizer.convert_tokens_to_ids(
        TARGET_LANG
    )

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=target_token_id,
            max_new_tokens=128,
            num_beams=4,
            do_sample=False,
        )

    translation = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0]

    return translation.strip()


def load_existing():
    if not OUTPUT_PATH.exists():
        return pd.DataFrame()

    existing = pd.read_csv(OUTPUT_PATH)

    print(
        f"Existing checkpoint found: "
        f"{len(existing)} translated pairs"
    )

    return existing


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing English dataset: {INPUT_PATH}"
        )

    english = pd.read_csv(INPUT_PATH)

    if len(english) != 60:
        raise ValueError(
            f"Expected 60 English pairs, "
            f"found {len(english)}"
        )

    existing = load_existing()

    if existing.empty:
        completed_ids = set()
        rows = []
    else:
        completed_ids = set(
            existing["pair_id"]
        )

        rows = existing.to_dict(
            orient="records"
        )

    tokenizer, model, revision = load_model()

    total = len(english)

    for index, row in english.iterrows():

        pair_id = row["pair_id"]

        if pair_id in completed_ids:
            print(
                f"[{index + 1:02d}/{total}] "
                f"{pair_id} already done - skipping"
            )
            continue

        print()
        print(
            f"[{index + 1:02d}/{total}] "
            f"{pair_id}"
        )

        print("  Translating original...")

        original_es = translate_text(
            row["original_en"],
            tokenizer,
            model,
        )

        print("  Translating counterfactual...")

        counterfactual_es = translate_text(
            row["counterfactual_en"],
            tokenizer,
            model,
        )

        translated_row = {
            "pair_id": pair_id,
            "family": row["family"],
            "source_type": row["source_type"],
            "original_direction": (
                row["original_direction"]
            ),
            "original_en": row["original_en"],
            "counterfactual_en": (
                row["counterfactual_en"]
            ),
            "original_es_nllb": original_es,
            "counterfactual_es_nllb": (
                counterfactual_es
            ),
            "expected_original": (
                row["expected_original"]
            ),
            "expected_counterfactual": (
                row["expected_counterfactual"]
            ),
        }

        rows.append(translated_row)

        # Save after every completed pair.
        checkpoint = pd.DataFrame(rows)

        checkpoint.to_csv(
            OUTPUT_PATH,
            index=False,
        )

        print(f"  ES original: {original_es}")
        print(
            f"  ES counterfactual: "
            f"{counterfactual_es}"
        )

    final = pd.DataFrame(rows)

    final = final.sort_values(
        "pair_id"
    ).reset_index(drop=True)

    final.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    if len(final) != 60:
        raise ValueError(
            f"Expected 60 translated pairs, "
            f"found {len(final)}"
        )

    if final["pair_id"].duplicated().any():
        raise ValueError(
            "Duplicate pair IDs found."
        )

    manifest = {
        "translation_stage": "spanish_raw_nllb",
        "model_id": MODEL_ID,
        "revision": revision,
        "source_language": SOURCE_LANG,
        "target_language": TARGET_LANG,
        "input_file": str(INPUT_PATH),
        "output_file": str(OUTPUT_PATH),
        "pairs": len(final),
        "translated_statements": len(final) * 2,
        "decoding": {
            "num_beams": 4,
            "do_sample": False,
            "max_new_tokens": 128,
        },
        "post_editing_applied": False,
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("SPANISH NLLB TRANSLATION COMPLETE")
    print("=" * 70)

    print(f"Pairs: {len(final)}")
    print(
        f"Translated statements: "
        f"{len(final) * 2}"
    )
    print(f"Raw output: {OUTPUT_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print()
    print(
        "IMPORTANT: this is RAW NLLB output. "
        "Do not manually edit this file."
    )


if __name__ == "__main__":
    main()