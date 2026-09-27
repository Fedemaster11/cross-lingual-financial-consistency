from pathlib import Path
import argparse
import hashlib
import json

import pandas as pd
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_ID = "facebook/nllb-200-distilled-600M"
INPUT_PATH = Path("data/final/probes_en.csv")

SOURCE_LANG = "eng_Latn"
DEVICE = "cpu"
LANGUAGES = {
    "es": "spa_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
}

def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_model(target_code):
    print("=" * 70)
    print("LOADING NLLB-200")
    print("=" * 70)

    api = HfApi()
    info = api.model_info(MODEL_ID)
    revision = info.sha

    print(f"Model: {MODEL_ID}")
    print(f"Revision: {revision}")
    print(f"Source: {SOURCE_LANG}")
    print(f"Target: {target_code}")
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
    target_code,
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
        target_code
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "language",
        choices=LANGUAGES.keys(),
    )

    args = parser.parse_args()

    language = args.language
    target_code = LANGUAGES[language]

    output_path = Path(
        f"data/interim/translations_{language}_nllb_raw.csv"
    )

    manifest_path = Path(
        f"data/interim/translation_manifest_{language}.json"
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_PATH}"
        )

    source_hash = sha256_file(INPUT_PATH)

    english = pd.read_csv(INPUT_PATH)

    if len(english) != 60:
        raise ValueError(
            f"Expected 60 English pairs, found {len(english)}"
        )

    # -----------------------------------------------------
    # Protect against stale checkpoints
    # -----------------------------------------------------

    if output_path.exists():

        if not manifest_path.exists():
            raise RuntimeError(
                "\nExisting translation checkpoint found, "
                "but no manifest exists.\n"
                "Rename or remove the checkpoint before continuing."
            )

        with manifest_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            old_manifest = json.load(f)

        old_hash = old_manifest.get(
            "source_sha256"
        )

        if old_hash != source_hash:
            raise RuntimeError(
                "\nSOURCE DATASET HAS CHANGED.\n"
                f"Checkpoint: {output_path}\n"
                f"Old hash: {old_hash}\n"
                f"New hash: {source_hash}\n\n"
                "Do not reuse this checkpoint."
            )

        existing = pd.read_csv(output_path)

        completed_ids = set(
            existing["pair_id"]
        )

        rows = existing.to_dict(
            orient="records"
        )

        print(
            f"Valid checkpoint found: "
            f"{len(existing)} pairs completed."
        )

    else:
        completed_ids = set()
        rows = []

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    tokenizer, model, revision = load_model(
        target_code
    )

    # Save manifest BEFORE translation so checkpoint
    # always has a corresponding source hash.

    manifest = {
        "translation_stage": (
            f"{language}_raw_nllb"
        ),
        "model_id": MODEL_ID,
        "revision": revision,
        "source_language": SOURCE_LANG,
        "target_language": target_code,
        "language": language,
        "input_file": str(INPUT_PATH),
        "source_sha256": source_hash,
        "output_file": str(output_path),
        "expected_pairs": 60,
        "expected_translated_statements": 120,
        "decoding": {
            "num_beams": 4,
            "do_sample": False,
            "max_new_tokens": 128,
        },
        "post_editing_applied": False,
    }

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # Translation
    # -----------------------------------------------------

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

        original_translation = translate_text(
            row["original_en"],
            target_code,
            tokenizer,
            model,
        )

        print("  Translating counterfactual...")

        counterfactual_translation = translate_text(
            row["counterfactual_en"],
            target_code,
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
            "original_en": (
                row["original_en"]
            ),
            "counterfactual_en": (
                row["counterfactual_en"]
            ),
            f"original_{language}_nllb": (
                original_translation
            ),
            f"counterfactual_{language}_nllb": (
                counterfactual_translation
            ),
            "expected_original": (
                row["expected_original"]
            ),
            "expected_counterfactual": (
                row["expected_counterfactual"]
            ),
        }

        rows.append(translated_row)

        checkpoint = pd.DataFrame(rows)

        checkpoint.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"  {language.upper()} original: "
            f"{original_translation}"
        )

        print(
            f"  {language.upper()} counterfactual: "
            f"{counterfactual_translation}"
        )

    # -----------------------------------------------------
    # Final validation
    # -----------------------------------------------------

    final = pd.DataFrame(rows)

    final = final.sort_values(
        "pair_id"
    ).reset_index(drop=True)

    if len(final) != 60:
        raise ValueError(
            f"Expected 60 translated pairs, "
            f"found {len(final)}"
        )

    if final["pair_id"].duplicated().any():
        raise ValueError(
            "Duplicate pair IDs detected."
        )

    final.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print(
        f"{language.upper()} NLLB TRANSLATION COMPLETE"
    )
    print("=" * 70)

    print("Pairs: 60")
    print("Translated statements: 120")
    print(f"Source SHA256: {source_hash}")
    print(f"Raw output: {output_path}")
    print(f"Manifest: {manifest_path}")

    print()
    print(
        "RAW NLLB OUTPUT. "
        "DO NOT MANUALLY EDIT THIS FILE."
    )


if __name__ == "__main__":
    main()