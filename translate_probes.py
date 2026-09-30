from pathlib import Path
import json

import pandas as pd
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_ID = "facebook/nllb-200-distilled-600M"

INPUT_PATH = Path("data/final/probes_en.csv")
OUTPUT_PATH = Path("data/interim/translations_pilot.csv")
MANIFEST_PATH = Path("data/interim/translation_manifest.json")

# Pilot only
PILOT_IDS = {"P001", "P002", "P003"}

LANGUAGES = {
    "es": "spa_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ar": "arb_Arab",
}

SOURCE_LANG = "eng_Latn"


def load_model():
    print("=" * 70)
    print("LOADING NLLB TRANSLATION MODEL")
    print("=" * 70)

    api = HfApi()
    info = api.model_info(MODEL_ID)
    revision = info.sha

    print(f"Model: {MODEL_ID}")
    print(f"Revision: {revision}")
    print("Device: CPU")
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

    model.to("cpu")
    model.eval()

    return tokenizer, model, revision


def translate_text(
    text: str,
    target_lang: str,
    tokenizer,
    model,
) -> str:
    tokenizer.src_lang = SOURCE_LANG

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    target_token_id = tokenizer.convert_tokens_to_ids(
        target_lang
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
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing English probes: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    pilot = df[
        df["pair_id"].isin(PILOT_IDS)
    ].copy()

    if len(pilot) != 3:
        raise ValueError(
            f"Expected 3 pilot pairs, found {len(pilot)}"
        )

    tokenizer, model, revision = load_model()

    rows = []

    # First copy English exactly
    for _, row in pilot.iterrows():
        rows.append(
            {
                "pair_id": row["pair_id"],
                "family": row["family"],
                "language": "en",
                "original_text": row["original_en"],
                "counterfactual_text": row[
                    "counterfactual_en"
                ],
                "expected_original": row[
                    "expected_original"
                ],
                "expected_counterfactual": row[
                    "expected_counterfactual"
                ],
            }
        )

    # Translate into six languages
    for language, nllb_code in LANGUAGES.items():

        print()
        print("=" * 70)
        print(
            f"TRANSLATING TO {language.upper()} "
            f"({nllb_code})"
        )
        print("=" * 70)

        for _, row in pilot.iterrows():

            print(
                f"{row['pair_id']} original..."
            )

            original_translation = translate_text(
                row["original_en"],
                nllb_code,
                tokenizer,
                model,
            )

            print(
                f"{row['pair_id']} counterfactual..."
            )

            counterfactual_translation = translate_text(
                row["counterfactual_en"],
                nllb_code,
                tokenizer,
                model,
            )

            rows.append(
                {
                    "pair_id": row["pair_id"],
                    "family": row["family"],
                    "language": language,
                    "original_text": original_translation,
                    "counterfactual_text": (
                        counterfactual_translation
                    ),
                    "expected_original": row[
                        "expected_original"
                    ],
                    "expected_counterfactual": row[
                        "expected_counterfactual"
                    ],
                }
            )

    output = pd.DataFrame(rows)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    manifest = {
        "model_id": MODEL_ID,
        "revision": revision,
        "source_language": SOURCE_LANG,
        "target_languages": LANGUAGES,
        "pilot_pair_ids": sorted(PILOT_IDS),
        "decoding": {
            "num_beams": 4,
            "do_sample": False,
            "max_new_tokens": 128,
        },
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
    print("PILOT TRANSLATION COMPLETE")
    print("=" * 70)

    print(
        f"Rows: {len(output)} "
        "(3 pairs x 7 languages)"
    )

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()