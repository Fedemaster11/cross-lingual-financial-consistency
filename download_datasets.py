from pathlib import Path
import hashlib
import json

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi


RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = RAW_DIR / "dataset_manifest.json"


DATASETS = {
    "flame_v2": {
        "repo_id": "Kenpache/multilingual-financial-sentiment-v2",
        "split": "train",
        "filename": "flame_v2.parquet",
        "expected_rows": 145_637,
        "expected_languages": {
            "ar",
            "de",
            "en",
            "es",
            "fr",
            "hi",
            "ja",
            "ko",
            "pt",
            "zh",
        },
        "required_columns": {
            "sentence",
            "label",
            "source",
            "language",
        },
    },
    "financial_eval_7lang": {
        "repo_id": "Kenpache/financial-sentiment-eval-7lang",
        "split": "test",
        "filename": "financial_sentiment_eval_7lang.parquet",
        "expected_rows": 4_993,
        "expected_languages": {
            "en",
            "zh",
            "ja",
            "es",
            "de",
            "fr",
            "ar",
        },
        "required_columns": {
            "row_id",
            "group_id",
            "sentence",
            "label",
            "source",
            "language",
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def validate_dataset(
    name: str,
    df: pd.DataFrame,
    config: dict,
) -> None:
    print(f"\nValidating {name}...")

    missing_columns = (
        config["required_columns"] - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{name}: missing columns "
            f"{sorted(missing_columns)}"
        )

    actual_rows = len(df)

    if actual_rows != config["expected_rows"]:
        raise ValueError(
            f"{name}: expected "
            f"{config['expected_rows']} rows, "
            f"found {actual_rows}"
        )

    actual_languages = set(
        df["language"].dropna().unique()
    )

    if actual_languages != config["expected_languages"]:
        raise ValueError(
            f"{name}: language mismatch.\n"
            f"Expected: "
            f"{sorted(config['expected_languages'])}\n"
            f"Found: {sorted(actual_languages)}"
        )

    if df["sentence"].isna().any():
        raise ValueError(
            f"{name}: missing sentence values found."
        )

    print(f"Rows: {actual_rows:,}")
    print(
        "Languages:",
        ", ".join(sorted(actual_languages)),
    )
    print(
        "Labels:",
        ", ".join(
            sorted(
                str(x)
                for x in df["label"]
                .dropna()
                .unique()
            )
        ),
    )


def download_one(
    name: str,
    config: dict,
    api: HfApi,
) -> dict:
    repo_id = config["repo_id"]

    print("\n" + "=" * 60)
    print(f"Dataset: {name}")
    print(f"Repository: {repo_id}")

    info = api.dataset_info(repo_id)

    revision = info.sha

    print(f"Hugging Face revision: {revision}")

    dataset = load_dataset(
        repo_id,
        split=config["split"],
        revision=revision,
    )

    df = dataset.to_pandas()

    validate_dataset(
        name=name,
        df=df,
        config=config,
    )

    output_path = RAW_DIR / config["filename"]

    df.to_parquet(
        output_path,
        index=False,
    )

    file_hash = sha256_file(output_path)

    print(f"Saved: {output_path}")
    print(f"SHA256: {file_hash}")

    language_counts = (
        df["language"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    label_counts = (
        df["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    return {
        "repo_id": repo_id,
        "revision": revision,
        "split": config["split"],
        "rows": len(df),
        "columns": list(df.columns),
        "languages": sorted(
            df["language"].unique().tolist()
        ),
        "language_counts": {
            str(k): int(v)
            for k, v in language_counts.items()
        },
        "label_counts": {
            str(k): int(v)
            for k, v in label_counts.items()
        },
        "local_file": str(output_path),
        "sha256": file_hash,
    }


def main() -> None:
    api = HfApi()

    manifest = {}

    for name, config in DATASETS.items():
        manifest[name] = download_one(
            name=name,
            config=config,
            api=api,
        )

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

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()