from pathlib import Path
import hashlib
import pandas as pd

FROZEN = Path("data/frozen")

LANGS = ["en", "es", "de", "fr", "zh", "ja", "ar"]

EXPECTED_IDS = [f"P{i:03d}" for i in range(1, 61)]

REFERENCE_COLS = [
    "pair_id",
    "family",
    "source_type",
    "original_direction",
    "original_en",
    "counterfactual_en",
    "expected_original",
    "expected_counterfactual",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    dfs = {}

    print("=" * 72)
    print("FROZEN MULTILINGUAL DATASET AUDIT")
    print("=" * 72)

    # ---------------------------------------------------------
    # Load all seven frozen datasets
    # ---------------------------------------------------------
    for lang in LANGS:
        path = FROZEN / f"probes_{lang}.csv"

        assert path.exists(), f"Missing file: {path}"

        df = pd.read_csv(path)

        assert len(df) == 60, f"{lang}: expected 60 rows, found {len(df)}"
        assert df["pair_id"].tolist() == EXPECTED_IDS, (
            f"{lang}: pair_id sequence differs from P001-P060"
        )
        assert df["pair_id"].is_unique, f"{lang}: duplicate pair_id"
        assert not df.isnull().any().any(), f"{lang}: missing values"

        dfs[lang] = df

        print(
            f"{lang.upper():2} | rows={len(df):2} | "
            f"cols={len(df.columns):2} | SHA256={sha256(path)}"
        )

    # ---------------------------------------------------------
    # English is the canonical reference
    # ---------------------------------------------------------
    ref = dfs["en"]

    for col in REFERENCE_COLS:
        assert col in ref.columns, f"English missing required column: {col}"

    # ---------------------------------------------------------
    # Verify shared metadata is identical across languages
    # ---------------------------------------------------------
    for lang in LANGS[1:]:
        df = dfs[lang]

        for col in REFERENCE_COLS:
            assert col in df.columns, f"{lang}: missing column {col}"

            if not df[col].equals(ref[col]):
                mismatches = df.loc[df[col] != ref[col], ["pair_id", col]]
                raise AssertionError(
                    f"{lang}: column '{col}' differs from English reference\n"
                    f"{mismatches}"
                )

    # ---------------------------------------------------------
    # Verify translated columns
    # ---------------------------------------------------------
    for lang in LANGS[1:]:
        df = dfs[lang]

        original_col = f"original_{lang}"
        cf_col = f"counterfactual_{lang}"

        assert original_col in df.columns, (
            f"{lang}: missing {original_col}"
        )
        assert cf_col in df.columns, (
            f"{lang}: missing {cf_col}"
        )

        assert (df[original_col].str.strip() != "").all(), (
            f"{lang}: empty original translation"
        )
        assert (df[cf_col].str.strip() != "").all(), (
            f"{lang}: empty counterfactual translation"
        )

        assert (df[original_col] != df[cf_col]).all(), (
            f"{lang}: original and counterfactual identical in at least one pair"
        )

    # ---------------------------------------------------------
    # Label balance
    # ---------------------------------------------------------
    pos = (ref["expected_original"] == 1).sum()
    neg = (ref["expected_original"] == -1).sum()

    assert pos == 30, f"Expected 30 positive originals, found {pos}"
    assert neg == 30, f"Expected 30 negative originals, found {neg}"

    expected_flip = (
        ref["expected_counterfactual"] == -ref["expected_original"]
    )
    assert expected_flip.all(), "Not all counterfactual labels flip sign"

    # ---------------------------------------------------------
    # Family counts
    # ---------------------------------------------------------
    expected_family_counts = {
        "profit_direction": 8,
        "loss_direction": 8,
        "cost_direction": 8,
        "revenue_direction": 8,
        "earnings_expectations": 8,
        "guidance": 8,
        "margin": 6,
        "contract_realization": 6,
    }

    observed = ref["family"].value_counts().to_dict()

    assert observed == expected_family_counts, (
        f"Unexpected family counts:\n{observed}"
    )

    print()
    print("Family counts:")
    for family, count in expected_family_counts.items():
        print(f"  {family:24} {count}")

    print()
    print("=" * 72)
    print("ALL FROZEN DATASET CHECKS PASSED")
    print("=" * 72)
    print("Languages: 7")
    print("Pairs per language: 60")
    print("Statements per language: 120")
    print("Total translated/test statements:", 60 * 2 * 7)


if __name__ == "__main__":
    main()