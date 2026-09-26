import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"
TEST_DIR = BASE_DIR / "dataset" / "test"


def inspect_file(file_path):
    print("\n" + "=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    # Read only a small sample first
    sample = pd.read_csv(
        file_path,
        sep="\t",
        nrows=5
    )

    print("\nColumns:")
    print(list(sample.columns))

    print("\nFirst 5 rows:")
    print(sample.to_string(index=False))

    # Count rows without loading the entire dataset
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        row_count = sum(1 for _ in f) - 1

    print(f"\nApproximate rows: {row_count:,}")

    # Missing values from a sample
    print("\nMissing values in first 5 rows:")
    print(sample.isnull().sum())


def main():

    print("\n" + "=" * 70)
    print("AMAZON ML CHALLENGE 2026")
    print("BUSINESS ENTITY RESOLUTION")
    print("=" * 70)

    print("\n\nTRAINING DATA")

    train_files = [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
        "train_ground_truth.tsv",
    ]

    for filename in train_files:
        inspect_file(TRAIN_DIR / filename)

    print("\n\nTEST DATA")

    test_files = [
        "test_source1.tsv",
        "test_source2.tsv",
        "test_source3.tsv",
    ]

    for filename in test_files:
        inspect_file(TEST_DIR / filename)


if __name__ == "__main__":
    main()