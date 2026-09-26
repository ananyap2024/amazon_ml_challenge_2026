from pathlib import Path

from blocking import (
    load_sample,
    prepare_dataframe,
    generate_candidates,
)


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"


def main():

    print("=" * 70)
    print("BLOCKING V1 TEST")
    print("=" * 70)

    print("\nLoading samples...")

    s1 = load_sample(
        TRAIN_DIR / "train_source1.tsv",
        nrows=10_000
    )

    s2 = load_sample(
        TRAIN_DIR / "train_source2.tsv",
        nrows=50_000
    )

    s3 = load_sample(
        TRAIN_DIR / "train_source3.tsv",
        nrows=50_000
    )

    print(f"S1 sample: {len(s1):,}")
    print(f"S2 sample: {len(s2):,}")
    print(f"S3 sample: {len(s3):,}")

    print("\nNormalizing...")

    s1 = prepare_dataframe(s1)
    s2 = prepare_dataframe(s2)
    s3 = prepare_dataframe(s3)

    print("Normalization complete.")

    print("\nGenerating candidates...")

    candidates = generate_candidates(
        s1,
        s2,
        s3
    )

    print("\nCandidate generation complete.")

    print("\nFirst 10 results:")

    print(
        candidates.head(10).to_string(
            index=False
        )
    )

    print("\nCandidate statistics:")

    candidate_counts = (
        candidates["candidate_entity_ids"]
        .map(len)
    )

    print(
        candidate_counts.describe()
    )


if __name__ == "__main__":
    main()