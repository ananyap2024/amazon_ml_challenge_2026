from pathlib import Path

from blocking_v2 import (
    load_sample,
    prepare_dataframe,
    generate_candidates,
)


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset" / "train"


def main():

    print("=" * 70)
    print("BLOCKING V2.1 TEST")
    print("=" * 70)

    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------

    MAX_TOKENS = 3
    MAX_FREQUENCY = 5000

    print("\nConfiguration:")

    print(
        f"Maximum tokens per S1: "
        f"{MAX_TOKENS}"
    )

    print(
        f"Maximum token frequency: "
        f"{MAX_FREQUENCY:,}"
    )

    # --------------------------------------------------
    # Load samples
    # --------------------------------------------------

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

    print(
        f"S1 sample: {len(s1):,}"
    )

    print(
        f"S2 sample: {len(s2):,}"
    )

    print(
        f"S3 sample: {len(s3):,}"
    )

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------

    print("\nPreparing data...")

    s1 = prepare_dataframe(s1)
    s2 = prepare_dataframe(s2)
    s3 = prepare_dataframe(s3)

    print(
        "Preparation complete."
    )

    # --------------------------------------------------
    # Generate candidates
    # --------------------------------------------------

    print(
        "\nGenerating V2.1 candidates..."
    )

    candidates = generate_candidates(
        s1,
        s2,
        s3,
        max_tokens=MAX_TOKENS,
        max_frequency=MAX_FREQUENCY
    )

    print(
        "\nCandidate generation complete."
    )

    # --------------------------------------------------
    # Candidate statistics
    # --------------------------------------------------

    candidate_counts = (
        candidates["candidate_entity_ids"]
        .map(len)
    )

    print("\nCandidate statistics:")

    print(
        candidate_counts.describe()
    )

    # --------------------------------------------------
    # Additional statistics
    # --------------------------------------------------

    print("\nAdditional statistics:")

    print(
        f"S1 rows with candidates: "
        f"{(candidate_counts > 0).sum():,}"
    )

    print(
        f"S1 rows without candidates: "
        f"{(candidate_counts == 0).sum():,}"
    )

    print(
        f"Mean candidates per S1: "
        f"{candidate_counts.mean():.2f}"
    )

    print(
        f"Median candidates per S1: "
        f"{candidate_counts.median():.2f}"
    )

    print(
        f"Maximum candidates for one S1: "
        f"{candidate_counts.max():,.0f}"
    )

    # --------------------------------------------------
    # First 10 results
    # --------------------------------------------------

    print("\nFirst 10 results:")

    for _, row in candidates.head(10).iterrows():

        candidate_ids = row[
            "candidate_entity_ids"
        ]

        print(
            f"{row['source1_entity_id']}: "
            f"{len(candidate_ids):,} candidates"
        )


if __name__ == "__main__":
    main()