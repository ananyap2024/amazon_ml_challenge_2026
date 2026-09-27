from pathlib import Path

import pandas as pd

from blocking_v3 import (
    prepare_dataframe,
    generate_candidates,
)


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"


def main():

    print("=" * 70)
    print("BLOCKING V3 RECALL TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    NAME_MAX_TOKENS = 3
    NAME_MAX_FREQUENCY = 10000

    ADDRESS_MAX_TOKENS = 2
    ADDRESS_MAX_FREQUENCY = 5000

    print("\nConfiguration:")
    print(f"Name max tokens:       {NAME_MAX_TOKENS}")
    print(f"Name max frequency:    {NAME_MAX_FREQUENCY:,}")
    print(f"Address max tokens:    {ADDRESS_MAX_TOKENS}")
    print(f"Address max frequency: {ADDRESS_MAX_FREQUENCY:,}")

    # ---------------------------------------------------------
    # Ground truth
    # ---------------------------------------------------------

    print("\nLoading ground truth...")

    ground_truth = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t"
    )

    ground_truth_map = dict(
        zip(
            ground_truth["source1_entity_id"],
            ground_truth["matched_entity_ids"]
        )
    )

    print(
        f"Ground truth rows: "
        f"{len(ground_truth):,}"
    )

    # ---------------------------------------------------------
    # S1
    # ---------------------------------------------------------

    print("\nLoading S1...")

    s1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t"
    )

    s1 = s1.sample(
        n=1000,
        random_state=42
    ).copy()

    print(
        f"S1 validation sample: "
        f"{len(s1):,}"
    )

    # ---------------------------------------------------------
    # S2
    # ---------------------------------------------------------

    print("\nLoading S2...")

    s2 = pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t"
    )

    print(f"S2 rows: {len(s2):,}")

    # ---------------------------------------------------------
    # S3
    # ---------------------------------------------------------

    print("\nLoading S3...")

    s3 = pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t"
    )

    print(f"S3 rows: {len(s3):,}")

    # ---------------------------------------------------------
    # Prepare
    # ---------------------------------------------------------

    print("\nPreparing data...")

    s1 = prepare_dataframe(s1)
    s2 = prepare_dataframe(s2)
    s3 = prepare_dataframe(s3)

    print("Preparation complete.")

    # ---------------------------------------------------------
    # Generate candidates
    # ---------------------------------------------------------

    candidates = generate_candidates(
        s1,
        s2,
        s3,
        name_max_tokens=NAME_MAX_TOKENS,
        name_max_frequency=NAME_MAX_FREQUENCY,
        address_max_tokens=ADDRESS_MAX_TOKENS,
        address_max_frequency=ADDRESS_MAX_FREQUENCY
    )

    print("\nCandidate generation complete.")

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    print("\nEvaluating V3 blocking...")

    candidate_counts = []

    total_true_matches = 0
    retrieved_true_matches = 0
    missed_true_matches = 0

    rows_with_candidates = 0
    rows_with_missed_matches = 0

    for _, row in candidates.iterrows():

        candidate_set = set(
            row["candidate_entity_ids"]
        )

        candidate_counts.append(
            len(candidate_set)
        )

        if candidate_set:
            rows_with_candidates += 1

        s1_id = row["source1_entity_id"]

        true_matches_raw = ground_truth_map.get(
            s1_id,
            ""
        )

        if pd.isna(true_matches_raw):
            true_matches_raw = ""

        if true_matches_raw:
            true_matches = set(
                str(true_matches_raw).split(",")
            )
        else:
            true_matches = set()

        total_true_matches += len(true_matches)

        retrieved = (
            true_matches.intersection(candidate_set)
        )

        retrieved_true_matches += len(retrieved)

        missed = (
            true_matches - candidate_set
        )

        missed_true_matches += len(missed)

        if missed:
            rows_with_missed_matches += 1

    candidate_series = pd.Series(
        candidate_counts
    )

    recall = (
        retrieved_true_matches /
        total_true_matches
    )

    print("\n")
    print("=" * 70)
    print("BLOCKING V3 RESULTS")
    print("=" * 70)

    print(
        f"S1 validation rows:       "
        f"{len(s1):,}"
    )

    print(
        f"S1 rows with candidates:  "
        f"{rows_with_candidates:,}"
    )

    print(
        f"S1 rows with missed:      "
        f"{rows_with_missed_matches:,}"
    )

    print()

    print(
        f"True matches:              "
        f"{total_true_matches:,}"
    )

    print(
        f"Retrieved matches:         "
        f"{retrieved_true_matches:,}"
    )

    print(
        f"Missed matches:            "
        f"{missed_true_matches:,}"
    )

    print()

    print(
        f"Blocking recall:           "
        f"{recall:.4f}"
    )

    print(
        f"Blocking recall %:         "
        f"{recall * 100:.2f}%"
    )

    print()

    print(
        f"Mean candidates:           "
        f"{candidate_series.mean():,.2f}"
    )

    print(
        f"Median candidates:         "
        f"{candidate_series.median():,.2f}"
    )

    print(
        f"75th percentile:           "
        f"{candidate_series.quantile(0.75):,.2f}"
    )

    print(
        f"Maximum candidates:        "
        f"{candidate_series.max():,.0f}"
    )


if __name__ == "__main__":
    main()