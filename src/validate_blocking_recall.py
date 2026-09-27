from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)

LABELED_CANDIDATES_FILE = (
    PROJECT_ROOT
    / "output"
    / "training_candidate_pairs_labeled.tsv"
)

S1_LIMIT = 50_000
CHUNK_SIZE = 500_000


# ============================================================
# LOAD THE 50K S1 SAMPLE
# ============================================================

def load_s1_sample():
    """
    Get the first 50K S1 entity IDs from train_source1.tsv.
    """

    s1_file = (
        PROJECT_ROOT
        / "dataset"
        / "train"
        / "train_source1.tsv"
    )

    s1 = pd.read_csv(
        s1_file,
        sep="\t",
        usecols=["entity_id"],
        nrows=S1_LIMIT,
        dtype=str,
    )

    s1_ids = set(s1["entity_id"])

    print(f"S1 entities selected: {len(s1_ids):,}")

    return s1_ids


# ============================================================
# LOAD GROUND TRUTH FOR 50K S1
# ============================================================

def load_ground_truth(s1_ids):
    """
    Build the set of ground-truth pairs for the selected S1 IDs.
    """

    print("\nLoading ground truth...")

    gt = pd.read_csv(
        GROUND_TRUTH_FILE,
        sep="\t",
        dtype=str,
    )

    print(f"Total ground-truth rows: {len(gt):,}")

    gt = gt[gt["source1_entity_id"].isin(s1_ids)].copy()

    print(
        f"Ground-truth rows for selected 50K S1: "
        f"{len(gt):,}"
    )

    ground_truth_pairs = set()

    s1_with_gt = set()

    s2_gt_pairs = set()
    s3_gt_pairs = set()

    for _, row in gt.iterrows():

        s1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if pd.isna(matched_ids) or not str(matched_ids).strip():
            continue

        matches = str(matched_ids).split(",")

        for candidate_id in matches:

            candidate_id = candidate_id.strip()

            if not candidate_id:
                continue

            ground_truth_pairs.add(
                (s1_id, candidate_id)
            )

            s1_with_gt.add(s1_id)

            if candidate_id.startswith("S2-"):
                s2_gt_pairs.add(
                    (s1_id, candidate_id)
                )

            elif candidate_id.startswith("S3-"):
                s3_gt_pairs.add(
                    (s1_id, candidate_id)
                )

    print(
        f"Ground-truth match pairs: "
        f"{len(ground_truth_pairs):,}"
    )

    print(
        f"S1 entities with >=1 ground-truth match: "
        f"{len(s1_with_gt):,}"
    )

    print(
        f"S2 ground-truth pairs: "
        f"{len(s2_gt_pairs):,}"
    )

    print(
        f"S3 ground-truth pairs: "
        f"{len(s3_gt_pairs):,}"
    )

    return (
        ground_truth_pairs,
        s1_with_gt,
        s2_gt_pairs,
        s3_gt_pairs,
    )


# ============================================================
# VALIDATE CANDIDATES
# ============================================================

def validate_candidates(
    s1_ids,
    ground_truth_pairs,
    s1_with_gt,
    s2_gt_pairs,
    s3_gt_pairs,
):

    print("\nLoading candidate pairs in chunks...")

    recovered_pairs = set()

    recovered_s1 = set()

    recovered_s2_pairs = set()
    recovered_s3_pairs = set()

    total_candidate_rows = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            LABELED_CANDIDATES_FILE,
            sep="\t",
            dtype={
                "source1_entity_id": str,
                "candidate_entity_id": str,
                "source": str,
                "label": int,
            },
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):

        total_candidate_rows += len(chunk)

        # We only care about the selected 50K S1 entities.
        chunk = chunk[
            chunk["source1_entity_id"].isin(s1_ids)
        ]

        # label=1 means the candidate is a ground-truth match.
        positive_chunk = chunk[
            chunk["label"] == 1
        ]

        for row in positive_chunk.itertuples(index=False):

            pair = (
                row.source1_entity_id,
                row.candidate_entity_id,
            )

            recovered_pairs.add(pair)

            recovered_s1.add(
                row.source1_entity_id
            )

            if row.candidate_entity_id.startswith("S2-"):
                recovered_s2_pairs.add(pair)

            elif row.candidate_entity_id.startswith("S3-"):
                recovered_s3_pairs.add(pair)

        print(
            f"Processed chunk {chunk_number}: "
            f"{total_candidate_rows:,} candidate rows"
        )

    return (
        recovered_pairs,
        recovered_s1,
        recovered_s2_pairs,
        recovered_s3_pairs,
        total_candidate_rows,
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_recall(
    ground_truth_pairs,
    recovered_pairs,
):
    if not ground_truth_pairs:
        return 0.0

    return (
        len(recovered_pairs & ground_truth_pairs)
        / len(ground_truth_pairs)
        * 100
    )


def main():

    print("=" * 70)
    print("BLOCKING RECALL VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1: Select same 50K S1 entities
    # --------------------------------------------------------

    s1_ids = load_s1_sample()

    # --------------------------------------------------------
    # Step 2: Load corresponding ground truth
    # --------------------------------------------------------

    (
        ground_truth_pairs,
        s1_with_gt,
        s2_gt_pairs,
        s3_gt_pairs,
    ) = load_ground_truth(s1_ids)

    # --------------------------------------------------------
    # Step 3: Validate labeled candidate pairs
    # --------------------------------------------------------

    (
        recovered_pairs,
        recovered_s1,
        recovered_s2_pairs,
        recovered_s3_pairs,
        total_candidate_rows,
    ) = validate_candidates(
        s1_ids,
        ground_truth_pairs,
        s1_with_gt,
        s2_gt_pairs,
        s3_gt_pairs,
    )

    # --------------------------------------------------------
    # Step 4: Calculate pair-level recall
    # --------------------------------------------------------

    recovered_gt_pairs = (
        recovered_pairs & ground_truth_pairs
    )

    missed_pairs = (
        ground_truth_pairs - recovered_pairs
    )

    pair_recall = calculate_recall(
        ground_truth_pairs,
        recovered_pairs,
    )

    # --------------------------------------------------------
    # Step 5: Calculate S1-level recall
    # --------------------------------------------------------

    recovered_s1_with_gt = (
        recovered_s1 & s1_with_gt
    )

    if s1_with_gt:

        s1_recall = (
            len(recovered_s1_with_gt)
            / len(s1_with_gt)
            * 100
        )

    else:

        s1_recall = 0.0

    # --------------------------------------------------------
    # Step 6: Calculate S2/S3 recall
    # --------------------------------------------------------

    s2_recall = calculate_recall(
        s2_gt_pairs,
        recovered_s2_pairs,
    )

    s3_recall = calculate_recall(
        s3_gt_pairs,
        recovered_s3_pairs,
    )

    # --------------------------------------------------------
    # Step 7: Print final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BLOCKING RECALL RESULTS")
    print("=" * 70)

    print(
        f"\nS1 entities evaluated: "
        f"{len(s1_ids):,}"
    )

    print(
        f"Total candidate rows evaluated: "
        f"{total_candidate_rows:,}"
    )

    print("\n" + "-" * 70)
    print("PAIR-LEVEL RECALL")
    print("-" * 70)

    print(
        f"Ground-truth match pairs: "
        f"{len(ground_truth_pairs):,}"
    )

    print(
        f"Recovered ground-truth pairs: "
        f"{len(recovered_gt_pairs):,}"
    )

    print(
        f"Missed ground-truth pairs: "
        f"{len(missed_pairs):,}"
    )

    print(
        f"Pair-level recall: "
        f"{pair_recall:.4f}%"
    )

    print("\n" + "-" * 70)
    print("S1-LEVEL RECALL")
    print("-" * 70)

    print(
        f"S1 entities with >=1 ground-truth match: "
        f"{len(s1_with_gt):,}"
    )

    print(
        f"S1 entities with >=1 recovered match: "
        f"{len(recovered_s1_with_gt):,}"
    )

    print(
        f"S1-level recall: "
        f"{s1_recall:.4f}%"
    )

    print("\n" + "-" * 70)
    print("SOURCE-WISE RECALL")
    print("-" * 70)

    print(
        f"S2 ground-truth pairs: "
        f"{len(s2_gt_pairs):,}"
    )

    print(
        f"S2 recovered pairs: "
        f"{len(recovered_s2_pairs & s2_gt_pairs):,}"
    )

    print(
        f"S2 recall: "
        f"{s2_recall:.4f}%"
    )

    print()

    print(
        f"S3 ground-truth pairs: "
        f"{len(s3_gt_pairs):,}"
    )

    print(
        f"S3 recovered pairs: "
        f"{len(recovered_s3_pairs & s3_gt_pairs):,}"
    )

    print(
        f"S3 recall: "
        f"{s3_recall:.4f}%"
    )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()