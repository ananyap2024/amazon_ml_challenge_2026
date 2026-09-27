import os
import time
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

TRAIN_DIR = os.path.join(
    DATA_DIR,
    "train"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

GROUND_TRUTH_PATH = os.path.join(
    TRAIN_DIR,
    "train_ground_truth.tsv"
)

CANDIDATE_PATH = os.path.join(
    OUTPUT_DIR,
    "training_candidate_pairs_internal.tsv"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

LABELED_OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "training_candidate_pairs_labeled.tsv"
)

# ------------------------------------------------------------
# CHUNK SIZE
# ------------------------------------------------------------

CHUNK_SIZE = 500_000


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth():

    print("\nLoading ground truth...")

    gt = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        dtype=str
    )

    print(
        f"Ground truth rows: "
        f"{len(gt):,}"
    )

    print(
        f"Ground truth columns: "
        f"{list(gt.columns)}"
    )

    # --------------------------------------------------------
    # Create a set of valid source1 IDs
    # --------------------------------------------------------

    ground_truth = {}

    for row in gt.itertuples(index=False):

        source1_id = row.source1_entity_id

        matched_ids = str(
            row.matched_entity_ids
        )

        if matched_ids == "nan":
            matched_ids = ""

        if matched_ids.strip():

            matched_set = set(
                matched_ids.split(",")
            )

        else:

            matched_set = set()

        ground_truth[
            source1_id
        ] = matched_set

    return ground_truth


# ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("TRAINING CANDIDATE LABELING")
    print("=" * 70)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    if not os.path.exists(
        CANDIDATE_PATH
    ):

        raise FileNotFoundError(
            f"Candidate file not found:\n"
            f"{CANDIDATE_PATH}"
        )

    if not os.path.exists(
        GROUND_TRUTH_PATH
    ):

        raise FileNotFoundError(
            f"Ground truth file not found:\n"
            f"{GROUND_TRUTH_PATH}"
        )

    # --------------------------------------------------------
    # Remove old output
    # --------------------------------------------------------

    if os.path.exists(
        LABELED_OUTPUT_PATH
    ):

        print(
            "\nRemoving previous labeled output..."
        )

        os.remove(
            LABELED_OUTPUT_PATH
        )

    # --------------------------------------------------------
    # Load GT
    # --------------------------------------------------------

    ground_truth = load_ground_truth()

    print(
        f"Ground-truth S1 entities: "
        f"{len(ground_truth):,}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_rows = 0
    positive_rows = 0
    negative_rows = 0

    s1_with_positive = set()

    first_write = True

    # --------------------------------------------------------
    # Read candidate file in chunks
    # --------------------------------------------------------

    print(
        "\nReading candidate file in chunks..."
    )

    candidate_reader = pd.read_csv(
        CANDIDATE_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    )

    for chunk_number, chunk in enumerate(
        candidate_reader,
        start=1
    ):

        # ----------------------------------------------------
        # Label each candidate
        # ----------------------------------------------------

        labels = []

        for row in chunk.itertuples(
            index=False
        ):

            source1_id = (
                row.source1_entity_id
            )

            candidate_id = (
                row.candidate_entity_id
            )

            matched_ids = (
                ground_truth.get(
                    source1_id,
                    set()
                )
            )

            if candidate_id in matched_ids:

                labels.append(1)

            else:

                labels.append(0)

        chunk["label"] = labels

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        chunk_positive = int(
            chunk["label"].sum()
        )

        chunk_total = len(chunk)

        chunk_negative = (
            chunk_total
            -
            chunk_positive
        )

        total_rows += chunk_total
        positive_rows += chunk_positive
        negative_rows += chunk_negative

        # ----------------------------------------------------
        # Positive S1 tracking
        # ----------------------------------------------------

        positive_chunk = chunk[
            chunk["label"] == 1
        ]

        if len(positive_chunk) > 0:

            s1_with_positive.update(
                positive_chunk[
                    "source1_entity_id"
                ].unique()
            )

        # ----------------------------------------------------
        # Write chunk
        # ----------------------------------------------------

        chunk.to_csv(
            LABELED_OUTPUT_PATH,
            sep="\t",
            index=False,
            mode="w" if first_write else "a",
            header=first_write
        )

        first_write = False

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        elapsed = (
            time.time()
            -
            start_time
        )

        print(
            f"Chunk {chunk_number:,} | "
            f"Rows: {total_rows:,} | "
            f"Positive: {positive_rows:,} | "
            f"Negative: {negative_rows:,} | "
            f"Elapsed: {elapsed / 60:.2f} min"
        )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    total_time = (
        time.time()
        -
        start_time
    )

    positive_rate = (
        positive_rows / total_rows * 100
        if total_rows > 0
        else 0
    )

    output_size_gb = (
        os.path.getsize(
            LABELED_OUTPUT_PATH
        )
        /
        (1024 ** 3)
        if os.path.exists(
            LABELED_OUTPUT_PATH
        )
        else 0
    )

    print("\n")
    print("=" * 70)
    print("LABELING COMPLETE")
    print("=" * 70)

    print(
        f"Total candidate rows : "
        f"{total_rows:,}"
    )

    print(
        f"Positive matches     : "
        f"{positive_rows:,}"
    )

    print(
        f"Negative matches     : "
        f"{negative_rows:,}"
    )

    print(
        f"Positive rate        : "
        f"{positive_rate:.6f}%"
    )

    print(
        f"S1 entities with ≥1 positive: "
        f"{len(s1_with_positive):,}"
    )

    print(
        f"Output size          : "
        f"{output_size_gb:.2f} GB"
    )

    print(
        f"Total time           : "
        f"{total_time / 60:.2f} minutes"
    )

    print("\nOutput:")
    print(
        LABELED_OUTPUT_PATH
    )

    print("=" * 70)


if __name__ == "__main__":
    main()