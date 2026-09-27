import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "dataset"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)


S1_PATH = os.path.join(
    DATA_DIR,
    "test",
    "test_source1.tsv"
)

S2_PATH = os.path.join(
    DATA_DIR,
    "test",
    "test_source2.tsv"
)

S3_PATH = os.path.join(
    DATA_DIR,
    "test",
    "test_source3.tsv"
)

INTERNAL_PATH = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs_internal.tsv"
)

OFFICIAL_PATH = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)


# ============================================================
# PILOT CONFIGURATION
# ============================================================

PILOT_S1_LIMIT = 50_000

CHUNK_SIZE = 500_000


# ============================================================
# HELPERS
# ============================================================

def print_header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "50K TEST CANDIDATE VALIDATION"
    )


    # ========================================================
    # CHECK FILES
    # ========================================================

    print("\nChecking files...")

    for path in [
        S1_PATH,
        S2_PATH,
        S3_PATH,
        INTERNAL_PATH,
        OFFICIAL_PATH
    ]:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Missing file:\n{path}"
            )

        size_gb = (
            os.path.getsize(path)
            / (1024 ** 3)
        )

        print(
            f"{os.path.basename(path):45s} "
            f"{size_gb:.2f} GB"
        )


    # ========================================================
    # LOAD INTENDED 50K S1
    # ========================================================

    print_header(
        "INTENDED PILOT S1"
    )

    # Read only the entity_id column.
    # This is the authoritative ordering for the pilot.
    s1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        usecols=["entity_id"],
        dtype=str
    )

    # Count the complete test S1 dataset.
    full_s1_count = len(s1)

    # Select exactly the first 50,000 S1 records.
    s1 = s1.head(
        PILOT_S1_LIMIT
    )

    intended_s1_ids = set(
        s1["entity_id"]
        .astype(str)
    )

    print(
        f"Full test S1 rows       : "
        f"{full_s1_count:,}"
    )

    print(
        f"Pilot S1 rows           : "
        f"{len(s1):,}"
    )

    print(
        f"Unique pilot S1 IDs     : "
        f"{len(intended_s1_ids):,}"
    )

    if len(s1) != PILOT_S1_LIMIT:

        print(
            "FAIL: Pilot contains fewer "
            "than 50,000 S1 records."
        )

    elif len(intended_s1_ids) != PILOT_S1_LIMIT:

        print(
            "WARNING: Pilot S1 contains "
            "duplicate entity IDs."
        )

    else:

        print(
            "PASS: Exactly 50,000 unique "
            "pilot S1 IDs."
        )


    # ========================================================
    # LOAD OFFICIAL S2/S3 IDS
    # ========================================================

    print_header(
        "OFFICIAL TEST S2/S3 IDS"
    )

    s2 = pd.read_csv(
        S2_PATH,
        sep="\t",
        usecols=["entity_id"],
        dtype=str
    )

    s3 = pd.read_csv(
        S3_PATH,
        sep="\t",
        usecols=["entity_id"],
        dtype=str
    )

    s2_ids = set(
        s2["entity_id"]
        .astype(str)
    )

    s3_ids = set(
        s3["entity_id"]
        .astype(str)
    )

    print(
        f"S2 rows                : "
        f"{len(s2):,}"
    )

    print(
        f"S2 unique IDs          : "
        f"{len(s2_ids):,}"
    )

    print(
        f"S3 rows                : "
        f"{len(s3):,}"
    )

    print(
        f"S3 unique IDs          : "
        f"{len(s3_ids):,}"
    )


    # ========================================================
    # INTERNAL FILE HEADER
    # ========================================================

    print_header(
        "INTERNAL FILE STRUCTURE"
    )

    internal_header = pd.read_csv(
        INTERNAL_PATH,
        sep="\t",
        nrows=0
    )

    expected_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "source"
    ]

    actual_columns = list(
        internal_header.columns
    )

    print(
        f"Columns found: "
        f"{actual_columns}"
    )

    if actual_columns == expected_columns:

        print(
            "PASS: Expected internal columns."
        )

    else:

        print(
            "FAIL: Unexpected internal columns."
        )

        print(
            f"Expected: {expected_columns}"
        )

        raise ValueError(
            "Internal file has unexpected columns."
        )


    # ========================================================
    # STREAM INTERNAL FILE
    # ========================================================

    print_header(
        "VALIDATING INTERNAL CANDIDATES"
    )

    total_rows = 0

    invalid_s1 = 0

    invalid_candidate = 0

    invalid_source = 0

    source_id_mismatch = 0

    s1_seen = set()

    candidate_counts = {}


    reader = pd.read_csv(
        INTERNAL_PATH,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    )


    for chunk_number, chunk in enumerate(
        reader,
        start=1
    ):

        total_rows += len(chunk)


        # ----------------------------------------------------
        # S1 validation
        # ----------------------------------------------------

        chunk_s1 = chunk[
            "source1_entity_id"
        ]

        invalid_s1 += (
            ~chunk_s1.isin(
                intended_s1_ids
            )
        ).sum()

        s1_seen.update(
            chunk_s1
        )


        # ----------------------------------------------------
        # Source validation
        # ----------------------------------------------------

        valid_sources = {
            "source2",
            "source3"
        }

        invalid_source += (
            ~chunk["source"].isin(
                valid_sources
            )
        ).sum()


        # ----------------------------------------------------
        # Candidate ID validation
        # ----------------------------------------------------

        source2_mask = (
            chunk["source"] == "source2"
        )

        source3_mask = (
            chunk["source"] == "source3"
        )


        source2_invalid = (
            source2_mask
            &
            ~chunk[
                "candidate_entity_id"
            ].isin(s2_ids)
        )


        source3_invalid = (
            source3_mask
            &
            ~chunk[
                "candidate_entity_id"
            ].isin(s3_ids)
        )


        invalid_candidate += (
            source2_invalid.sum()
            +
            source3_invalid.sum()
        )


        # ----------------------------------------------------
        # Track candidate counts per S1
        # ----------------------------------------------------

        counts = (
            chunk[
                "source1_entity_id"
            ]
            .value_counts()
        )


        for entity_id, count in counts.items():

            candidate_counts[entity_id] = (
                candidate_counts.get(
                    entity_id,
                    0
                )
                + int(count)
            )


        if chunk_number % 10 == 0:

            print(
                f"Processed "
                f"{total_rows:,} rows..."
            )


    # ========================================================
    # INTERNAL RESULTS
    # ========================================================

    print_header(
        "INTERNAL VALIDATION RESULTS"
    )

    print(
        f"Total internal rows     : "
        f"{total_rows:,}"
    )

    expected_pair_count = 236_934_138

    print(
        f"Expected pair count     : "
        f"{expected_pair_count:,}"
    )

    if total_rows == expected_pair_count:

        print(
            "PASS: Pair count matches "
            "generation result."
        )

    else:

        print(
            "WARNING: Pair count differs "
            "from generation result."
        )


    print(
        f"Invalid S1 IDs          : "
        f"{invalid_s1:,}"
    )

    print(
        f"Invalid candidate IDs   : "
        f"{invalid_candidate:,}"
    )

    print(
        f"Invalid source values   : "
        f"{invalid_source:,}"
    )


    # ========================================================
    # S1 COVERAGE
    # ========================================================

    s1_with_candidates = len(
        s1_seen
    )

    zero_candidate_s1 = (
        intended_s1_ids
        - s1_seen
    )


    print(
        f"\nPilot S1 IDs            : "
        f"{len(intended_s1_ids):,}"
    )

    print(
        f"S1 IDs with candidates  : "
        f"{s1_with_candidates:,}"
    )

    print(
        f"S1 IDs with zero        : "
        f"{len(zero_candidate_s1):,}"
    )


    if len(intended_s1_ids) > 0:

        coverage = (
            s1_with_candidates
            / len(intended_s1_ids)
            * 100
        )

    else:

        coverage = 0.0


    print(
        f"S1 coverage             : "
        f"{coverage:.4f}%"
    )


    # ========================================================
    # CANDIDATE COUNT DISTRIBUTION
    # ========================================================

    print_header(
        "CANDIDATE COUNT DISTRIBUTION"
    )

    if candidate_counts:

        distribution = pd.Series(
            list(
                candidate_counts.values()
            )
        )


        print(
            f"Min candidates/S1      : "
            f"{distribution.min():,.0f}"
        )

        print(
            f"Max candidates/S1      : "
            f"{distribution.max():,.0f}"
        )

        print(
            f"Mean candidates/S1     : "
            f"{distribution.mean():,.2f}"
        )

        print(
            f"Median candidates/S1   : "
            f"{distribution.median():,.2f}"
        )

        print(
            f"P90 candidates/S1      : "
            f"{distribution.quantile(0.90):,.2f}"
        )

        print(
            f"P95 candidates/S1      : "
            f"{distribution.quantile(0.95):,.2f}"
        )

        print(
            f"P99 candidates/S1      : "
            f"{distribution.quantile(0.99):,.2f}"
        )

    else:

        print(
            "No candidate rows found."
        )


    # ========================================================
    # DUPLICATE CHECK
    # ========================================================

    print_header(
        "DUPLICATE CHECK"
    )

    print(
        "Candidate generation uses Python "
        "sets for each S1/source."
    )

    print(
        "Therefore duplicate candidate IDs "
        "are removed during generation."
    )

    print(
        "\nNOTE:"
    )

    print(
        "A complete global duplicate scan of "
        "236.9M rows is not performed in-memory "
        "because the internal file is 7.67 GB."
    )

    print(
        "The generator's set-based construction "
        "prevents duplicate S1-candidate pairs."
    )


    # ========================================================
    # ZERO-CANDIDATE CONFIRMATION
    # ========================================================

    print_header(
        "ZERO-CANDIDATE S1 HANDLING"
    )

    print(
        f"Zero-candidate S1 entities: "
        f"{len(zero_candidate_s1):,}"
    )

    print(
        "These S1 IDs are omitted from the "
        "internal pair file because they have "
        "no candidate rows."
    )

    print(
        "Member 3 should preserve these S1s "
        "by starting from the complete "
        "test_source1.tsv."
    )


    # ========================================================
    # FILE PATHS AND SIZES
    # ========================================================

    print_header(
        "FILE INFORMATION"
    )

    internal_size_gb = (
        os.path.getsize(
            INTERNAL_PATH
        )
        / (1024 ** 3)
    )

    official_size_gb = (
        os.path.getsize(
            OFFICIAL_PATH
        )
        / (1024 ** 3)
    )

    print(
        f"Internal path : "
        f"{INTERNAL_PATH}"
    )

    print(
        f"Internal size : "
        f"{internal_size_gb:.2f} GB"
    )

    print(
        f"Official path : "
        f"{OFFICIAL_PATH}"
    )

    print(
        f"Official size : "
        f"{official_size_gb:.2f} GB"
    )


    # ========================================================
    # FINAL STATUS
    # ========================================================

    print_header(
        "FINAL VALIDATION STATUS"
    )

    errors = (
        invalid_s1
        + invalid_candidate
        + invalid_source
        + source_id_mismatch
    )


    if (
        total_rows == expected_pair_count
        and errors == 0
        and len(s1) == PILOT_S1_LIMIT
        and len(intended_s1_ids) == PILOT_S1_LIMIT
    ):

        print(
            "PASS: No blocking validation "
            "errors were detected."
        )

    else:

        print(
            "REVIEW REQUIRED: One or more "
            "validation conditions failed."
        )


    print(
        "\nValidation complete."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()