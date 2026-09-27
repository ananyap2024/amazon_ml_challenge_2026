import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ============================================================
# PATH SETUP
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"

sys.path.insert(0, str(SRC_DIR))

from preprocessing import (
    normalize_business_name,
    normalize_address,
    tokenize,
)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_SIZE = 1000
RANDOM_STATE = 42

# Maximum frequency for tokens used in the broad 2-token block
MAX_TOKEN_FREQUENCY = 20_000

# Maximum frequency for a single-token fallback
# This is intentionally much stricter.
SINGLE_TOKEN_MAX_FREQUENCY = 2_000

# Maximum number of address tokens considered for overlap
MAX_ADDRESS_TOKENS = 8

# Minimum number of shared address tokens required for
# the single-name-token fallback.
MIN_ADDRESS_TOKEN_OVERLAP = 2


# ============================================================
# GENERIC BUSINESS TOKENS
# ============================================================

GENERIC_NAME_TOKENS = {
    "private",
    "limited",
    "ltd",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "plc",
    "pvt",
    "public",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)

    s1_path = TRAIN_DIR / "train_source1.tsv"
    s2_path = TRAIN_DIR / "train_source2.tsv"
    s3_path = TRAIN_DIR / "train_source3.tsv"
    gt_path = TRAIN_DIR / "train_ground_truth.tsv"

    print("Loading S1...")
    s1 = pd.read_csv(s1_path, sep="\t")

    print("Loading S2...")
    s2 = pd.read_csv(s2_path, sep="\t")

    print("Loading S3...")
    s3 = pd.read_csv(s3_path, sep="\t")

    print("Loading ground truth...")
    gt = pd.read_csv(gt_path, sep="\t")

    print()
    print(f"S1 rows: {len(s1):,}")
    print(f"S2 rows: {len(s2):,}")
    print(f"S3 rows: {len(s3):,}")
    print(f"GT rows: {len(gt):,}")

    return s1, s2, s3, gt


# ============================================================
# NAME TOKENIZATION
# ============================================================

def tokenize_name(name):

    tokens = tokenize(name)

    useful_tokens = []

    for token in tokens:

        if token in GENERIC_NAME_TOKENS:
            continue

        if len(token) < 3:
            continue

        useful_tokens.append(token)

    return list(set(useful_tokens))


# ============================================================
# ADDRESS TOKENIZATION
# ============================================================

def tokenize_address(address):

    tokens = tokenize(address)

    useful_tokens = []

    for token in tokens:

        if len(token) < 2:
            continue

        useful_tokens.append(token)

    return list(set(useful_tokens))


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_dataframe(df):

    df = df.copy()

    df["name_norm"] = df["business_name"].map(
        normalize_business_name
    )

    df["address_norm"] = df["business_address"].map(
        normalize_address
    )

    df["name_tokens"] = df["name_norm"].map(
        tokenize_name
    )

    df["address_tokens"] = df["address_norm"].map(
        tokenize_address
    )

    return df


# ============================================================
# EXACT INDEX
# ============================================================

def build_exact_index(df, column):

    index = defaultdict(set)

    for value, group in df.groupby(column):

        if not value:
            continue

        for entity_id in group["entity_id"]:
            index[value].add(entity_id)

    return dict(index)


# ============================================================
# TOKEN INDEX
# ============================================================

def build_token_index(df, column):

    index = defaultdict(set)

    for _, row in df.iterrows():

        entity_id = row["entity_id"]

        for token in row[column]:
            index[token].add(entity_id)

    return dict(index)


# ============================================================
# TOKEN FREQUENCY
# ============================================================

def build_token_frequency(index):

    return {
        token: len(entity_ids)
        for token, entity_ids in index.items()
    }


# ============================================================
# SELECT DISTINCTIVE TOKENS
# ============================================================

def select_distinctive_tokens(
    tokens,
    frequency,
    max_frequency,
    max_tokens=2,
):

    valid = [
        token
        for token in tokens
        if token in frequency
        and frequency[token] <= max_frequency
    ]

    valid.sort(
        key=lambda token: frequency[token]
    )

    return valid[:max_tokens]


# ============================================================
# GROUND TRUTH
# ============================================================

def parse_ground_truth(gt):

    ground_truth = {}

    for _, row in gt.iterrows():

        s1_id = row["source1_entity_id"]

        matched = row["matched_entity_ids"]

        if pd.isna(matched):
            matched = ""

        matched = str(matched).strip()

        if not matched:
            ground_truth[s1_id] = set()

        else:
            ground_truth[s1_id] = set(
                x.strip()
                for x in matched.split(",")
                if x.strip()
            )

    return ground_truth


# ============================================================
# HYBRID CANDIDATE GENERATION
# ============================================================

def generate_hybrid_candidates(
    row,
    indexes,
):

    (
        s2_name_index,
        s3_name_index,

        s2_address_index,
        s3_address_index,

        s2_name_token_index,
        s3_name_token_index,

        s2_name_frequency,
        s3_name_frequency,

        s2_address_token_index,
        s3_address_token_index,

        s2_address_frequency,
        s3_address_frequency,
    ) = indexes

    # --------------------------------------------------------
    # Candidate sets
    # --------------------------------------------------------

    s2_candidates = set()
    s3_candidates = set()

    # ========================================================
    # BLOCK 1
    # EXACT NORMALIZED NAME
    # ========================================================

    name = row["name_norm"]

    if name:

        s2_candidates.update(
            s2_name_index.get(
                name,
                set(),
            )
        )

        s3_candidates.update(
            s3_name_index.get(
                name,
                set(),
            )
        )

    # ========================================================
    # BLOCK 2
    # EXACT NORMALIZED ADDRESS
    # ========================================================

    address = row["address_norm"]

    if address:

        s2_candidates.update(
            s2_address_index.get(
                address,
                set(),
            )
        )

        s3_candidates.update(
            s3_address_index.get(
                address,
                set(),
            )
        )

    # ========================================================
    # BLOCK 3
    # TWO DISTINCTIVE NAME TOKENS
    #
    # We retain candidates that appear under either of the
    # two rarest useful tokens.
    # ========================================================

    s2_tokens = select_distinctive_tokens(
        row["name_tokens"],
        s2_name_frequency,
        MAX_TOKEN_FREQUENCY,
        max_tokens=2,
    )

    s3_tokens = select_distinctive_tokens(
        row["name_tokens"],
        s3_name_frequency,
        MAX_TOKEN_FREQUENCY,
        max_tokens=2,
    )

    for token in s2_tokens:

        s2_candidates.update(
            s2_name_token_index.get(
                token,
                set(),
            )
        )

    for token in s3_tokens:

        s3_candidates.update(
            s3_name_token_index.get(
                token,
                set(),
            )
        )

    # ========================================================
    # BLOCK 4
    # SELECTIVE SINGLE-TOKEN + ADDRESS-TOKEN OVERLAP
    #
    # This is NOT applied to every candidate from the common
    # token index.
    #
    # First retrieve candidates using only a relatively rare
    # token (<= 2,000 occurrences).
    #
    # Then require at least MIN_ADDRESS_TOKEN_OVERLAP shared
    # address tokens.
    # ========================================================

    s1_address_tokens = set(
        row["address_tokens"]
    )

    if s1_address_tokens:

        # Keep only the most useful address tokens.
        s1_address_tokens = set(
            sorted(
                s1_address_tokens,
                key=lambda token: (
                    s2_address_frequency.get(
                        token,
                        float("inf")
                    )
                    +
                    s3_address_frequency.get(
                        token,
                        float("inf")
                    )
                ),
            )[:MAX_ADDRESS_TOKENS]
        )

    # --------------------------------------------------------
    # S2 selective single-token candidates
    # --------------------------------------------------------

    s2_rare_tokens = select_distinctive_tokens(
        row["name_tokens"],
        s2_name_frequency,
        SINGLE_TOKEN_MAX_FREQUENCY,
        max_tokens=2,
    )

    for token in s2_rare_tokens:

        candidate_ids = s2_name_token_index.get(
            token,
            set(),
        )

        for candidate_id in candidate_ids:

            # We need address-token information for this
            # candidate. It is retrieved through the reverse
            # address-token index below.
            #
            # Instead of expensive dataframe lookups,
            # use the address-token candidate sets to find
            # overlap.

            overlap_found = False

            for address_token in s1_address_tokens:

                address_candidates = (
                    s2_address_token_index.get(
                        address_token,
                        set(),
                    )
                )

                if candidate_id in address_candidates:

                    overlap_found = True
                    break

            if overlap_found:
                s2_candidates.add(candidate_id)

    # --------------------------------------------------------
    # S3 selective single-token candidates
    # --------------------------------------------------------

    s3_rare_tokens = select_distinctive_tokens(
        row["name_tokens"],
        s3_name_frequency,
        SINGLE_TOKEN_MAX_FREQUENCY,
        max_tokens=2,
    )

    for token in s3_rare_tokens:

        candidate_ids = s3_name_token_index.get(
            token,
            set(),
        )

        for candidate_id in candidate_ids:

            overlap_found = False

            for address_token in s1_address_tokens:

                address_candidates = (
                    s3_address_token_index.get(
                        address_token,
                        set(),
                    )
                )

                if candidate_id in address_candidates:

                    overlap_found = True
                    break

            if overlap_found:
                s3_candidates.add(candidate_id)

    # ========================================================
    # RETURN SOURCE-SEPARATED CANDIDATES
    # ========================================================

    return s2_candidates, s3_candidates


# ============================================================
# BUILD INDEXES
# ============================================================

def build_indexes(s2, s3):

    print()
    print("=" * 70)
    print("BUILDING S2 INDEXES")
    print("=" * 70)

    print("Exact-name index...")
    s2_name_index = build_exact_index(
        s2,
        "name_norm",
    )

    print("Exact-address index...")
    s2_address_index = build_exact_index(
        s2,
        "address_norm",
    )

    print("Name-token index...")
    s2_name_token_index = build_token_index(
        s2,
        "name_tokens",
    )

    print("Address-token index...")
    s2_address_token_index = build_token_index(
        s2,
        "address_tokens",
    )

    print("Name frequencies...")
    s2_name_frequency = build_token_frequency(
        s2_name_token_index
    )

    print("Address frequencies...")
    s2_address_frequency = build_token_frequency(
        s2_address_token_index
    )

    print()
    print("=" * 70)
    print("BUILDING S3 INDEXES")
    print("=" * 70)

    print("Exact-name index...")
    s3_name_index = build_exact_index(
        s3,
        "name_norm",
    )

    print("Exact-address index...")
    s3_address_index = build_exact_index(
        s3,
        "address_norm",
    )

    print("Name-token index...")
    s3_name_token_index = build_token_index(
        s3,
        "name_tokens",
    )

    print("Address-token index...")
    s3_address_token_index = build_token_index(
        s3,
        "address_tokens",
    )

    print("Name frequencies...")
    s3_name_frequency = build_token_frequency(
        s3_name_token_index
    )

    print("Address frequencies...")
    s3_address_frequency = build_token_frequency(
        s3_address_token_index
    )

    print()
    print("All indexes ready.")

    return (
        s2_name_index,
        s3_name_index,

        s2_address_index,
        s3_address_index,

        s2_name_token_index,
        s3_name_token_index,

        s2_name_frequency,
        s3_name_frequency,

        s2_address_token_index,
        s3_address_token_index,

        s2_address_frequency,
        s3_address_frequency,
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    s1_sample,
    ground_truth,
    indexes,
):

    print()
    print("=" * 70)
    print("EVALUATING HYBRID BLOCKING")
    print("=" * 70)

    total_true_matches = 0
    total_retrieved = 0

    candidate_counts = []

    s1_with_candidates = 0
    missed_s1 = 0

    for i, (_, row) in enumerate(
        s1_sample.iterrows(),
        start=1,
    ):

        s1_id = row["entity_id"]

        true_matches = ground_truth.get(
            s1_id,
            set(),
        )

        total_true_matches += len(true_matches)

        s2_candidates, s3_candidates = (
            generate_hybrid_candidates(
                row,
                indexes,
            )
        )

        candidates = (
            s2_candidates
            | s3_candidates
        )

        candidate_count = len(candidates)

        candidate_counts.append(
            candidate_count
        )

        if candidate_count > 0:
            s1_with_candidates += 1

        retrieved = len(
            true_matches & candidates
        )

        total_retrieved += retrieved

        if (
            true_matches
            and retrieved < len(true_matches)
        ):
            missed_s1 += 1

        if i % 100 == 0:

            print(
                f"Processed "
                f"{i:,}/{len(s1_sample):,}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    recall = (
        total_retrieved / total_true_matches
        if total_true_matches
        else 0
    )

    mean_candidates = (
        sum(candidate_counts)
        / len(candidate_counts)
        if candidate_counts
        else 0
    )

    max_candidates = (
        max(candidate_counts)
        if candidate_counts
        else 0
    )

    median_candidates = (
        sorted(candidate_counts)[
            len(candidate_counts) // 2
        ]
        if candidate_counts
        else 0
    )

    print()
    print("=" * 70)
    print("HYBRID BLOCKING RESULTS")
    print("=" * 70)

    print(
        f"S1 validation rows       : "
        f"{len(s1_sample):,}"
    )

    print(
        f"S1 with candidates       : "
        f"{s1_with_candidates:,}"
    )

    print(
        f"S1 rows with missed match: "
        f"{missed_s1:,}"
    )

    print(
        f"Total true matches       : "
        f"{total_true_matches:,}"
    )

    print(
        f"Retrieved true matches   : "
        f"{total_retrieved:,}"
    )

    print(
        f"Missed true matches      : "
        f"{total_true_matches - total_retrieved:,}"
    )

    print(
        f"Blocking recall          : "
        f"{recall * 100:.4f}%"
    )

    print(
        f"Mean candidates / S1     : "
        f"{mean_candidates:,.2f}"
    )

    print(
        f"Median candidates / S1   : "
        f"{median_candidates:,.2f}"
    )

    print(
        f"Maximum candidates / S1  : "
        f"{max_candidates:,}"
    )

    # ========================================================
    # ESTIMATED FULL TEST VOLUME
    # ========================================================

    TEST_S1_COUNT = 1_732_544

    estimated_total = (
        TEST_S1_COUNT
        * mean_candidates
    )

    print()
    print("=" * 70)
    print("ESTIMATED FULL TEST VOLUME")
    print("=" * 70)

    print(
        f"Test S1 rows             : "
        f"{TEST_S1_COUNT:,}"
    )

    print(
        f"Estimated candidates     : "
        f"{estimated_total:,.0f}"
    )

    print()
    print(
        "This estimate is based on the "
        "1,000-row validation sample."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    s1, s2, s3, gt = load_data()

    print()
    print("=" * 70)
    print("PREPARING DATA")
    print("=" * 70)

    print("Preparing S1...")
    s1 = prepare_dataframe(s1)

    print("Preparing S2...")
    s2 = prepare_dataframe(s2)

    print("Preparing S3...")
    s3 = prepare_dataframe(s3)

    # --------------------------------------------------------
    # VALIDATION SAMPLE
    # --------------------------------------------------------

    s1_sample = s1.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    ).copy()

    print()
    print(
        f"Validation sample: "
        f"{len(s1_sample):,}"
    )

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    ground_truth = parse_ground_truth(gt)

    # --------------------------------------------------------
    # INDEXES
    # --------------------------------------------------------

    indexes = build_indexes(
        s2,
        s3,
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    evaluate(
        s1_sample,
        ground_truth,
        indexes,
    )

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()