import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ---------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

SAMPLE_SIZE = 1000

MAX_TOKENS = 2
MAX_FREQUENCY = 20_000

TOP_K_VALUES = [500, 1000, 2000, 5000]

RANDOM_STATE = 42


# ---------------------------------------------------------
# GENERIC BUSINESS-NAME TOKENS
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():
    print("=" * 70)
    print("LOADING DATA")
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

    print(f"S1 rows: {len(s1):,}")
    print(f"S2 rows: {len(s2):,}")
    print(f"S3 rows: {len(s3):,}")
    print(f"GT rows: {len(gt):,}")

    return s1, s2, s3, gt


# ---------------------------------------------------------
# PREPROCESS
# ---------------------------------------------------------

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

    return df


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


# ---------------------------------------------------------
# INDEX BUILDING
# ---------------------------------------------------------

def build_exact_index(df, column):
    index = defaultdict(set)

    for value, group in df.groupby(column):

        if not value:
            continue

        for entity_id in group["entity_id"]:
            index[value].add(entity_id)

    return dict(index)


def build_token_index(df):
    index = defaultdict(set)

    for _, row in df.iterrows():

        entity_id = row["entity_id"]

        for token in row["name_tokens"]:
            index[token].add(entity_id)

    return dict(index)


def build_token_frequency(token_index):
    return {
        token: len(entity_ids)
        for token, entity_ids in token_index.items()
    }


# ---------------------------------------------------------
# SELECT DISTINCTIVE TOKENS
# ---------------------------------------------------------

def select_distinctive_tokens(
    tokens,
    token_frequency,
    max_tokens=MAX_TOKENS,
    max_frequency=MAX_FREQUENCY,
):

    valid_tokens = [
        token
        for token in tokens
        if token in token_frequency
        and token_frequency[token] <= max_frequency
    ]

    valid_tokens.sort(
        key=lambda token: token_frequency[token]
    )

    return valid_tokens[:max_tokens]


# ---------------------------------------------------------
# SCORE CANDIDATES
# ---------------------------------------------------------

def add_scores(score_map, entity_ids, points):
    for entity_id in entity_ids:
        score_map[entity_id] += points


def generate_ranked_candidates(
    s1_row,
    s2_name_index,
    s3_name_index,
    s2_address_index,
    s3_address_index,
    s2_token_index,
    s3_token_index,
    s2_token_frequency,
    s3_token_frequency,
):

    s2_scores = defaultdict(int)
    s3_scores = defaultdict(int)

    # -----------------------------------------------------
    # 1. EXACT NORMALIZED NAME
    # -----------------------------------------------------

    name = s1_row["name_norm"]

    if name:

        add_scores(
            s2_scores,
            s2_name_index.get(name, set()),
            5,
        )

        add_scores(
            s3_scores,
            s3_name_index.get(name, set()),
            5,
        )

    # -----------------------------------------------------
    # 2. EXACT NORMALIZED ADDRESS
    # -----------------------------------------------------

    address = s1_row["address_norm"]

    if address:

        add_scores(
            s2_scores,
            s2_address_index.get(address, set()),
            5,
        )

        add_scores(
            s3_scores,
            s3_address_index.get(address, set()),
            5,
        )

    # -----------------------------------------------------
    # 3. TWO DISTINCTIVE NAME TOKENS
    # -----------------------------------------------------

    selected_tokens_s2 = select_distinctive_tokens(
        s1_row["name_tokens"],
        s2_token_frequency,
    )

    selected_tokens_s3 = select_distinctive_tokens(
        s1_row["name_tokens"],
        s3_token_frequency,
    )

    for token in selected_tokens_s2:

        add_scores(
            s2_scores,
            s2_token_index.get(token, set()),
            1,
        )

    for token in selected_tokens_s3:

        add_scores(
            s3_scores,
            s3_token_index.get(token, set()),
            1,
        )

    # -----------------------------------------------------
    # COMBINE S2 + S3
    # -----------------------------------------------------

    all_scores = {}

    for entity_id, score in s2_scores.items():
        all_scores[entity_id] = score

    for entity_id, score in s3_scores.items():
        all_scores[entity_id] = score

    # -----------------------------------------------------
    # DETERMINISTIC RANKING
    #
    # Higher score first.
    # Entity ID breaks ties deterministically.
    # -----------------------------------------------------

    ranked_candidates = sorted(
        all_scores.items(),
        key=lambda x: (-x[1], x[0]),
    )

    return ranked_candidates


# ---------------------------------------------------------
# GROUND TRUTH
# ---------------------------------------------------------

def parse_ground_truth(gt):
    ground_truth = {}

    for _, row in gt.iterrows():

        s1_id = row["source1_entity_id"]

        matched_ids = row["matched_entity_ids"]

        if pd.isna(matched_ids):
            matched_ids = ""

        matched_ids = str(matched_ids).strip()

        if not matched_ids:
            ground_truth[s1_id] = set()

        else:
            ground_truth[s1_id] = set(
                x.strip()
                for x in matched_ids.split(",")
                if x.strip()
            )

    return ground_truth


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate_top_k(
    s1_sample,
    ground_truth,
    indexes,
):

    (
        s2_name_index,
        s3_name_index,
        s2_address_index,
        s3_address_index,
        s2_token_index,
        s3_token_index,
        s2_token_frequency,
        s3_token_frequency,
    ) = indexes

    total_true_matches = 0

    all_candidate_counts = []

    top_k_stats = {
        k: {
            "retrieved": 0,
            "candidate_counts": [],
            "missed_s1": 0,
        }
        for k in TOP_K_VALUES
    }

    print()
    print("=" * 70)
    print("GENERATING RANKED CANDIDATES")
    print("=" * 70)

    for i, (_, row) in enumerate(s1_sample.iterrows(), start=1):

        s1_id = row["entity_id"]

        true_matches = ground_truth.get(
            s1_id,
            set(),
        )

        total_true_matches += len(true_matches)

        ranked_candidates = generate_ranked_candidates(
            row,
            s2_name_index,
            s3_name_index,
            s2_address_index,
            s3_address_index,
            s2_token_index,
            s3_token_index,
            s2_token_frequency,
            s3_token_frequency,
        )

        # Candidate IDs only
        candidate_ids = [
            entity_id
            for entity_id, score in ranked_candidates
        ]

        all_candidate_counts.append(
            len(candidate_ids)
        )

        # -------------------------------------------------
        # ALL CANDIDATES BASELINE
        # -------------------------------------------------

        all_candidate_set = set(candidate_ids)

        if true_matches:
            # We only count true matches for recall.
            pass

        # -------------------------------------------------
        # TOP-K
        # -------------------------------------------------

        for k in TOP_K_VALUES:

            selected_ids = candidate_ids[:k]

            selected_set = set(selected_ids)

            retrieved = len(
                true_matches & selected_set
            )

            top_k_stats[k]["retrieved"] += retrieved

            top_k_stats[k]["candidate_counts"].append(
                len(selected_ids)
            )

            if true_matches and retrieved < len(true_matches):
                top_k_stats[k]["missed_s1"] += 1

        # Progress
        if i % 100 == 0:
            print(
                f"Processed {i:,}/{len(s1_sample):,} S1 rows"
            )

    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("TOP-K BLOCKING RESULTS")
    print("=" * 70)

    print(
        f"Total true matches: {total_true_matches:,}"
    )

    print()

    print(
        f"{'Configuration':<15}"
        f"{'Recall':>12}"
        f"{'Mean Cand/S1':>18}"
        f"{'Max Cand/S1':>18}"
        f"{'Missed S1':>15}"
    )

    print("-" * 80)

    # -----------------------------------------------------
    # BROAD BLOCKER BASELINE
    # -----------------------------------------------------

    # Calculate broad-block recall.
    #
    # We need to recompute using all candidate counts because
    # Top-K stats only contain the retained candidates.

    broad_retrieved = 0
    broad_missed_s1 = 0

    # Re-run candidate generation only for recall.
    #
    # This is intentionally explicit so the reported baseline
    # is directly comparable with Top-K.

    for _, row in s1_sample.iterrows():

        s1_id = row["entity_id"]

        true_matches = ground_truth.get(
            s1_id,
            set(),
        )

        ranked_candidates = generate_ranked_candidates(
            row,
            s2_name_index,
            s3_name_index,
            s2_address_index,
            s3_address_index,
            s2_token_index,
            s3_token_index,
            s2_token_frequency,
            s3_token_frequency,
        )

        candidate_set = {
            entity_id
            for entity_id, score in ranked_candidates
        }

        retrieved = len(
            true_matches & candidate_set
        )

        broad_retrieved += retrieved

        if true_matches and retrieved < len(true_matches):
            broad_missed_s1 += 1

    broad_recall = (
        broad_retrieved / total_true_matches
        if total_true_matches
        else 0
    )

    print(
        f"{'ALL':<15}"
        f"{broad_recall * 100:>11.2f}%"
        f"{sum(all_candidate_counts) / len(all_candidate_counts):>18,.2f}"
        f"{max(all_candidate_counts):>18,}"
        f"{broad_missed_s1:>15,}"
    )

    # -----------------------------------------------------
    # TOP-K RESULTS
    # -----------------------------------------------------

    for k in TOP_K_VALUES:

        retrieved = top_k_stats[k]["retrieved"]

        recall = (
            retrieved / total_true_matches
            if total_true_matches
            else 0
        )

        counts = top_k_stats[k]["candidate_counts"]

        mean_candidates = (
            sum(counts) / len(counts)
            if counts
            else 0
        )

        max_candidates = (
            max(counts)
            if counts
            else 0
        )

        missed_s1 = top_k_stats[k]["missed_s1"]

        print(
            f"{'TOP-' + str(k):<15}"
            f"{recall * 100:>11.2f}%"
            f"{mean_candidates:>18,.2f}"
            f"{max_candidates:>18,}"
            f"{missed_s1:>15,}"
        )

    # -----------------------------------------------------
    # ESTIMATED FULL TEST VOLUME
    # -----------------------------------------------------

    TEST_S1_COUNT = 1_732_544

    print()
    print("=" * 70)
    print("ESTIMATED FULL TEST CANDIDATE VOLUME")
    print("=" * 70)

    print(
        f"Test S1 rows: {TEST_S1_COUNT:,}"
    )

    for k in TOP_K_VALUES:

        counts = top_k_stats[k]["candidate_counts"]

        mean_candidates = (
            sum(counts) / len(counts)
            if counts
            else 0
        )

        estimated_total = (
            TEST_S1_COUNT * mean_candidates
        )

        print(
            f"Top-{k:<5}: "
            f"{estimated_total:,.0f} candidate relationships"
        )

    print()
    print(
        "NOTE: These are estimates based on the 1,000-row "
        "validation sample."
    )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    s1, s2, s3, gt = load_data()

    print()
    print("=" * 70)
    print("PREPARING DATA")
    print("=" * 70)

    s1 = prepare_dataframe(s1)
    s2 = prepare_dataframe(s2)
    s3 = prepare_dataframe(s3)

    # -----------------------------------------------------
    # VALIDATION SAMPLE
    # -----------------------------------------------------

    s1_sample = s1.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    ).copy()

    print(
        f"Validation S1 sample: {len(s1_sample):,}"
    )

    # -----------------------------------------------------
    # BUILD INDEXES
    # -----------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING S2 INDEXES")
    print("=" * 70)

    print("Building exact-name index...")
    s2_name_index = build_exact_index(
        s2,
        "name_norm",
    )

    print("Building exact-address index...")
    s2_address_index = build_exact_index(
        s2,
        "address_norm",
    )

    print("Building token index...")
    s2_token_index = build_token_index(s2)

    print("Calculating token frequencies...")
    s2_token_frequency = build_token_frequency(
        s2_token_index
    )

    print()
    print("=" * 70)
    print("BUILDING S3 INDEXES")
    print("=" * 70)

    print("Building exact-name index...")
    s3_name_index = build_exact_index(
        s3,
        "name_norm",
    )

    print("Building exact-address index...")
    s3_address_index = build_exact_index(
        s3,
        "address_norm",
    )

    print("Building token index...")
    s3_token_index = build_token_index(s3)

    print("Calculating token frequencies...")
    s3_token_frequency = build_token_frequency(
        s3_token_index
    )

    # -----------------------------------------------------
    # GROUND TRUTH
    # -----------------------------------------------------

    ground_truth = parse_ground_truth(gt)

    indexes = (
        s2_name_index,
        s3_name_index,
        s2_address_index,
        s3_address_index,
        s2_token_index,
        s3_token_index,
        s2_token_frequency,
        s3_token_frequency,
    )

    # -----------------------------------------------------
    # EVALUATE
    # -----------------------------------------------------

    evaluate_top_k(
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