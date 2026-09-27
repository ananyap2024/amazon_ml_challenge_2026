import os
import re
import time
from collections import defaultdict

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
    "output",
    "blocking_optimization_50k"
)

S1_LIMIT = 50_000

S1_PATH = os.path.join(
    TRAIN_DIR,
    "train_source1.tsv"
)

S2_PATH = os.path.join(
    TRAIN_DIR,
    "train_source2.tsv"
)

S3_PATH = os.path.join(
    TRAIN_DIR,
    "train_source3.tsv"
)

GROUND_TRUTH_PATH = os.path.join(
    TRAIN_DIR,
    "train_ground_truth.tsv"
)


# ============================================================
# EXPERIMENTS
# ============================================================

EXPERIMENTS = [
    {
        "name": "20k_7.5k",
        "pass1_max_frequency": 20_000,
        "pass2_max_frequency": 7_500,
    },
    {
        "name": "20k_10k",
        "pass1_max_frequency": 20_000,
        "pass2_max_frequency": 10_000,
    },
    {
        "name": "20k_12.5k",
        "pass1_max_frequency": 20_000,
        "pass2_max_frequency": 12_500,
    },
    {
        "name": "20k_15k",
        "pass1_max_frequency": 20_000,
        "pass2_max_frequency": 15_000,
    },
]


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = re.sub(
        r"[^a-z0-9\s]",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def tokenize(value):

    if not value:
        return []

    return value.split()


# ============================================================
# LOAD SOURCE
# ============================================================

def load_source(path, source_name):

    print(
        f"\nLoading {source_name}..."
    )

    df = pd.read_csv(
        path,
        sep="\t"
    )

    print(
        f"{source_name} rows: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# PREPARE SOURCE
# ============================================================

def prepare_source(df, source_name):

    result = pd.DataFrame()

    result["entity_id"] = (
        df["entity_id"]
        .astype(str)
    )

    result["business_name"] = (
        df["business_name"]
        .fillna("")
        .astype(str)
    )

    result["normalized_name"] = (
        result["business_name"]
        .map(normalize_text)
    )

    result["tokens"] = (
        result["normalized_name"]
        .map(tokenize)
    )

    print(
        f"{source_name} prepared: "
        f"{len(result):,} rows"
    )

    return result


# ============================================================
# TOKEN FREQUENCY
# ============================================================

def build_token_frequency(source_df):

    token_frequency = defaultdict(int)

    for tokens in source_df["tokens"]:

        for token in set(tokens):

            token_frequency[token] += 1

    return token_frequency


# ============================================================
# SELECT DISTINCTIVE TOKENS
#
# THIS IS COPIED FROM THE REAL GENERATOR LOGIC.
# ============================================================

def select_tokens(
    tokens,
    token_frequency,
    max_tokens,
    max_frequency
):

    candidates = []

    for token in set(tokens):

        frequency = token_frequency.get(
            token,
            0
        )

        if frequency <= max_frequency:

            candidates.append(
                (
                    frequency,
                    token
                )
            )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1]
        )
    )

    return [
        token
        for _, token in candidates[:max_tokens]
    ]


# ============================================================
# EXACT NAME INDEX
# ============================================================

def build_exact_name_index(source_df):

    index = defaultdict(set)

    for row in source_df.itertuples(
        index=False
    ):

        name = row.normalized_name

        if name:

            index[name].add(
                row.entity_id
            )

    return index


# ============================================================
# TOKEN INDEX
#
# We build ONE complete token index.
# Frequency filtering is handled during token selection,
# exactly as in the real generator.
# ============================================================

def build_token_index(source_df):

    token_index = defaultdict(set)

    for row in source_df.itertuples(
        index=False
    ):

        for token in set(row.tokens):

            token_index[token].add(
                row.entity_id
            )

    return token_index


# ============================================================
# GENERATE CANDIDATES FOR ONE S1
#
# THIS REPRODUCES THE ACTUAL GENERATOR.
# ============================================================

def generate_for_s1(
    s1_row,
    exact_index,
    token_index,
    token_frequency,
    s2_entity_ids,
    s3_entity_ids,
    pass1_max_frequency,
    pass2_max_frequency
):

    candidates_s2 = set()
    candidates_s3 = set()

    normalized_name = (
        s1_row.normalized_name
    )

    tokens = s1_row.tokens

    # ========================================================
    # EXACT NAME
    # ========================================================

    if normalized_name:

        exact_matches = (
            exact_index.get(
                normalized_name,
                set()
            )
        )

        for entity_id in exact_matches:

            if entity_id in s2_entity_ids:

                candidates_s2.add(
                    entity_id
                )

            elif entity_id in s3_entity_ids:

                candidates_s3.add(
                    entity_id
                )

    # ========================================================
    # PASS 1
    #
    # 1 distinctive token
    # ========================================================

    pass1_tokens = select_tokens(
        tokens=tokens,
        token_frequency=token_frequency,
        max_tokens=1,
        max_frequency=pass1_max_frequency
    )

    for token in pass1_tokens:

        matching_ids = (
            token_index.get(
                token,
                set()
            )
        )

        for entity_id in matching_ids:

            if entity_id in s2_entity_ids:

                candidates_s2.add(
                    entity_id
                )

            elif entity_id in s3_entity_ids:

                candidates_s3.add(
                    entity_id
                )

    # ========================================================
    # PASS 2
    #
    # 2 distinctive tokens
    #
    # IMPORTANT:
    # This is UNION, NOT INTERSECTION.
    # ========================================================

    pass2_tokens = select_tokens(
        tokens=tokens,
        token_frequency=token_frequency,
        max_tokens=2,
        max_frequency=pass2_max_frequency
    )

    for token in pass2_tokens:

        matching_ids = (
            token_index.get(
                token,
                set()
            )
        )

        for entity_id in matching_ids:

            if entity_id in s2_entity_ids:

                candidates_s2.add(
                    entity_id
                )

            elif entity_id in s3_entity_ids:

                candidates_s3.add(
                    entity_id
                )

    return (
        candidates_s2,
        candidates_s3
    )


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(s1_ids):

    print(
        "\nLoading ground truth..."
    )

    gt = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        dtype=str
    )

    gt = gt[
        gt[
            "source1_entity_id"
        ].isin(s1_ids)
    ]

    ground_truth_pairs = set()
    ground_truth_s1 = set()

    s2_ground_truth = set()
    s3_ground_truth = set()

    for row in gt.itertuples(
        index=False
    ):

        s1_id = (
            row.source1_entity_id
        )

        matched_ids = (
            row.matched_entity_ids
        )

        if (
            pd.isna(matched_ids)
            or not str(
                matched_ids
            ).strip()
        ):
            continue

        for candidate_id in str(
            matched_ids
        ).split(","):

            candidate_id = (
                candidate_id.strip()
            )

            if not candidate_id:
                continue

            pair = (
                s1_id,
                candidate_id
            )

            ground_truth_pairs.add(
                pair
            )

            ground_truth_s1.add(
                s1_id
            )

            if candidate_id.startswith(
                "S2-"
            ):

                s2_ground_truth.add(
                    pair
                )

            elif candidate_id.startswith(
                "S3-"
            ):

                s3_ground_truth.add(
                    pair
                )

    print(
        f"Ground-truth pairs: "
        f"{len(ground_truth_pairs):,}"
    )

    print(
        f"S1 entities with GT matches: "
        f"{len(ground_truth_s1):,}"
    )

    return (
        ground_truth_pairs,
        ground_truth_s1,
        s2_ground_truth,
        s3_ground_truth
    )


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(
    experiment,
    s1,
    exact_index,
    token_index,
    token_frequency,
    s2_entity_ids,
    s3_entity_ids,
    ground_truth_pairs,
    ground_truth_s1,
    s2_ground_truth,
    s3_ground_truth
):

    name = experiment["name"]

    pass1_max_frequency = (
        experiment[
            "pass1_max_frequency"
        ]
    )

    pass2_max_frequency = (
        experiment[
            "pass2_max_frequency"
        ]
    )

    print("\n")
    print("=" * 80)
    print(
        f"EXPERIMENT: {name}"
    )
    print("=" * 80)

    print(
        f"Pass 1 max frequency: "
        f"{pass1_max_frequency:,}"
    )

    print(
        f"Pass 2 max frequency: "
        f"{pass2_max_frequency:,}"
    )

    start = time.time()

    total_candidate_pairs = 0
    max_candidates = 0
    s1_with_candidates = 0

    recovered_pairs = set()
    recovered_s1 = set()

    recovered_s2 = set()
    recovered_s3 = set()

    for counter, row in enumerate(
        s1.itertuples(index=False),
        start=1
    ):

        candidates_s2, candidates_s3 = (
            generate_for_s1(
                s1_row=row,
                exact_index=exact_index,
                token_index=token_index,
                token_frequency=token_frequency,
                s2_entity_ids=s2_entity_ids,
                s3_entity_ids=s3_entity_ids,
                pass1_max_frequency=(
                    pass1_max_frequency
                ),
                pass2_max_frequency=(
                    pass2_max_frequency
                )
            )
        )

        candidate_count = (
            len(candidates_s2)
            +
            len(candidates_s3)
        )

        total_candidate_pairs += (
            candidate_count
        )

        max_candidates = max(
            max_candidates,
            candidate_count
        )

        if candidate_count > 0:

            s1_with_candidates += 1

        s1_id = row.entity_id

        # ----------------------------------------------------
        # Recall
        # ----------------------------------------------------

        for entity_id in candidates_s2:

            pair = (
                s1_id,
                entity_id
            )

            if pair in ground_truth_pairs:

                recovered_pairs.add(
                    pair
                )

                recovered_s1.add(
                    s1_id
                )

                recovered_s2.add(
                    pair
                )

        for entity_id in candidates_s3:

            pair = (
                s1_id,
                entity_id
            )

            if pair in ground_truth_pairs:

                recovered_pairs.add(
                    pair
                )

                recovered_s1.add(
                    s1_id
                )

                recovered_s3.add(
                    pair
                )

        if counter % 5_000 == 0:

            elapsed = (
                time.time() - start
            )

            print(
                f"Processed "
                f"{counter:,}/{len(s1):,} "
                f"S1 | "
                f"Candidates: "
                f"{total_candidate_pairs:,} | "
                f"Elapsed: "
                f"{elapsed / 60:.2f} min"
            )

    # ========================================================
    # METRICS
    # ========================================================

    total_gt = len(
        ground_truth_pairs
    )

    recovered_gt = len(
        recovered_pairs
    )

    missed_gt = (
        total_gt
        -
        recovered_gt
    )

    pair_recall = (
        recovered_gt
        /
        total_gt
        *
        100
        if total_gt
        else 0
    )

    total_gt_s1 = len(
        ground_truth_s1
    )

    recovered_gt_s1 = len(
        recovered_s1
        &
        ground_truth_s1
    )

    s1_recall = (
        recovered_gt_s1
        /
        total_gt_s1
        *
        100
        if total_gt_s1
        else 0
    )

    s2_recall = (
        len(
            recovered_s2
            &
            s2_ground_truth
        )
        /
        len(s2_ground_truth)
        *
        100
        if s2_ground_truth
        else 0
    )

    s3_recall = (
        len(
            recovered_s3
            &
            s3_ground_truth
        )
        /
        len(s3_ground_truth)
        *
        100
        if s3_ground_truth
        else 0
    )

    average_candidates = (
        total_candidate_pairs
        /
        len(s1)
    )

    elapsed = (
        time.time() - start
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = {
        "experiment": name,
        "pass1_max_frequency": (
            pass1_max_frequency
        ),
        "pass2_max_frequency": (
            pass2_max_frequency
        ),
        "total_candidate_pairs": (
            total_candidate_pairs
        ),
        "avg_candidates_per_s1": (
            average_candidates
        ),
        "max_candidates_per_s1": (
            max_candidates
        ),
        "ground_truth_pairs": (
            total_gt
        ),
        "recovered_pairs": (
            recovered_gt
        ),
        "missed_pairs": (
            missed_gt
        ),
        "pair_recall_percent": (
            pair_recall
        ),
        "ground_truth_s1": (
            total_gt_s1
        ),
        "recovered_s1": (
            recovered_gt_s1
        ),
        "s1_recall_percent": (
            s1_recall
        ),
        "s2_recall_percent": (
            s2_recall
        ),
        "s3_recall_percent": (
            s3_recall
        ),
        "time_seconds": (
            elapsed
        )
    }

    print("\n" + "-" * 70)
    print("EXPERIMENT RESULT")
    print("-" * 70)

    print(
        f"Candidate pairs       : "
        f"{total_candidate_pairs:,}"
    )

    print(
        f"Average candidates/S1: "
        f"{average_candidates:,.2f}"
    )

    print(
        f"Maximum candidates/S1: "
        f"{max_candidates:,}"
    )

    print(
        f"Ground-truth pairs    : "
        f"{total_gt:,}"
    )

    print(
        f"Recovered pairs       : "
        f"{recovered_gt:,}"
    )

    print(
        f"Missed pairs          : "
        f"{missed_gt:,}"
    )

    print(
        f"Pair recall           : "
        f"{pair_recall:.4f}%"
    )

    print(
        f"S1-level recall       : "
        f"{s1_recall:.4f}%"
    )

    print(
        f"S2 recall             : "
        f"{s2_recall:.4f}%"
    )

    print(
        f"S3 recall             : "
        f"{s3_recall:.4f}%"
    )

    print(
        f"Time                  : "
        f"{elapsed / 60:.2f} minutes"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    print("=" * 80)
    print(
        "EXACT REPRODUCTION OF TRAINING BLOCKER"
    )
    print("=" * 80)

    print(
        "\nThis experiment reproduces "
        "generate_training_candidates.py"
    )

    print(
        "\nExperiments:"
    )

    for experiment in EXPERIMENTS:

        print(
            f"  {experiment['name']}: "
            f"Pass1 <= "
            f"{experiment['pass1_max_frequency']:,}, "
            f"Pass2 <= "
            f"{experiment['pass2_max_frequency']:,}"
        )

    # ========================================================
    # LOAD
    # ========================================================

    s1_raw = load_source(
        S1_PATH,
        "S1"
    )

    s2_raw = load_source(
        S2_PATH,
        "S2"
    )

    s3_raw = load_source(
        S3_PATH,
        "S3"
    )

    # Same 50K pilot
    s1_raw = (
        s1_raw
        .head(S1_LIMIT)
        .copy()
    )

    print(
        f"\nS1 rows selected: "
        f"{len(s1_raw):,}"
    )

    # ========================================================
    # PREPARE
    # ========================================================

    s1 = prepare_source(
        s1_raw,
        "S1"
    )

    s2 = prepare_source(
        s2_raw,
        "S2"
    )

    s3 = prepare_source(
        s3_raw,
        "S3"
    )

    # ========================================================
    # SOURCE IDS
    # ========================================================

    s2_entity_ids = set(
        s2["entity_id"]
    )

    s3_entity_ids = set(
        s3["entity_id"]
    )

    # ========================================================
    # COMBINED S2 + S3
    # ========================================================

    combined = pd.concat(
        [
            s2[
                [
                    "entity_id",
                    "normalized_name",
                    "tokens"
                ]
            ],
            s3[
                [
                    "entity_id",
                    "normalized_name",
                    "tokens"
                ]
            ]
        ],
        ignore_index=True
    )

    # ========================================================
    # TOKEN FREQUENCY
    # ========================================================

    print(
        "\nBuilding token frequency..."
    )

    token_frequency = (
        build_token_frequency(
            combined
        )
    )

    print(
        f"Unique tokens: "
        f"{len(token_frequency):,}"
    )

    # ========================================================
    # EXACT INDEX
    # ========================================================

    print(
        "\nBuilding exact-name index..."
    )

    exact_index = (
        build_exact_name_index(
            combined
        )
    )

    print(
        f"Unique normalized names: "
        f"{len(exact_index):,}"
    )

    # ========================================================
    # TOKEN INDEX
    # ========================================================

    print(
        "\nBuilding combined token index..."
    )

    token_index = (
        build_token_index(
            combined
        )
    )

    print(
        f"Indexed tokens: "
        f"{len(token_index):,}"
    )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    s1_ids = set(
        s1["entity_id"]
    )

    (
        ground_truth_pairs,
        ground_truth_s1,
        s2_ground_truth,
        s3_ground_truth
    ) = load_ground_truth(
        s1_ids
    )

    # ========================================================
    # EXPERIMENTS
    # ========================================================

    results = []

    for experiment in EXPERIMENTS:

        result = run_experiment(
            experiment=experiment,
            s1=s1,
            exact_index=exact_index,
            token_index=token_index,
            token_frequency=token_frequency,
            s2_entity_ids=s2_entity_ids,
            s3_entity_ids=s3_entity_ids,
            ground_truth_pairs=(
                ground_truth_pairs
            ),
            ground_truth_s1=(
                ground_truth_s1
            ),
            s2_ground_truth=(
                s2_ground_truth
            ),
            s3_ground_truth=(
                s3_ground_truth
            )
        )

        results.append(
            result
        )

    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results_path = os.path.join(
        OUTPUT_DIR,
        "exact_generator_blocking_results.tsv"
    )

    results_df.to_csv(
        results_path,
        sep="\t",
        index=False
    )

    print("\n\n")

    print("=" * 110)
    print(
        "FINAL EXACT-GENERATOR BLOCKING COMPARISON"
    )
    print("=" * 110)

    columns = [
        "experiment",
        "pass1_max_frequency",
        "pass2_max_frequency",
        "total_candidate_pairs",
        "avg_candidates_per_s1",
        "max_candidates_per_s1",
        "ground_truth_pairs",
        "recovered_pairs",
        "missed_pairs",
        "pair_recall_percent",
        "s1_recall_percent",
        "s2_recall_percent",
        "s3_recall_percent",
        "time_seconds"
    ]

    print(
        results_df[
            columns
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # BEST RECALL
    # ========================================================

    best_idx = (
        results_df[
            "pair_recall_percent"
        ].idxmax()
    )

    best = results_df.loc[
        best_idx
    ]

    print("\n" + "=" * 80)
    print(
        "HIGHEST PAIR-RECALL CONFIGURATION"
    )
    print("=" * 80)

    print(
        f"Experiment: "
        f"{best['experiment']}"
    )

    print(
        f"Pair recall: "
        f"{best['pair_recall_percent']:.4f}%"
    )

    print(
        f"S1 recall: "
        f"{best['s1_recall_percent']:.4f}%"
    )

    print(
        f"S2 recall: "
        f"{best['s2_recall_percent']:.4f}%"
    )

    print(
        f"S3 recall: "
        f"{best['s3_recall_percent']:.4f}%"
    )

    print(
        f"Average candidates/S1: "
        f"{best['avg_candidates_per_s1']:,.2f}"
    )

    print(
        f"Total candidates: "
        f"{int(best['total_candidate_pairs']):,}"
    )

    print(
        f"Maximum candidates/S1: "
        f"{int(best['max_candidates_per_s1']):,}"
    )

    # ========================================================
    # FINISH
    # ========================================================

    total_time = (
        time.time()
        - overall_start
    )

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

    print(
        f"Results saved to:\n"
        f"{results_path}"
    )

    print(
        f"\nTotal execution time: "
        f"{total_time / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()