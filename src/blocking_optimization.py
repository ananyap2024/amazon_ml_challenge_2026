import time
from pathlib import Path
from collections import defaultdict

import pandas as pd

from preprocessing import normalize_business_name, tokenize


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "dataset" / "train"

S1_FILE = TRAIN_DIR / "train_source1.tsv"
S2_FILE = TRAIN_DIR / "train_source2.tsv"
S3_FILE = TRAIN_DIR / "train_source3.tsv"
GT_FILE = TRAIN_DIR / "train_ground_truth.tsv"

# Keep this at 10K for the experiment
S1_LIMIT = 10_000


# ============================================================
# MULTI-PASS CONFIGURATIONS
# ============================================================
#
# Each configuration consists of multiple blocking passes.
#
# Every pass produces candidates.
# The final candidate set is the UNION of all passes.
#
# Examples:
#
# exact
# 1-token-20K
# 2-token-10K
#
# ============================================================

CONFIGURATIONS = [

    {
        "name": "1token_15K_plus_2token_7500",
        "passes": [
            ("token", 1, 15_000),
            ("token", 2, 7_500),
        ],
    },

    {
        "name": "1token_15K_plus_2token_10K",
        "passes": [
            ("token", 1, 15_000),
            ("token", 2, 10_000),
        ],
    },

    {
        "name": "1token_17500_plus_2token_7500",
        "passes": [
            ("token", 1, 17_500),
            ("token", 2, 7_500),
        ],
    },

    {
        "name": "1token_17500_plus_2token_10K",
        "passes": [
            ("token", 1, 17_500),
            ("token", 2, 10_000),
        ],
    },

    {
        "name": "1token_20K_plus_2token_7500",
        "passes": [
            ("token", 1, 20_000),
            ("token", 2, 7_500),
        ],
    },

    {
        "name": "1token_20K_plus_2token_10K",
        "passes": [
            ("token", 1, 20_000),
            ("token", 2, 10_000),
        ],
    },

]


# ============================================================
# GENERIC TOKENS
# ============================================================

GENERIC_TOKENS = {
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
# TOKENIZATION
# ============================================================

def tokenize_name(name):

    tokens = tokenize(name)

    result = []

    for token in tokens:

        if token in GENERIC_TOKENS:
            continue

        if len(token) < 3:
            continue

        result.append(token)

    return list(set(result))


# ============================================================
# PREPARE DATAFRAME
# ============================================================

def prepare_dataframe(df):

    df = df.copy()

    df["name_norm"] = df["business_name"].map(
        normalize_business_name
    )

    df["name_tokens"] = df["name_norm"].map(
        tokenize_name
    )

    return df


# ============================================================
# EXACT NAME INDEX
# ============================================================

def build_exact_name_index(df):

    print("Building exact-name index...")

    index = defaultdict(set)

    for entity_id, name in zip(
        df["entity_id"],
        df["name_norm"],
    ):

        if not name:
            continue

        index[name].add(entity_id)

    return dict(index)


# ============================================================
# TOKEN INDEX
# ============================================================

def build_token_index(df):

    print("Building token index...")

    index = defaultdict(set)

    for entity_id, tokens in zip(
        df["entity_id"],
        df["name_tokens"],
    ):

        for token in tokens:
            index[token].add(entity_id)

    return dict(index)


# ============================================================
# TOKEN FREQUENCIES
# ============================================================

def calculate_frequencies(index):

    return {
        token: len(entity_ids)
        for token, entity_ids in index.items()
    }


# ============================================================
# SELECT DISTINCTIVE TOKENS
# ============================================================

def select_tokens(
    tokens,
    frequency,
    max_tokens,
    max_frequency,
):

    valid = []

    for token in tokens:

        freq = frequency.get(token)

        if freq is None:
            continue

        if freq <= max_frequency:

            valid.append(
                (freq, token)
            )

    # Rarest tokens first
    valid.sort(
        key=lambda x: x[0]
    )

    return [
        token
        for _, token in valid[:max_tokens]
    ]


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth():

    print()
    print("=" * 70)
    print("LOADING GROUND TRUTH")
    print("=" * 70)

    gt = pd.read_csv(
        GT_FILE,
        sep="\t",
        dtype=str,
    )

    print(
        f"Ground truth rows: {len(gt):,}"
    )

    truth = {}

    for _, row in gt.iterrows():

        source1_id = str(
            row["source1_entity_id"]
        )

        matched = row["matched_entity_ids"]

        if pd.isna(matched):

            truth[source1_id] = set()

            continue

        matched_string = str(
            matched
        ).strip()

        if not matched_string:

            truth[source1_id] = set()

            continue

        truth[source1_id] = {
            x.strip()
            for x in matched_string.split(",")
            if x.strip()
        }

    return truth


# ============================================================
# LOAD TRAINING DATA
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)

    print("Loading S1...")

    if S1_LIMIT is None:

        s1 = pd.read_csv(
            S1_FILE,
            sep="\t",
            dtype=str,
        )

    else:

        s1 = pd.read_csv(
            S1_FILE,
            sep="\t",
            dtype=str,
            nrows=S1_LIMIT,
        )

    print(
        f"S1 rows: {len(s1):,}"
    )

    print("Loading S2...")

    s2 = pd.read_csv(
        S2_FILE,
        sep="\t",
        dtype=str,
    )

    print(
        f"S2 rows: {len(s2):,}"
    )

    print("Loading S3...")

    s3 = pd.read_csv(
        S3_FILE,
        sep="\t",
        dtype=str,
    )

    print(
        f"S3 rows: {len(s3):,}"
    )

    return s1, s2, s3


# ============================================================
# BUILD INDEXES
# ============================================================

def build_indexes(s2, s3):

    print()
    print("=" * 70)
    print("PREPARING SOURCE DATA")
    print("=" * 70)

    print("Preparing S2...")
    s2 = prepare_dataframe(s2)

    print("Preparing S3...")
    s3 = prepare_dataframe(s3)

    print()
    print("=" * 70)
    print("BUILDING INDEXES")
    print("=" * 70)

    print()
    print("SOURCE 2")

    s2_name_index = build_exact_name_index(s2)
    s2_token_index = build_token_index(s2)
    s2_frequency = calculate_frequencies(
        s2_token_index
    )

    print()
    print("SOURCE 3")

    s3_name_index = build_exact_name_index(s3)
    s3_token_index = build_token_index(s3)
    s3_frequency = calculate_frequencies(
        s3_token_index
    )

    print()
    print("All indexes ready.")

    return (
        s2_name_index,
        s3_name_index,
        s2_token_index,
        s3_token_index,
        s2_frequency,
        s3_frequency,
    )


# ============================================================
# APPLY ONE BLOCKING PASS
# ============================================================

def apply_pass(
    row,
    pass_type,
    max_tokens,
    max_frequency,
    s2_name_index,
    s3_name_index,
    s2_token_index,
    s3_token_index,
    s2_frequency,
    s3_frequency,
):

    s2_candidates = set()
    s3_candidates = set()

    # --------------------------------------------------------
    # EXACT NAME PASS
    # --------------------------------------------------------

    if pass_type == "exact":

        name = row["name_norm"]

        if name:

            s2_candidates.update(
                s2_name_index.get(
                    name,
                    ()
                )
            )

            s3_candidates.update(
                s3_name_index.get(
                    name,
                    ()
                )
            )

        return (
            s2_candidates,
            s3_candidates,
        )

    # --------------------------------------------------------
    # TOKEN PASS
    # --------------------------------------------------------

    if pass_type == "token":

        s2_tokens = select_tokens(
            row["name_tokens"],
            s2_frequency,
            max_tokens,
            max_frequency,
        )

        s3_tokens = select_tokens(
            row["name_tokens"],
            s3_frequency,
            max_tokens,
            max_frequency,
        )

        for token in s2_tokens:

            s2_candidates.update(
                s2_token_index.get(
                    token,
                    ()
                )
            )

        for token in s3_tokens:

            s3_candidates.update(
                s3_token_index.get(
                    token,
                    ()
                )
            )

        return (
            s2_candidates,
            s3_candidates,
        )

    return (
        s2_candidates,
        s3_candidates,
    )


# ============================================================
# GENERATE MULTI-PASS CANDIDATES
# ============================================================

def generate_candidates(
    row,
    config,
    indexes,
):

    (
        s2_name_index,
        s3_name_index,
        s2_token_index,
        s3_token_index,
        s2_frequency,
        s3_frequency,
    ) = indexes

    final_s2_candidates = set()
    final_s3_candidates = set()

    # --------------------------------------------------------
    # Run every blocking pass
    # --------------------------------------------------------

    for (
        pass_type,
        max_tokens,
        max_frequency,
    ) in config["passes"]:

        (
            s2_candidates,
            s3_candidates,
        ) = apply_pass(
            row,
            pass_type,
            max_tokens,
            max_frequency,
            s2_name_index,
            s3_name_index,
            s2_token_index,
            s3_token_index,
            s2_frequency,
            s3_frequency,
        )

        # UNION across passes

        final_s2_candidates.update(
            s2_candidates
        )

        final_s3_candidates.update(
            s3_candidates
        )

    return (
        final_s2_candidates,
        final_s3_candidates,
    )


# ============================================================
# EVALUATE CONFIGURATION
# ============================================================

def evaluate_configuration(
    config,
    s1,
    truth,
    indexes,
):

    print()
    print("=" * 70)
    print(
        f"TESTING: {config['name']}"
    )
    print("=" * 70)

    print(
        "Passes:"
    )

    for blocking_pass in config["passes"]:

        print(
            f"  {blocking_pass}"
        )

    start_time = time.time()

    total_candidates = 0
    max_candidates = 0

    s1_with_candidates = 0

    total_true_matches = 0
    true_matches_found = 0

    rows_processed = 0

    # --------------------------------------------------------
    # PROCESS S1
    # --------------------------------------------------------

    for _, row in s1.iterrows():

        source1_id = str(
            row["entity_id"]
        )

        (
            s2_candidates,
            s3_candidates,
        ) = generate_candidates(
            row,
            config,
            indexes,
        )

        candidates = (
            s2_candidates
            | s3_candidates
        )

        candidate_count = len(
            candidates
        )

        total_candidates += candidate_count

        if candidate_count > max_candidates:

            max_candidates = candidate_count

        if candidate_count > 0:

            s1_with_candidates += 1

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        true_ids = truth.get(
            source1_id,
            set(),
        )

        if true_ids:

            total_true_matches += len(
                true_ids
            )

            found = (
                true_ids
                & candidates
            )

            true_matches_found += len(
                found
            )

        rows_processed += 1

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )

    average_candidates = (
        total_candidates / rows_processed
        if rows_processed
        else 0
    )

    pair_recall = (
        true_matches_found
        / total_true_matches
        if total_true_matches
        else 0
    )

    rows_with_candidates_percent = (
        s1_with_candidates
        / rows_processed
        * 100
        if rows_processed
        else 0
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()

    print(
        f"Rows processed        : "
        f"{rows_processed:,}"
    )

    print(
        f"S1 with candidates   : "
        f"{s1_with_candidates:,}"
    )

    print(
        f"Total candidates      : "
        f"{total_candidates:,}"
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
        f"True matches          : "
        f"{total_true_matches:,}"
    )

    print(
        f"True matches found    : "
        f"{true_matches_found:,}"
    )

    print(
        f"Pair recall           : "
        f"{pair_recall * 100:.4f}%"
    )

    print(
        f"Rows with candidates  : "
        f"{rows_with_candidates_percent:.4f}%"
    )

    print(
        f"Runtime               : "
        f"{elapsed / 60:.2f} minutes"
    )

    return {
        "strategy": config["name"],
        "average_candidates": average_candidates,
        "max_candidates": max_candidates,
        "total_candidates": total_candidates,
        "true_matches": total_true_matches,
        "true_matches_found": true_matches_found,
        "recall_percent": pair_recall * 100,
        "rows_with_candidates_percent":
            rows_with_candidates_percent,
        "runtime_minutes":
            elapsed / 60,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = time.time()

    print("=" * 70)
    print("MULTI-PASS BLOCKING OPTIMIZATION")
    print("=" * 70)

    print()

    print(
        "S1 limit: "
        f"{S1_LIMIT if S1_LIMIT is not None else 'ALL'}"
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    truth = load_ground_truth()

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    s1, s2, s3 = load_data()

    # --------------------------------------------------------
    # Prepare S1
    # --------------------------------------------------------

    print()
    print("Preparing S1...")

    s1 = prepare_dataframe(
        s1
    )

    # --------------------------------------------------------
    # Build indexes
    # --------------------------------------------------------

    indexes = build_indexes(
        s2,
        s3,
    )

    # --------------------------------------------------------
    # Run configurations
    # --------------------------------------------------------

    results = []

    for config in CONFIGURATIONS:

        result = evaluate_configuration(
            config,
            s1,
            truth,
            indexes,
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Results dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="recall_percent",
        ascending=False,
    )

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print()

    print(
        results_df[
            [
                "strategy",
                "average_candidates",
                "max_candidates",
                "recall_percent",
                "total_candidates",
                "runtime_minutes",
            ]
        ].to_string(
            index=False,
            formatters={
                "average_candidates":
                    "{:,.2f}".format,

                "max_candidates":
                    "{:,.0f}".format,

                "recall_percent":
                    "{:.4f}%".format,

                "total_candidates":
                    "{:,.0f}".format,

                "runtime_minutes":
                    "{:.2f}".format,
            },
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        PROJECT_DIR
        / "output"
        / "blocking_optimization_results.tsv"
    )

    results_df.to_csv(
        output_file,
        sep="\t",
        index=False,
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        output_file
    )

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------

    total_time = (
        time.time()
        - total_start
    )

    print()
    print("=" * 70)
    print(
        f"TOTAL RUNTIME: "
        f"{total_time / 60:.2f} minutes"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()