from pathlib import Path

import pandas as pd

from blocking_v2 import (
    prepare_dataframe,
    build_exact_name_index,
    build_name_token_index,
)

from preprocessing import normalize_address


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_SIZE = 1000
RANDOM_STATE = 42


# ============================================================
# CONFIGURATIONS
# ============================================================

CONFIGS = [
    # --------------------------------------------------------
    # Existing UNION configurations
    # --------------------------------------------------------

    {
        "name": "2-token-15K-union",
        "max_frequency": 15_000,
        "max_tokens": 2,
        "use_address": True,
        "mode": "union",
    },

    {
        "name": "2-token-17.5K-union",
        "max_frequency": 17_500,
        "max_tokens": 2,
        "use_address": True,
        "mode": "union",
    },

    {
        "name": "2-token-20K-union",
        "max_frequency": 20_000,
        "max_tokens": 2,
        "use_address": True,
        "mode": "union",
    },

    # --------------------------------------------------------
    # NEW INTERSECTION configurations
    # --------------------------------------------------------

    {
        "name": "2-token-15K-intersection",
        "max_frequency": 15_000,
        "max_tokens": 2,
        "use_address": True,
        "mode": "intersection",
    },

    {
        "name": "2-token-17.5K-intersection",
        "max_frequency": 17_500,
        "max_tokens": 2,
        "use_address": True,
        "mode": "intersection",
    },

    {
        "name": "2-token-20K-intersection",
        "max_frequency": 20_000,
        "max_tokens": 2,
        "use_address": True,
        "mode": "intersection",
    },
]


# ============================================================
# TOKEN SELECTION
# ============================================================

def select_distinctive_tokens(
    tokens,
    token_frequency,
    max_frequency,
    max_tokens,
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


# ============================================================
# NAME CANDIDATES
# ============================================================

def get_name_candidates(
    s1_row,
    exact_name_index,
    token_index,
    token_frequency,
    max_frequency,
    max_tokens,
    mode,
):
    candidates = set()

    # --------------------------------------------------------
    # Exact normalized name
    # --------------------------------------------------------

    name = s1_row["name_norm"]

    if name:
        candidates.update(
            exact_name_index.get(
                name,
                set(),
            )
        )

    # --------------------------------------------------------
    # Select distinctive tokens
    # --------------------------------------------------------

    selected_tokens = select_distinctive_tokens(
        s1_row["name_tokens"],
        token_frequency,
        max_frequency,
        max_tokens,
    )

    # --------------------------------------------------------
    # No useful tokens
    # --------------------------------------------------------

    if not selected_tokens:
        return candidates

    # --------------------------------------------------------
    # UNION
    # --------------------------------------------------------

    if mode == "union":

        for token in selected_tokens:

            candidates.update(
                token_index.get(
                    token,
                    set(),
                )
            )

    # --------------------------------------------------------
    # INTERSECTION
    # --------------------------------------------------------

    elif mode == "intersection":

        token_sets = [
            token_index.get(
                token,
                set(),
            )
            for token in selected_tokens
        ]

        # Only perform intersection if we actually
        # have two useful tokens.
        if len(token_sets) >= 2:

            intersection = token_sets[0]

            for token_set in token_sets[1:]:

                intersection = (
                    intersection &
                    token_set
                )

            candidates.update(
                intersection
            )

        else:

            # If only one useful token exists,
            # retain that token's candidates.
            candidates.update(
                token_sets[0]
            )

    return candidates


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth():

    print("Loading ground truth...")

    gt = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t",
    )

    gt_map = dict(
        zip(
            gt["source1_entity_id"],
            gt["matched_entity_ids"].fillna(""),
        )
    )

    print(
        f"Ground truth rows: {len(gt_map):,}"
    )

    return gt_map


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("TOKEN INTERSECTION BLOCKING EXPERIMENT")
    print("=" * 75)

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    gt_map = load_ground_truth()

    # ========================================================
    # LOAD S1
    # ========================================================

    print("\nLoading S1...")

    s1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t",
    )

    s1_sample = s1.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    ).copy()

    print(
        f"S1 validation sample: "
        f"{len(s1_sample):,}"
    )

    # ========================================================
    # LOAD S2
    # ========================================================

    print("\nLoading S2...")

    s2 = pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t",
    )

    print(
        f"S2 rows: {len(s2):,}"
    )

    # ========================================================
    # LOAD S3
    # ========================================================

    print("\nLoading S3...")

    s3 = pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t",
    )

    print(
        f"S3 rows: {len(s3):,}"
    )

    # ========================================================
    # PREPARE
    # ========================================================

    print("\nPreparing S1...")

    s1_sample = prepare_dataframe(
        s1_sample
    )

    s1_sample["address_norm"] = (
        s1_sample["business_address"]
        .map(normalize_address)
    )

    print("Preparing S2...")

    s2 = prepare_dataframe(
        s2
    )

    s2["address_norm"] = (
        s2["business_address"]
        .map(normalize_address)
    )

    print("Preparing S3...")

    s3 = prepare_dataframe(
        s3
    )

    s3["address_norm"] = (
        s3["business_address"]
        .map(normalize_address)
    )

    # ========================================================
    # EXACT NAME INDEXES
    # ========================================================

    print(
        "\nBuilding S2 exact-name index..."
    )

    s2_exact_name_index = (
        build_exact_name_index(s2)
    )

    print(
        "Building S3 exact-name index..."
    )

    s3_exact_name_index = (
        build_exact_name_index(s3)
    )

    # ========================================================
    # TOKEN INDEXES
    # ========================================================

    print(
        "Building S2 name-token index..."
    )

    s2_token_index = (
        build_name_token_index(s2)
    )

    print(
        "Building S3 name-token index..."
    )

    s3_token_index = (
        build_name_token_index(s3)
    )

    # ========================================================
    # TOKEN FREQUENCIES
    # ========================================================

    print(
        "Calculating token frequencies..."
    )

    s2_token_frequency = {
        token: len(entity_ids)
        for token, entity_ids
        in s2_token_index.items()
    }

    s3_token_frequency = {
        token: len(entity_ids)
        for token, entity_ids
        in s3_token_index.items()
    }

    # ========================================================
    # ADDRESS INDEXES
    # ========================================================

    print(
        "\nBuilding S2 exact-address index..."
    )

    s2_address_index = {}

    for address, group in s2.groupby(
        "address_norm"
    ):

        if not address:
            continue

        s2_address_index[address] = set(
            group["entity_id"]
        )

    print(
        "Building S3 exact-address index..."
    )

    s3_address_index = {}

    for address, group in s3.groupby(
        "address_norm"
    ):

        if not address:
            continue

        s3_address_index[address] = set(
            group["entity_id"]
        )

    print(
        "\nAll indexes ready."
    )

    # ========================================================
    # RUN EXPERIMENTS
    # ========================================================

    results = []

    for config in CONFIGS:

        config_name = config["name"]
        max_frequency = config["max_frequency"]
        max_tokens = config["max_tokens"]
        use_address = config["use_address"]
        mode = config["mode"]

        print("\n" + "=" * 75)

        print(
            f"TESTING: {config_name}"
        )

        print(
            f"Maximum token frequency: "
            f"{max_frequency:,}"
        )

        print(
            f"Maximum tokens: "
            f"{max_tokens}"
        )

        print(
            f"Mode: {mode}"
        )

        print(
            f"Exact address: "
            f"{use_address}"
        )

        print("=" * 75)

        total_candidates = 0
        max_candidates = 0

        s1_with_candidates = 0
        s1_with_missed_matches = 0

        total_true_matches = 0
        retrieved_true_matches = 0

        # ----------------------------------------------------
        # Validation loop
        # ----------------------------------------------------

        for _, row in s1_sample.iterrows():

            s1_id = row["entity_id"]

            # =================================================
            # S2 NAME CANDIDATES
            # =================================================

            s2_candidates = get_name_candidates(
                row,
                s2_exact_name_index,
                s2_token_index,
                s2_token_frequency,
                max_frequency,
                max_tokens,
                mode,
            )

            # =================================================
            # S3 NAME CANDIDATES
            # =================================================

            s3_candidates = get_name_candidates(
                row,
                s3_exact_name_index,
                s3_token_index,
                s3_token_frequency,
                max_frequency,
                max_tokens,
                mode,
            )

            # =================================================
            # EXACT ADDRESS
            #
            # Address candidates are added independently.
            # =================================================

            if use_address:

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

            # =================================================
            # COMBINE S2 + S3
            # =================================================

            candidates = (
                s2_candidates |
                s3_candidates
            )

            candidate_count = len(
                candidates
            )

            total_candidates += (
                candidate_count
            )

            max_candidates = max(
                max_candidates,
                candidate_count,
            )

            if candidate_count > 0:
                s1_with_candidates += 1

            # =================================================
            # GROUND TRUTH
            # =================================================

            true_string = gt_map.get(
                s1_id,
                "",
            )

            if true_string:

                true_ids = {
                    x.strip()
                    for x in true_string.split(",")
                    if x.strip()
                }

            else:

                true_ids = set()

            total_true_matches += (
                len(true_ids)
            )

            # =================================================
            # RETRIEVED TRUE MATCHES
            # =================================================

            retrieved = (
                true_ids &
                candidates
            )

            retrieved_true_matches += (
                len(retrieved)
            )

            if len(retrieved) < len(true_ids):

                s1_with_missed_matches += 1

        # ====================================================
        # METRICS
        # ====================================================

        recall = (
            retrieved_true_matches /
            total_true_matches
            if total_true_matches > 0
            else 0
        )

        mean_candidates = (
            total_candidates /
            len(s1_sample)
        )

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print("\nRESULTS")
        print("-" * 75)

        print(
            f"S1 validation rows:        "
            f"{len(s1_sample):,}"
        )

        print(
            f"S1 with candidates:        "
            f"{s1_with_candidates:,}"
        )

        print(
            f"S1 rows with missed match: "
            f"{s1_with_missed_matches:,}"
        )

        print(
            f"Total true matches:         "
            f"{total_true_matches:,}"
        )

        print(
            f"Retrieved true matches:     "
            f"{retrieved_true_matches:,}"
        )

        print(
            f"Missed true matches:        "
            f"{total_true_matches - retrieved_true_matches:,}"
        )

        print(
            f"Blocking recall:            "
            f"{recall:.4%}"
        )

        print(
            f"Mean candidates / S1:       "
            f"{mean_candidates:,.2f}"
        )

        print(
            f"Maximum candidates / S1:    "
            f"{max_candidates:,}"
        )

        results.append(
            {
                "configuration": config_name,
                "recall": recall,
                "mean_candidates": mean_candidates,
                "max_candidates": max_candidates,
            }
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n")

    print("=" * 75)
    print("FINAL COMPARISON")
    print("=" * 75)

    result_df = pd.DataFrame(
        results
    )

    print(
        result_df.to_string(
            index=False
        )
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()