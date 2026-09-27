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
    "output"
)

# ------------------------------------------------------------
# TRAINING PILOT
# ------------------------------------------------------------

S1_LIMIT = 50_000

# ------------------------------------------------------------
# FROZEN BLOCKING STRATEGY
#
# Pass 1:
#   1 distinctive token
#   max frequency = 20,000
#
# Pass 2:
#   2 distinctive tokens
#   max frequency = 7,500
#
# Exact normalized name is also included.
# ------------------------------------------------------------

PASS1_MAX_TOKENS = 1
PASS1_MAX_TOKEN_FREQUENCY = 20_000

PASS2_MAX_TOKENS = 2
PASS2_MAX_TOKEN_FREQUENCY = 12_500

S1_CHUNK_SIZE = 5_000

# ------------------------------------------------------------
# INPUT FILES
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "training_candidate_pairs_internal.tsv"
)


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
# LOAD DATA
# ============================================================

def load_source(path, source_name):

    print(f"\nLoading {source_name}...")
    print(f"Path: {path}")

    df = pd.read_csv(
        path,
        sep="\t"
    )

    print(
        f"{source_name} rows: "
        f"{len(df):,}"
    )

    print(
        f"{source_name} columns: "
        f"{list(df.columns)}"
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
# BUILD EXACT NAME INDEX
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
# BUILD TOKEN INDEX
# ============================================================

def build_token_index(
    source_df,
    token_frequency,
    max_frequency
):

    token_index = defaultdict(set)

    for row in source_df.itertuples(
        index=False
    ):

        for token in set(row.tokens):

            frequency = token_frequency.get(
                token,
                0
            )

            if frequency <= max_frequency:

                token_index[token].add(
                    row.entity_id
                )

    return token_index


# ============================================================
# GENERATE CANDIDATES FOR ONE S1 ENTITY
# ============================================================

def generate_for_s1(
    s1_row,
    exact_index,
    token_index_pass1,
    token_index_pass2,
    token_frequency,
    s2_entity_ids,
    s3_entity_ids
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

        exact_matches = exact_index.get(
            normalized_name,
            set()
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
    # ========================================================
    # 1 distinctive token
    # frequency <= 20,000
    # ========================================================

    pass1_tokens = select_tokens(
        tokens=tokens,
        token_frequency=token_frequency,
        max_tokens=PASS1_MAX_TOKENS,
        max_frequency=PASS1_MAX_TOKEN_FREQUENCY
    )

    for token in pass1_tokens:

        matching_ids = (
            token_index_pass1.get(
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
    # ========================================================
    # 2 distinctive tokens
    # frequency <= 7,500
    # ========================================================

    pass2_tokens = select_tokens(
        tokens=tokens,
        token_frequency=token_frequency,
        max_tokens=PASS2_MAX_TOKENS,
        max_frequency=PASS2_MAX_TOKEN_FREQUENCY
    )

    for token in pass2_tokens:

        matching_ids = (
            token_index_pass2.get(
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
# MAIN
# ============================================================

def main():

    start_time = time.time()

    print("=" * 70)
    print("TRAINING CANDIDATE GENERATION")
    print("=" * 70)

    print("\nConfiguration:")

    print(
        f"  S1 limit                 : "
        f"{S1_LIMIT:,}"
    )

    print(
        f"  Pass 1 max tokens        : "
        f"{PASS1_MAX_TOKENS}"
    )

    print(
        f"  Pass 1 max frequency     : "
        f"{PASS1_MAX_TOKEN_FREQUENCY:,}"
    )

    print(
        f"  Pass 2 max tokens        : "
        f"{PASS2_MAX_TOKENS}"
    )

    print(
        f"  Pass 2 max frequency     : "
        f"{PASS2_MAX_TOKEN_FREQUENCY:,}"
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    if os.path.exists(
        OUTPUT_PATH
    ):

        print(
            f"\nRemoving previous output:"
            f"\n{OUTPUT_PATH}"
        )

        os.remove(
            OUTPUT_PATH
        )

    # ========================================================
    # LOAD SOURCES
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

    # ========================================================
    # LIMIT S1
    # ========================================================

    s1_raw = (
        s1_raw
        .head(S1_LIMIT)
        .copy()
    )

    print(
        f"\nS1 rows selected for pilot: "
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
    # ENTITY IDS
    # ========================================================

    s2_entity_ids = set(
        s2["entity_id"]
    )

    s3_entity_ids = set(
        s3["entity_id"]
    )

    # ========================================================
    # COMBINED CANDIDATE SOURCES
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
    # PASS 1 INDEX
    # ========================================================

    print(
        "\nBuilding Pass 1 token index..."
    )

    token_index_pass1 = (
        build_token_index(
            combined,
            token_frequency,
            PASS1_MAX_TOKEN_FREQUENCY
        )
    )

    print(
        f"Pass 1 indexed tokens: "
        f"{len(token_index_pass1):,}"
    )

    # ========================================================
    # PASS 2 INDEX
    # ========================================================

    print(
        "\nBuilding Pass 2 token index..."
    )

    token_index_pass2 = (
        build_token_index(
            combined,
            token_frequency,
            PASS2_MAX_TOKEN_FREQUENCY
        )
    )

    print(
        f"Pass 2 indexed tokens: "
        f"{len(token_index_pass2):,}"
    )

    # ========================================================
    # GENERATION
    # ========================================================

    print(
        "\nGenerating training candidate pairs..."
    )

    first_write = True

    total_pairs = 0
    s1_with_candidates = 0
    max_candidates = 0

    generation_start = time.time()

    # ========================================================
    # PROCESS S1
    # ========================================================

    for chunk_start in range(
        0,
        len(s1),
        S1_CHUNK_SIZE
    ):

        chunk_end = min(
            chunk_start + S1_CHUNK_SIZE,
            len(s1)
        )

        chunk = s1.iloc[
            chunk_start:chunk_end
        ]

        output_rows = []

        for row in chunk.itertuples(
            index=False
        ):

            candidates_s2, candidates_s3 = (
                generate_for_s1(
                    s1_row=row,
                    exact_index=exact_index,
                    token_index_pass1=token_index_pass1,
                    token_index_pass2=token_index_pass2,
                    token_frequency=token_frequency,
                    s2_entity_ids=s2_entity_ids,
                    s3_entity_ids=s3_entity_ids
                )
            )

            candidate_count = (
                len(candidates_s2)
                +
                len(candidates_s3)
            )

            total_pairs += (
                candidate_count
            )

            if candidate_count > 0:

                s1_with_candidates += 1

            max_candidates = max(
                max_candidates,
                candidate_count
            )

            # ------------------------------------------------
            # SOURCE 2
            # ------------------------------------------------

            for entity_id in candidates_s2:

                output_rows.append(
                    (
                        row.entity_id,
                        entity_id,
                        "source2"
                    )
                )

            # ------------------------------------------------
            # SOURCE 3
            # ------------------------------------------------

            for entity_id in candidates_s3:

                output_rows.append(
                    (
                        row.entity_id,
                        entity_id,
                        "source3"
                    )
                )

        # ====================================================
        # WRITE CHUNK
        # ====================================================

        if output_rows:

            output_df = pd.DataFrame(
                output_rows,
                columns=[
                    "source1_entity_id",
                    "candidate_entity_id",
                    "source"
                ]
            )

            output_df.to_csv(
                OUTPUT_PATH,
                sep="\t",
                index=False,
                mode="w" if first_write else "a",
                header=first_write
            )

            first_write = False

        elapsed = (
            time.time()
            - generation_start
        )

        rate = (
            chunk_end / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"Processed "
            f"{chunk_end:,}/{len(s1):,} "
            f"S1 rows | "
            f"Pairs: {total_pairs:,} | "
            f"Rate: {rate:.2f} S1/sec"
        )

    # ========================================================
    # FINAL STATISTICS
    # ========================================================

    generation_time = (
        time.time()
        - generation_start
    )

    total_time = (
        time.time()
        - start_time
    )

    average_candidates = (
        total_pairs / len(s1)
        if len(s1) > 0
        else 0
    )

    output_size_gb = (
        os.path.getsize(
            OUTPUT_PATH
        )
        / (1024 ** 3)
        if os.path.exists(
            OUTPUT_PATH
        )
        else 0
    )

    print("\n")
    print("=" * 70)
    print("TRAINING CANDIDATE GENERATION RESULTS")
    print("=" * 70)

    print(
        f"S1 rows processed      : "
        f"{len(s1):,}"
    )

    print(
        f"S1 with candidates     : "
        f"{s1_with_candidates:,}"
    )

    print(
        f"Total candidate pairs  : "
        f"{total_pairs:,}"
    )

    print(
        f"Average candidates/S1  : "
        f"{average_candidates:,.2f}"
    )

    print(
        f"Maximum candidates/S1  : "
        f"{max_candidates:,}"
    )

    print(
        f"Output size            : "
        f"{output_size_gb:.2f} GB"
    )

    print(
        f"Generation time        : "
        f"{generation_time / 60:.2f} minutes"
    )

    print(
        f"Total execution time   : "
        f"{total_time / 60:.2f} minutes"
    )

    print(
        f"Processing rate        : "
        f"{len(s1) / generation_time:.2f} S1/sec"
    )

    print("\nOutput:")
    print(OUTPUT_PATH)

    print("=" * 70)


if __name__ == "__main__":
    main()