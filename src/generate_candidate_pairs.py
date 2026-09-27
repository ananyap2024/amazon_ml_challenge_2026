import os
import re
import time
from collections import defaultdict

import pandas as pd


# ============================================================
# CONFIGURATION
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


# ============================================================
# TEST DATA PATHS
# ============================================================

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


# ============================================================
# S1 LIMIT
# ============================================================
# None = FULL TEST DATASET
#
# For a smoke test, temporarily use:
# S1_LIMIT = 50_000
#
# For official generation:
# S1_LIMIT = None
# ============================================================

S1_LIMIT = 50_000


# ============================================================
# MULTI-PASS BLOCKING STRATEGY
# ============================================================
#
# Pass 1:
#   1 distinctive token
#   max token frequency = 20,000
#
# Pass 2:
#   2 distinctive tokens
#   max token frequency = 12,500
#
# Exact normalized-name blocking is also included.
#
# Candidates from all passes are UNIONED.
# ============================================================

PASS1_MAX_TOKENS = 1

PASS1_MAX_TOKEN_FREQUENCY = 20_000


PASS2_MAX_TOKENS = 2

PASS2_MAX_TOKEN_FREQUENCY = 12_500


# ============================================================
# CHUNK SIZE
# ============================================================

S1_CHUNK_SIZE = 10_000


# ============================================================
# OUTPUT PATHS
# ============================================================

INTERNAL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs_internal.tsv"
)

OFFICIAL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize text for blocking.

    Steps:
    1. Handle missing values.
    2. Convert to lowercase.
    3. Remove non-alphanumeric characters.
    4. Collapse repeated whitespace.
    """

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


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(value):
    """
    Convert normalized name into tokens.
    """

    if not value:
        return []

    return value.split()


# ============================================================
# LOAD DATA
# ============================================================

def load_source(path, source_name):

    print(
        f"\nLoading {source_name}..."
    )

    print(
        f"Path: {path}"
    )

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
# IDENTIFY COLUMNS
# ============================================================

def find_column(df, candidates):
    """
    Find a column using case-insensitive matching.
    """

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in normalized:
            return normalized[key]

    # Partial matching fallback

    for col in df.columns:

        col_lower = str(col).lower()

        for candidate in candidates:

            if candidate.lower() in col_lower:
                return col

    return None


# ============================================================
# PREPARE SOURCE
# ============================================================

def prepare_source(df, source_name):

    id_col = find_column(
        df,
        [
            "entity_id",
            "id",
            f"{source_name}_entity_id"
        ]
    )

    name_col = find_column(
        df,
        [
            "name",
            "business_name",
            "company_name",
            "entity_name"
        ]
    )

    if id_col is None:

        raise ValueError(
            f"Could not identify ID column "
            f"in {source_name}"
        )

    if name_col is None:

        raise ValueError(
            f"Could not identify name column "
            f"in {source_name}"
        )

    result = pd.DataFrame()

    result["entity_id"] = (
        df[id_col].astype(str)
    )

    result["name"] = (
        df[name_col]
        .fillna("")
        .astype(str)
    )

    result["normalized_name"] = (
        result["name"]
        .map(normalize_text)
    )

    result["tokens"] = (
        result["normalized_name"]
        .map(tokenize)
    )

    print(
        f"{source_name}: "
        f"ID column = {id_col}, "
        f"name column = {name_col}"
    )

    return result


# ============================================================
# TOKEN FREQUENCY
# ============================================================

def build_token_frequency(source_df):

    token_frequency = defaultdict(int)

    for tokens in source_df["tokens"]:

        unique_tokens = set(tokens)

        for token in unique_tokens:

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
    """
    Select up to max_tokens distinctive tokens.

    Lower-frequency tokens are preferred because
    they generate smaller candidate blocks.
    """

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

        normalized_name = (
            row.normalized_name
        )

        if normalized_name:

            index[
                normalized_name
            ].add(
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
    """
    Build an inverted index:

        token -> set(entity_ids)

    Only tokens whose frequency is <=
    max_frequency are indexed.
    """

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
# GENERATE CANDIDATES FOR ONE S1 ROW
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
    """
    Generate candidates using:

    1. Exact normalized-name blocking

    2. Pass 1:
       1 distinctive token
       max frequency = 20,000

    3. Pass 2:
       2 distinctive tokens
       max frequency = 12,500

    Candidates from all passes are UNIONED.
    """

    candidates_s2 = set()

    candidates_s3 = set()

    normalized_name = (
        s1_row.normalized_name
    )

    tokens = s1_row.tokens


    # ========================================================
    # EXACT NAME PASS
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
    # max frequency = 20,000
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
    # max frequency = 12,500
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

    print(
        "OFFICIAL TEST CANDIDATE GENERATION"
    )

    print("=" * 70)


    # ========================================================
    # CONFIGURATION DISPLAY
    # ========================================================

    print("\nConfiguration:")

    if S1_LIMIT is None:

        s1_limit_display = "ALL"

    else:

        s1_limit_display = (
            f"{S1_LIMIT:,}"
        )

    print(
        f"  S1 limit                 : "
        f"{s1_limit_display}"
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

    print(
        f"  S1 chunk size            : "
        f"{S1_CHUNK_SIZE:,}"
    )


    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    # ========================================================
    # REMOVE PREVIOUS OUTPUTS
    # ========================================================

    for path in [
        INTERNAL_OUTPUT,
        OFFICIAL_OUTPUT
    ]:

        if os.path.exists(path):

            print(
                f"\nRemoving previous output: "
                f"{path}"
            )

            os.remove(path)


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
    # APPLY S1 LIMIT
    # ========================================================

    if S1_LIMIT is not None:

        s1_raw = (
            s1_raw
            .head(S1_LIMIT)
            .copy()
        )

        print(
            f"\nS1 rows selected for pilot: "
            f"{len(s1_raw):,}"
        )

    else:

        s1_raw = s1_raw.copy()

        print(
            f"\nFull S1 dataset selected: "
            f"{len(s1_raw):,} rows"
        )


    # ========================================================
    # PREPARE SOURCES
    # ========================================================

    print("\nPreparing sources...")

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
    # ID SETS
    # ========================================================

    print("\nBuilding ID sets...")

    s2_entity_ids = set(
        s2["entity_id"]
    )

    s3_entity_ids = set(
        s3["entity_id"]
    )

    print(
        f"S2 unique IDs: "
        f"{len(s2_entity_ids):,}"
    )

    print(
        f"S3 unique IDs: "
        f"{len(s3_entity_ids):,}"
    )


    # ========================================================
    # COMBINED SOURCE
    # ========================================================

    print(
        "\nCombining S2 + S3..."
    )

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
        "\nBuilding token frequency index..."
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
    # EXACT NAME INDEX
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
    # TOKEN INDEX — PASS 1
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
    # TOKEN INDEX — PASS 2
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
        "\nGenerating candidate pairs..."
    )

    internal_first_write = True

    official_first_write = True

    total_pairs = 0

    s1_with_candidates = 0

    max_candidates = 0

    generation_start = time.time()


    # ========================================================
    # PROCESS S1 IN CHUNKS
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

        internal_rows = []

        official_rows = []


        # ====================================================
        # PROCESS EACH S1 ROW
        # ====================================================

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


            # =================================================
            # UNION S2 + S3 CANDIDATES
            # =================================================

            all_candidates = (
                candidates_s2 |
                candidates_s3
            )


            candidate_count = len(
                all_candidates
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


            # =================================================
            # INTERNAL FORMAT
            # =================================================

            for entity_id in sorted(
                candidates_s2
            ):

                internal_rows.append(
                    (
                        row.entity_id,
                        entity_id,
                        "source2"
                    )
                )


            for entity_id in sorted(
                candidates_s3
            ):

                internal_rows.append(
                    (
                        row.entity_id,
                        entity_id,
                        "source3"
                    )
                )


            # =================================================
            # OFFICIAL FORMAT
            # =================================================

            if all_candidates:

                ordered_candidates = sorted(
                    all_candidates
                )

                official_rows.append(
                    (
                        row.entity_id,
                        ",".join(
                            ordered_candidates
                        )
                    )
                )


        # ====================================================
        # WRITE INTERNAL OUTPUT
        # ====================================================

        if internal_rows:

            internal_df = pd.DataFrame(
                internal_rows,
                columns=[
                    "source1_entity_id",
                    "candidate_entity_id",
                    "source"
                ]
            )

            internal_df.to_csv(
                INTERNAL_OUTPUT,
                sep="\t",
                index=False,
                mode=(
                    "w"
                    if internal_first_write
                    else "a"
                ),
                header=internal_first_write
            )

            internal_first_write = False


        # ====================================================
        # WRITE OFFICIAL OUTPUT
        # ====================================================

        if official_rows:

            official_df = pd.DataFrame(
                official_rows,
                columns=[
                    "source1_entity_id",
                    "candidate_entity_ids"
                ]
            )

            official_df.to_csv(
                OFFICIAL_OUTPUT,
                sep="\t",
                index=False,
                mode=(
                    "w"
                    if official_first_write
                    else "a"
                ),
                header=official_first_write
            )

            official_first_write = False


        # ====================================================
        # PROGRESS
        # ====================================================

        elapsed = (
            time.time()
            - generation_start
        )

        rate = (
            chunk_end / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            len(s1)
            - chunk_end
        )

        eta_seconds = (
            remaining / rate
            if rate > 0
            else 0
        )

        eta_hours = (
            eta_seconds / 3600
        )

        print(
            f"Processed "
            f"{chunk_end:,}/"
            f"{len(s1):,} S1 rows | "
            f"Pairs: "
            f"{total_pairs:,} | "
            f"Rate: "
            f"{rate:.2f} S1/sec | "
            f"ETA: "
            f"{eta_hours:.2f} hrs"
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


    # ========================================================
    # OUTPUT SIZES
    # ========================================================

    internal_size_gb = (

        os.path.getsize(
            INTERNAL_OUTPUT
        )

        / (1024 ** 3)

        if os.path.exists(
            INTERNAL_OUTPUT
        )

        else 0
    )


    official_size_gb = (

        os.path.getsize(
            OFFICIAL_OUTPUT
        )

        / (1024 ** 3)

        if os.path.exists(
            OFFICIAL_OUTPUT
        )

        else 0
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")

    print("=" * 70)

    print(
        "FINAL TEST CANDIDATE "
        "GENERATION RESULTS"
    )

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
        f"S1 without candidates  : "
        f"{len(s1) - s1_with_candidates:,}"
    )

    if len(s1) > 0:
        s1_coverage = (
            s1_with_candidates / len(s1)
        ) * 100
    else:
        s1_coverage = 0.0

    print(
        f"S1 coverage            : "
        f"{s1_coverage:.4f}%"
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
        f"Internal output size   : "
        f"{internal_size_gb:.2f} GB"
    )

    print(
        f"Official output size   : "
        f"{official_size_gb:.2f} GB"
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
        if generation_time > 0
        else "Processing rate        : 0"
    )


    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print(
        "\nOutput files:"
    )

    print(
        f"  Internal : "
        f"{INTERNAL_OUTPUT}"
    )

    print(
        f"  Official : "
        f"{OFFICIAL_OUTPUT}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()