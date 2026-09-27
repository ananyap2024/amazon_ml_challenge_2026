import sys
from pathlib import Path
from collections import defaultdict
import time

import pandas as pd

# ============================================================
# PATHS
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATASET_DIR = PROJECT_DIR / "dataset"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

from preprocessing import (
    normalize_business_name,
    normalize_address,
    tokenize,
)


# ============================================================
# CONFIGURATION
# ============================================================

S1_PILOT_ROWS = 10_000

MAX_TOKENS = 2
MAX_TOKEN_FREQUENCY = 20_000

OUTPUT_FILE = OUTPUT_DIR / "candidate_pairs_pilot.tsv"


# ============================================================
# GENERIC NAME TOKENS
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
    print("LOADING TEST DATA")
    print("=" * 70)

    s1_path = TEST_DIR / "test_source1.tsv"
    s2_path = TEST_DIR / "test_source2.tsv"
    s3_path = TEST_DIR / "test_source3.tsv"

    print("Loading S1 pilot...")
    s1 = pd.read_csv(
        s1_path,
        sep="\t",
        nrows=S1_PILOT_ROWS,
    )

    print("Loading S2...")
    s2 = pd.read_csv(
        s2_path,
        sep="\t",
    )

    print("Loading S3...")
    s3 = pd.read_csv(
        s3_path,
        sep="\t",
    )

    print()
    print(f"S1 pilot rows : {len(s1):,}")
    print(f"S2 rows       : {len(s2):,}")
    print(f"S3 rows       : {len(s3):,}")

    return s1, s2, s3


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
# PREPARE DATAFRAME
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

def build_token_index(df):

    index = defaultdict(set)

    for _, row in df.iterrows():

        entity_id = row["entity_id"]

        for token in row["name_tokens"]:
            index[token].add(entity_id)

    return dict(index)


# ============================================================
# TOKEN FREQUENCY
# ============================================================

def build_token_frequency(token_index):

    return {
        token: len(entity_ids)
        for token, entity_ids in token_index.items()
    }


# ============================================================
# SELECT DISTINCTIVE TOKENS
# ============================================================

def select_distinctive_tokens(
    tokens,
    token_frequency,
):

    valid_tokens = [
        token
        for token in tokens
        if token in token_frequency
        and token_frequency[token] <= MAX_TOKEN_FREQUENCY
    ]

    valid_tokens.sort(
        key=lambda token: token_frequency[token]
    )

    return valid_tokens[:MAX_TOKENS]


# ============================================================
# BUILD ALL INDEXES
# ============================================================

def build_indexes(s2, s3):

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

    print("Building name-token index...")
    s2_token_index = build_token_index(s2)

    print("Calculating S2 token frequencies...")
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

    print("Building name-token index...")
    s3_token_index = build_token_index(s3)

    print("Calculating S3 token frequencies...")
    s3_token_frequency = build_token_frequency(
        s3_token_index
    )

    print()
    print("All indexes ready.")

    return (
        s2_name_index,
        s3_name_index,
        s2_address_index,
        s3_address_index,
        s2_token_index,
        s3_token_index,
        s2_token_frequency,
        s3_token_frequency,
    )


# ============================================================
# GENERATE CANDIDATES FOR ONE S1 ROW
# ============================================================

def generate_candidates(
    row,
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

    s2_candidates = set()
    s3_candidates = set()

    # --------------------------------------------------------
    # BLOCK 1: EXACT NORMALIZED NAME
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # BLOCK 2: EXACT NORMALIZED ADDRESS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # BLOCK 3: TWO DISTINCTIVE NAME TOKENS
    #
    # UNION:
    # Any candidate retrieved by either selected token
    # is included.
    # --------------------------------------------------------

    s2_tokens = select_distinctive_tokens(
        row["name_tokens"],
        s2_token_frequency,
    )

    s3_tokens = select_distinctive_tokens(
        row["name_tokens"],
        s3_token_frequency,
    )

    for token in s2_tokens:

        s2_candidates.update(
            s2_token_index.get(
                token,
                set(),
            )
        )

    for token in s3_tokens:

        s3_candidates.update(
            s3_token_index.get(
                token,
                set(),
            )
        )

    # --------------------------------------------------------
    # BUILD INTERNAL OUTPUT RECORDS
    # --------------------------------------------------------

    source1_id = row["entity_id"]

    records = []

    for candidate_id in s2_candidates:

        records.append(
            (
                source1_id,
                candidate_id,
                "source2",
            )
        )

    for candidate_id in s3_candidates:

        records.append(
            (
                source1_id,
                candidate_id,
                "source3",
            )
        )

    return records


# ============================================================
# MAIN GENERATION
# ============================================================

def main():

    start_time = time.time()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    s1, s2, s3 = load_data()

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

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
    # INDEXES
    # --------------------------------------------------------

    indexes = build_indexes(
        s2,
        s3,
    )

    # --------------------------------------------------------
    # REMOVE OLD OUTPUT
    # --------------------------------------------------------

    if OUTPUT_FILE.exists():

        print()
        print(
            f"Removing existing output: {OUTPUT_FILE}"
        )

        OUTPUT_FILE.unlink()

    # --------------------------------------------------------
    # OUTPUT HEADER
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        f.write(
            "source1_entity_id\t"
            "candidate_entity_id\t"
            "source\n"
        )

    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GENERATING PILOT CANDIDATES")
    print("=" * 70)

    total_candidates = 0
    max_candidates = 0
    min_candidates = None

    start_generation = time.time()

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8",
        newline="",
    ) as f:

        for i, (_, row) in enumerate(
            s1.iterrows(),
            start=1,
        ):

            records = generate_candidates(
                row,
                indexes,
            )

            candidate_count = len(records)

            total_candidates += candidate_count

            max_candidates = max(
                max_candidates,
                candidate_count,
            )

            if min_candidates is None:
                min_candidates = candidate_count
            else:
                min_candidates = min(
                    min_candidates,
                    candidate_count,
                )

            # Write immediately.
            for (
                source1_id,
                candidate_id,
                source,
            ) in records:

                f.write(
                    f"{source1_id}\t"
                    f"{candidate_id}\t"
                    f"{source}\n"
                )

            if i % 100 == 0:

                elapsed = (
                    time.time()
                    - start_generation
                )

                rate = (
                    i / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"Processed "
                    f"{i:,}/{len(s1):,} "
                    f"S1 | "
                    f"Candidates: "
                    f"{total_candidates:,} | "
                    f"Avg/S1: "
                    f"{total_candidates / i:,.2f} | "
                    f"Rate: "
                    f"{rate:.2f} S1/sec"
                )

    # --------------------------------------------------------
    # FINAL STATISTICS
    # --------------------------------------------------------

    elapsed_generation = (
        time.time()
        - start_generation
    )

    total_elapsed = (
        time.time()
        - start_time
    )

    average_candidates = (
        total_candidates / len(s1)
        if len(s1) > 0
        else 0
    )

    output_size_mb = (
        OUTPUT_FILE.stat().st_size
        / (1024 * 1024)
    )

    print()
    print("=" * 70)
    print("PILOT GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"S1 rows processed       : {len(s1):,}"
    )

    print(
        f"Total candidates        : {total_candidates:,}"
    )

    print(
        f"Average candidates/S1   : "
        f"{average_candidates:,.2f}"
    )

    print(
        f"Minimum candidates/S1   : "
        f"{min_candidates:,}"
    )

    print(
        f"Maximum candidates/S1   : "
        f"{max_candidates:,}"
    )

    print(
        f"Output file size        : "
        f"{output_size_mb:,.2f} MB"
    )

    print(
        f"Generation time         : "
        f"{elapsed_generation / 60:.2f} minutes"
    )

    print(
        f"Total execution time    : "
        f"{total_elapsed / 60:.2f} minutes"
    )

    print()
    print(
        f"Output file:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print(
        "This is ONLY a 10,000-S1 pilot."
    )

    print(
        "Do NOT start full test generation yet."
    )

    print(
        "Use the statistics above to decide "
        "whether the full generation is practical."
    )


if __name__ == "__main__":
    main()