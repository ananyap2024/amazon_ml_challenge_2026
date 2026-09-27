import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ============================================================
# PATHS / DEFAULTS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing import normalize_business_name, tokenize

TEST_DIR = PROJECT_DIR / "dataset" / "test"
OUTPUT_DIR = PROJECT_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

S1_PATH = TEST_DIR / "test_source1.tsv"
S2_PATH = TEST_DIR / "test_source2.tsv"
S3_PATH = TEST_DIR / "test_source3.tsv"

INTERNAL_OUTPUT = OUTPUT_DIR / "candidate_pairs_internal.tsv"
OFFICIAL_OUTPUT = OUTPUT_DIR / "candidate_pairs.tsv"
COVERAGE_OUTPUT = OUTPUT_DIR / "test_s1_coverage.tsv"
STATE_OUTPUT = OUTPUT_DIR / "capped_candidate_generation_state.json"

DEFAULT_CAP = 20
DEFAULT_POSTING_TOP_K = 20
DEFAULT_CHUNK_SIZE = 5_000

GENERIC_TOKENS = {
    "private", "limited", "ltd", "llc", "inc", "incorporated",
    "corp", "corporation", "company", "co", "plc", "pvt", "public",
}


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate capped test candidate pairs incrementally."
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N S1 rows. Default: all S1.")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP,
                        help="Maximum unique candidates per S1. Default: 20.")
    parser.add_argument("--posting-top-k", type=int, default=DEFAULT_POSTING_TOP_K,
                        help="Maximum IDs retained per token/name posting list. Default: 20.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="S1 processing chunk size. Default: 5000.")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete previous capped outputs/state and start from scratch.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last committed chunk.")
    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def normalize_id(value):
    return str(value)


def tokenize_name(name):
    tokens = tokenize(name)
    return sorted({
        token for token in tokens
        if token not in GENERIC_TOKENS and len(token) >= 3
    })


def prepare_source(df):
    df = df[["entity_id", "business_name"]].copy()
    df["entity_id"] = df["entity_id"].astype(str)
    df["business_name"] = df["business_name"].fillna("").astype(str)
    df["name_norm"] = df["business_name"].map(normalize_business_name)
    df["name_tokens"] = df["name_norm"].map(tokenize_name)
    return df


def prepare_s1_chunk(df):
    df = df.copy()
    df["entity_id"] = df["entity_id"].astype(str)
    df["business_name"] = df["business_name"].fillna("").astype(str)
    df["name_norm"] = df["business_name"].map(normalize_business_name)
    df["name_tokens"] = df["name_norm"].map(tokenize_name)
    return df


def trim_postings(index, top_k):
    """Keep a deterministic bounded posting list for every key."""
    return {
        key: tuple(sorted(ids)[:top_k])
        for key, ids in index.items()
    }


def build_indexes(df, top_k):
    exact = defaultdict(list)
    token = defaultdict(list)

    for entity_id, name_norm, tokens in zip(
        df["entity_id"], df["name_norm"], df["name_tokens"]
    ):
        entity_id = normalize_id(entity_id)

        if name_norm:
            exact[name_norm].append(entity_id)

        for t in tokens:
            token[t].append(entity_id)

    exact = trim_postings(exact, top_k)
    token = trim_postings(token, top_k)

    frequency = {k: len(v) for k, v in token.items()}

    return exact, token, frequency


def select_tokens(tokens, frequency, max_tokens=2, max_frequency=20_000):
    valid = [
        (frequency[t], t)
        for t in tokens
        if t in frequency and frequency[t] <= max_frequency
    ]
    valid.sort(key=lambda x: (x[0], x[1]))
    return [t for _, t in valid[:max_tokens]]


def generate_top_candidates(row, indexes, cap, posting_top_k):
    """Generate only a bounded candidate pool; never materialize the uncapped set."""
    s2_exact, s2_token, s2_freq, s3_exact, s3_token, s3_freq = indexes

    scores = {}
    source_for_id = {}

    name = row["name_norm"]
    tokens = row["name_tokens"]

    # Exact normalized name gets the strongest priority.
    if name:
        for cid in s2_exact.get(name, ()):
            scores[("source2", cid)] = 1_000_000.0
            source_for_id[cid] = "source2"
        for cid in s3_exact.get(name, ()):
            scores[("source3", cid)] = 1_000_000.0
            source_for_id[cid] = "source3"

    # Only the two most distinctive tokens are consulted per source.
    for source, token_index, frequency in (
        ("source2", s2_token, s2_freq),
        ("source3", s3_token, s3_freq),
    ):
        selected = select_tokens(tokens, frequency)
        for token in selected:
            freq = frequency[token]
            # Rarer tokens receive more weight.
            weight = 1.0 / (1.0 + (freq ** 0.5))
            for cid in token_index.get(token, ()):
                key = (source, cid)
                scores[key] = scores.get(key, 0.0) + weight

    # Deterministic ranking: score desc, then source, then ID.
    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1])
    )

    selected = ranked[:cap]

    s2 = [cid for (source, cid), _ in selected if source == "source2"]
    s3 = [cid for (source, cid), _ in selected if source == "source3"]

    return s2, s3


def count_s1_rows():
    with open(S1_PATH, "r", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def write_state(next_row, internal_offset, official_offset, coverage_offset):
    state = {
        "next_s1_row": next_row,
        "internal_offset": internal_offset,
        "official_offset": official_offset,
        "coverage_offset": coverage_offset,
    }
    tmp = STATE_OUTPUT.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_OUTPUT)


def read_state():
    if not STATE_OUTPUT.exists():
        return None
    with open(STATE_OUTPUT, "r", encoding="utf-8") as f:
        return json.load(f)


def truncate_and_open(path, offset):
    mode = "r+b" if path.exists() else "w+b"
    f = open(path, mode)
    f.truncate(offset)
    f.seek(offset)
    return f


def initialize_outputs(fresh):
    if fresh:
        for path in [INTERNAL_OUTPUT, OFFICIAL_OUTPUT, COVERAGE_OUTPUT, STATE_OUTPUT]:
            if path.exists():
                path.unlink()

    if not INTERNAL_OUTPUT.exists():
        INTERNAL_OUTPUT.write_text(
            "source1_entity_id\tcandidate_entity_id\tsource\n",
            encoding="utf-8",
        )

    if not OFFICIAL_OUTPUT.exists():
        OFFICIAL_OUTPUT.write_text(
            "source1_entity_id\tcandidate_entity_ids\n",
            encoding="utf-8",
        )

    if not COVERAGE_OUTPUT.exists():
        COVERAGE_OUTPUT.write_text(
            "source1_entity_id\tcandidate_count\thas_candidates\n",
            encoding="utf-8",
        )


def format_internal_line(s1_id, cid, source):
    return f"{s1_id}\t{cid}\t{source}\n".encode("utf-8")


def format_official_line(s1_id, candidate_ids):
    return f"{s1_id}\t{','.join(candidate_ids)}\n".encode("utf-8")


def format_coverage_line(s1_id, count):
    return f"{s1_id}\t{count}\t{1 if count else 0}\n".encode("utf-8")


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if args.cap < 1:
        raise ValueError("--cap must be >= 1")
    if args.posting_top_k < 1:
        raise ValueError("--posting-top-k must be >= 1")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    if args.fresh and args.resume:
        raise ValueError("Use either --fresh or --resume, not both.")

    print("=" * 72)
    print("CAPPED TEST CANDIDATE GENERATION")
    print("=" * 72)
    print(f"Candidate cap           : {args.cap}")
    print(f"Posting-list cap        : {args.posting_top_k}")
    print(f"S1 chunk size           : {args.chunk_size:,}")
    print(f"S1 limit                : {args.limit if args.limit is not None else 'ALL'}")
    print("Cap applied during generation: YES")
    print("Uncapped candidate set generated: NO")

    initialize_outputs(args.fresh)

    state = read_state() if args.resume else None
    start_row = int(state["next_s1_row"]) if state else 0

    if args.resume and state is None:
        raise RuntimeError("--resume requested but no checkpoint state exists.")

    # --------------------------------------------------------
    # Load S2/S3 only. Candidate generation is never run first.
    # --------------------------------------------------------
    print("\nLoading official test S2/S3...")
    s2 = prepare_source(pd.read_csv(S2_PATH, sep="\t", usecols=["entity_id", "business_name"], dtype=str))
    s3 = prepare_source(pd.read_csv(S3_PATH, sep="\t", usecols=["entity_id", "business_name"], dtype=str))

    s2_ids = set(s2["entity_id"])
    s3_ids = set(s3["entity_id"])

    print(f"S2 rows                 : {len(s2):,}")
    print(f"S3 rows                 : {len(s3):,}")
    print(f"Unique S2 IDs           : {len(s2_ids):,}")
    print(f"Unique S3 IDs           : {len(s3_ids):,}")

    print("\nBuilding bounded indexes...")
    s2_exact, s2_token, s2_freq = build_indexes(s2, args.posting_top_k)
    s3_exact, s3_token, s3_freq = build_indexes(s3, args.posting_top_k)

    indexes = (s2_exact, s2_token, s2_freq, s3_exact, s3_token, s3_freq)

    total_s1 = count_s1_rows()
    target_s1 = min(total_s1, args.limit) if args.limit is not None else total_s1

    print(f"\nOfficial test S1 rows  : {total_s1:,}")
    print(f"S1 rows to process     : {target_s1:,}")

    if start_row > target_s1:
        raise RuntimeError("Checkpoint is beyond requested S1 limit.")

    # --------------------------------------------------------
    # Resume-safe output handling.
    # --------------------------------------------------------
    if state:
        internal_file = truncate_and_open(INTERNAL_OUTPUT, int(state["internal_offset"]))
        official_file = truncate_and_open(OFFICIAL_OUTPUT, int(state["official_offset"]))
        coverage_file = truncate_and_open(COVERAGE_OUTPUT, int(state["coverage_offset"]))
        print(f"\nResuming from S1 row {start_row:,}...")
    else:
        internal_file = open(INTERNAL_OUTPUT, "ab")
        official_file = open(OFFICIAL_OUTPUT, "ab")
        coverage_file = open(COVERAGE_OUTPUT, "ab")
        start_row = 0

    processed = start_row
    total_pairs = 0
    zero_candidates = 0
    generation_start = time.time()
    last_report = generation_start

    try:
        reader = pd.read_csv(
            S1_PATH,
            sep="\t",
            usecols=["entity_id", "business_name"],
            dtype=str,
            chunksize=args.chunk_size,
        )

        for chunk_start, chunk in enumerate(reader):
            row_start = chunk_start * args.chunk_size
            row_end = row_start + len(chunk)

            if row_end <= start_row:
                continue

            if row_start < start_row:
                chunk = chunk.iloc[start_row - row_start:]
                row_start = start_row

            if row_start >= target_s1:
                break

            if row_end > target_s1:
                chunk = chunk.iloc[:target_s1 - row_start]

            chunk = prepare_s1_chunk(chunk)

            for _, row in chunk.iterrows():
                s1_id = row["entity_id"]
                s2_candidates, s3_candidates = generate_top_candidates(
                    row, indexes, args.cap, args.posting_top_k
                )

                # Defensive validation before writing.
                s2_candidates = [cid for cid in s2_candidates if cid in s2_ids]
                s3_candidates = [cid for cid in s3_candidates if cid in s3_ids]

                # De-duplicate while preserving deterministic order.
                s2_candidates = list(dict.fromkeys(s2_candidates))
                s3_candidates = list(dict.fromkeys(s3_candidates))

                candidate_ids = s2_candidates + s3_candidates
                if len(candidate_ids) > args.cap:
                    candidate_ids = candidate_ids[:args.cap]
                    s2_candidates = [cid for cid in candidate_ids if cid in s2_ids]
                    s3_candidates = [cid for cid in candidate_ids if cid in s3_ids]

                for cid in s2_candidates:
                    internal_file.write(format_internal_line(s1_id, cid, "source2"))
                for cid in s3_candidates:
                    internal_file.write(format_internal_line(s1_id, cid, "source3"))

                official_file.write(format_official_line(s1_id, candidate_ids))
                coverage_file.write(format_coverage_line(s1_id, len(candidate_ids)))

                processed += 1
                total_pairs += len(candidate_ids)

                if not candidate_ids:
                    zero_candidates += 1

            # Commit the completed chunk only after all three outputs are flushed.
            internal_file.flush()
            official_file.flush()
            coverage_file.flush()
            os.fsync(internal_file.fileno())
            os.fsync(official_file.fileno())
            os.fsync(coverage_file.fileno())

            write_state(
                processed,
                internal_file.tell(),
                official_file.tell(),
                coverage_file.tell(),
            )

            now = time.time()
            elapsed = now - generation_start
            rate = (processed - start_row) / elapsed if elapsed > 0 else 0
            remaining = target_s1 - processed
            eta_seconds = remaining / rate if rate > 0 else 0

            print(
                f"Processed {processed:,}/{target_s1:,} S1 | "
                f"pairs={total_pairs:,} | "
                f"rate={rate:.2f} S1/s | "
                f"ETA={eta_seconds / 60:.1f} min"
            )

            last_report = now

    except KeyboardInterrupt:
        print("\nInterrupted safely. The last completed chunk is checkpointed.")
        print(f"Resume with: python src/generate_capped_test_candidates.py --resume")
        return
    finally:
        internal_file.close()
        official_file.close()
        coverage_file.close()

    # --------------------------------------------------------
    # Final state and summary.
    # --------------------------------------------------------
    elapsed = time.time() - generation_start
    internal_size = INTERNAL_OUTPUT.stat().st_size / (1024 ** 3)
    official_size = OFFICIAL_OUTPUT.stat().st_size / (1024 ** 3)
    coverage_size = COVERAGE_OUTPUT.stat().st_size / (1024 ** 3)

    print("\n" + "=" * 72)
    print("CAPPED CANDIDATE GENERATION COMPLETE")
    print("=" * 72)
    print(f"S1 processed            : {processed:,}")
    print(f"Candidate pairs         : {total_pairs:,}")
    print(f"Average candidates/S1   : {total_pairs / processed:.2f}")
    print(f"Zero-candidate S1s      : {zero_candidates:,}")
    print(f"Maximum candidates/S1   : {args.cap}")
    print(f"Generation time         : {elapsed / 60:.2f} min")
    print(f"Internal output         : {internal_size:.2f} GB")
    print(f"Official output         : {official_size:.2f} GB")
    print(f"Coverage output         : {coverage_size:.2f} GB")
    print(f"\nInternal file: {INTERNAL_OUTPUT}")
    print(f"Official file: {OFFICIAL_OUTPUT}")
    print(f"Coverage file: {COVERAGE_OUTPUT}")
    print(f"Checkpoint : {STATE_OUTPUT}")

    # Mark complete only after all outputs are durable.
    write_state(
        processed,
        INTERNAL_OUTPUT.stat().st_size,
        OFFICIAL_OUTPUT.stat().st_size,
        COVERAGE_OUTPUT.stat().st_size,
    )


if __name__ == "__main__":
    main()
