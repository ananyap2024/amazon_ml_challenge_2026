"""Production batch/streaming inference driver for Member 2 matching model.

Connects Member 1's candidate pairs (candidate_pairs_internal.tsv) to the
13-feature XGBoost matching model (inference.py, features_13.py) and produces
scored candidate matches for Member 3's post-processing and submission stage.

Key Properties:
1. Reuses inference.py and features_13.py directly.
2. Streams candidate pairs in chunks (never loads the 161.5M file into memory).
3. Preserves input candidate IDs, source, and exact row order.
4. Threshold-decoupled: NO default production threshold is hard-coded.
   If threshold is omitted (None), outputs match_probability only.
   If threshold is provided (e.g. 0.60), also outputs predicted_match (0/1).
5. Fast record lookup: source records are loaded/indexed once, allowing O(1)
   lookups per candidate chunk rather than re-reading gigabytes of disk per chunk.
6. Safe missing record / missing value handling with detailed audit stats.

CLI Example (Member 3 usage):
    python code/business_entity_resolution/src/batch_inference.py \\
        --pairs dataset/test/candidate_pairs_internal.tsv \\
        --source1 dataset/test/test_source1.tsv \\
        --source2 dataset/test/test_source2.tsv \\
        --source3 dataset/test/test_source3.tsv \\
        --output matching_results_scored.tsv \\
        --threshold 0.60
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure sibling modules under code/business_entity_resolution/src/ are importable
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import inference  # noqa: E402
from features_13 import RECORD_COLUMNS  # noqa: E402

DEFAULT_CHUNK_SIZE = 10_000
SOURCE_COLUMNS = ["entity_id"] + RECORD_COLUMNS  # entity_id, business_name, business_address, country
PAIR_COLUMNS = ["source1_entity_id", "candidate_entity_id", "source"]
SOURCE_PREFIX = {"source2": "S2-", "source3": "S3-"}


def load_source_records(path, needed_ids=None):
    """Load records from a source TSV into an entity_id -> dict lookup.

    Parameters
    ----------
    path : str or Path
        Path to source TSV (e.g. source1.tsv, source2.tsv, source3.tsv).
    needed_ids : set or None
        If provided, only loads records whose entity_id is in needed_ids and
        breaks early once all needed IDs are found.
        If None, loads all records from the file.

    Returns
    -------
    dict
        Mapping entity_id -> {
            "entity_id": str,
            "business_name": str/None,
            "business_address": str/None,
            "country": str/None
        }
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    lookup = {}
    usecols = SOURCE_COLUMNS
    needed_set = set(needed_ids) if needed_ids is not None else None

    # Read TSV with pandas standard defaults so NaNs are preserved consistently
    for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=500_000):
        if needed_set is not None:
            chunk = chunk[chunk["entity_id"].isin(needed_set)]
            if chunk.empty:
                continue

        chunk_dict = chunk.set_index("entity_id").to_dict(orient="index")
        for eid, rec in chunk_dict.items():
            rec["entity_id"] = eid
        lookup.update(chunk_dict)

        if needed_set is not None and len(lookup) >= len(needed_set):
            break

    return lookup


def validate_pair_format(s1_id, cand_id, source):
    """Validate candidate pair schema and ID prefixes.

    Returns empty string if valid, or a descriptive reason if invalid.
    """
    if not s1_id or not cand_id or not source:
        return "missing_pair_fields"
    if source not in SOURCE_PREFIX:
        return f"invalid_source_{source}"
    if not s1_id.startswith("S1-"):
        return "source1_id_bad_prefix"
    expected_prefix = SOURCE_PREFIX[source]
    if not cand_id.startswith(expected_prefix):
        return f"candidate_id_prefix_mismatch_expected_{expected_prefix}"
    return ""


def write_checkpoint(state_path, stats, output_offset):
    """Save an atomic checkpoint of streaming inference progress."""
    state = {
        "total_pairs_read": stats["total_pairs_read"],
        "total_pairs_scored": stats["total_pairs_scored"],
        "total_pairs_skipped": stats["total_pairs_skipped"],
        "predicted_matches_count": stats["predicted_matches_count"],
        "chunks_processed": stats["chunks_processed"],
        "output_offset": output_offset,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp = Path(state_path).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, state_path)


def read_checkpoint(state_path):
    """Read existing checkpoint state if present."""
    state_path = Path(state_path)
    if not state_path.is_file():
        return None
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_batch_inference(
    pairs_path,
    source1_path,
    source2_path,
    source3_path,
    output_path,
    model_path=None,
    tfidf_path=None,
    threshold=None,
    chunk_size=DEFAULT_CHUNK_SIZE,
    max_pairs=None,
    summary_path=None,
    log_interval=10,
    resume=False,
    fresh=False,
):
    """Score candidate pairs in streaming chunks and write predictions.

    Parameters
    ----------
    pairs_path : str or Path
        Candidate pairs TSV with columns: source1_entity_id, candidate_entity_id, source.
    source1_path : str or Path
        Source 1 TSV containing deduplicated reference entities.
    source2_path : str or Path
        Source 2 TSV containing candidate records.
    source3_path : str or Path
        Source 3 TSV containing candidate records.
    output_path : str or Path
        Output TSV file path for scored candidates.
    model_path : str or Path, optional
        Path to matching_model.pkl (defaults to inference.DEFAULT_MODEL_PATH).
    tfidf_path : str or Path, optional
        Path to tfidf_vectorizers.pkl (defaults to inference.DEFAULT_TFIDF_PATH).
    threshold : float, optional
        Binary classification threshold. If None, only match_probability is output.
        If a float (e.g. 0.60), adds a predicted_match column (0 or 1).
    chunk_size : int
        Number of candidate pairs to process in memory per batch (default 10,000).
    max_pairs : int, optional
        Maximum total candidate pairs to process. Useful for smoke tests / debugging.
    summary_path : str or Path, optional
        Optional path to write a JSON summary with execution statistics.
    log_interval : int
        Print progress every N chunks.

    Returns
    -------
    dict
        Run summary with counts, timings, and skipped statistics.
    """
    start_time = time.time()
    pairs_path = Path(pairs_path)
    output_path = Path(output_path)

    if not pairs_path.is_file():
        raise FileNotFoundError(f"Candidate pairs file not found: {pairs_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_suffix(".tsv.tmp")
    state_path = output_path.with_suffix(".state.json")

    if fresh and resume:
        raise ValueError("Use either --fresh or --resume, not both.")

    if fresh:
        if temp_output_path.exists():
            temp_output_path.unlink()
        if state_path.exists():
            state_path.unlink()
        if output_path.exists():
            output_path.unlink()

    checkpoint = read_checkpoint(state_path) if resume else None
    if resume and checkpoint is None:
        raise RuntimeError(f"--resume requested but no checkpoint state exists at: {state_path}")

    print("=" * 70)
    print("PRODUCTION BATCH INFERENCE DRIVER")
    print("=" * 70)
    print(f"Candidate pairs : {pairs_path}")
    print(f"Source 1        : {source1_path}")
    print(f"Source 2        : {source2_path}")
    print(f"Source 3        : {source3_path}")
    print(f"Output path     : {output_path}")
    print(f"Chunk size      : {chunk_size:,}")
    print(f"Threshold       : {threshold if threshold is not None else 'None (outputting probabilities only)'}")
    if max_pairs is not None:
        print(f"Max pairs cap   : {max_pairs:,}")
    if resume:
        print(f"Mode            : RESUME from checkpoint ({checkpoint.get('chunks_processed', 0):,} chunks done)")
    elif fresh:
        print(f"Mode            : FRESH start (cleared previous outputs/checkpoints)")

    # 1. Load model and vectorizers
    print("\n[1/3] Loading production model and TF-IDF vectorizers...")
    model_kwargs = {"model_path": model_path} if model_path else {}
    tfidf_kwargs = {"path": tfidf_path} if tfidf_path else {}
    model = inference.load_model(**model_kwargs)
    name_vectorizer, address_vectorizer = inference.load_vectorizers(**tfidf_kwargs)
    print("      Model and vectorizers successfully loaded and validated.")

    # 2. Load source lookups
    print("\n[2/3] Loading source record lookups...")
    t_source_start = time.time()

    # For small runs or sample files (<10MB), selectively load only needed IDs for extreme speed
    file_size_bytes = pairs_path.stat().st_size
    is_small_candidate_set = (max_pairs is not None and max_pairs <= 50_000) or (file_size_bytes <= 10_000_000)

    if is_small_candidate_set:
        print(f"      Small candidate input detected ({file_size_bytes / 1024:.1f} KB). Extracting needed entity IDs...")
        sample_df = pd.read_csv(pairs_path, sep="\t", nrows=max_pairs)
        s1_needed = set(sample_df["source1_entity_id"].dropna())
        cand_needed = set(sample_df["candidate_entity_id"].dropna())
        s2_needed = {eid for eid in cand_needed if eid.startswith("S2-")}
        s3_needed = {eid for eid in cand_needed if eid.startswith("S3-")}
        s1_lookup = load_source_records(source1_path, needed_ids=s1_needed)
        s2_lookup = load_source_records(source2_path, needed_ids=s2_needed)
        s3_lookup = load_source_records(source3_path, needed_ids=s3_needed)
    else:
        s1_lookup = load_source_records(source1_path)
        s2_lookup = load_source_records(source2_path)
        s3_lookup = load_source_records(source3_path)

    print(f"      Source 1 records loaded: {len(s1_lookup):,}")
    print(f"      Source 2 records loaded: {len(s2_lookup):,}")
    print(f"      Source 3 records loaded: {len(s3_lookup):,}")
    print(f"      Source loading took {time.time() - t_source_start:.2f}s")

    # 3. Stream candidate chunks and perform inference
    print("\n[3/3] Streaming candidate pairs and predicting match probabilities...")
    stats = {
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "pairs_path": str(pairs_path),
        "output_path": str(output_path),
        "threshold": threshold,
        "chunk_size": chunk_size,
        "total_pairs_read": 0,
        "total_pairs_scored": 0,
        "total_pairs_skipped": 0,
        "skipped_reasons": {},
        "predicted_matches_count": 0 if threshold is not None else None,
        "chunks_processed": 0,
    }

    output_columns = PAIR_COLUMNS + ["match_probability"]
    if threshold is not None:
        output_columns.append("predicted_match")

    t_inference_start = time.time()

    if checkpoint:
        resume_offset = checkpoint.get("output_offset", 0)
        stats["total_pairs_read"] = checkpoint.get("total_pairs_read", 0)
        stats["total_pairs_scored"] = checkpoint.get("total_pairs_scored", 0)
        stats["total_pairs_skipped"] = checkpoint.get("total_pairs_skipped", 0)
        stats["predicted_matches_count"] = checkpoint.get("predicted_matches_count", 0 if threshold is not None else None)
        stats["chunks_processed"] = checkpoint.get("chunks_processed", 0)

        out_f = open(temp_output_path, "r+b")
        out_f.truncate(resume_offset)
        out_f.seek(resume_offset)
        print(f"      Resuming from chunk {stats['chunks_processed']:,} ({stats['total_pairs_read']:,} pairs read) at byte offset {resume_offset:,}...")
    else:
        out_f = open(temp_output_path, "w+b")
        out_f.write(("\t".join(output_columns) + "\n").encode("utf-8"))
        out_f.flush()
        os.fsync(out_f.fileno())

    try:
        reader = pd.read_csv(pairs_path, sep="\t", chunksize=chunk_size)

        for chunk_idx, chunk in enumerate(reader):
            if resume and chunk_idx < stats["chunks_processed"]:
                continue

            if max_pairs is not None and stats["total_pairs_read"] >= max_pairs:
                break

            if max_pairs is not None:
                remaining = max_pairs - stats["total_pairs_read"]
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()

            chunk_pairs_count = len(chunk)
            stats["total_pairs_read"] += chunk_pairs_count

            batch = []
            valid_rows = []

            for row in chunk.itertuples(index=False):
                s1_id = getattr(row, "source1_entity_id", "")
                cand_id = getattr(row, "candidate_entity_id", "")
                source_name = getattr(row, "source", "")

                fmt_error = validate_pair_format(s1_id, cand_id, source_name)
                if fmt_error:
                    stats["total_pairs_skipped"] += 1
                    stats["skipped_reasons"][fmt_error] = stats["skipped_reasons"].get(fmt_error, 0) + 1
                    continue

                s1_rec = s1_lookup.get(s1_id)
                if s1_rec is None:
                    stats["total_pairs_skipped"] += 1
                    reason = "source1_record_not_found"
                    stats["skipped_reasons"][reason] = stats["skipped_reasons"].get(reason, 0) + 1
                    continue

                cand_rec = s2_lookup.get(cand_id) if source_name == "source2" else s3_lookup.get(cand_id)
                if cand_rec is None:
                    stats["total_pairs_skipped"] += 1
                    reason = "candidate_record_not_found"
                    stats["skipped_reasons"][reason] = stats["skipped_reasons"].get(reason, 0) + 1
                    continue

                batch.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cand_id,
                    "source": source_name,
                    "source1_record": s1_rec,
                    "candidate_record": cand_rec,
                })
                valid_rows.append((s1_id, cand_id, source_name))

            if batch:
                probabilities = inference.predict_batch(
                    batch,
                    model=model,
                    name_vectorizer=name_vectorizer,
                    address_vectorizer=address_vectorizer,
                )

                if threshold is not None:
                    predictions = inference.apply_threshold(probabilities, threshold)
                    stats["predicted_matches_count"] += int(predictions.sum())

                    for (s1_id, cand_id, src), prob, pred in zip(valid_rows, probabilities, predictions):
                        out_f.write(f"{s1_id}\t{cand_id}\t{src}\t{prob:.6f}\t{pred}\n".encode("utf-8"))
                else:
                    for (s1_id, cand_id, src), prob in zip(valid_rows, probabilities):
                        out_f.write(f"{s1_id}\t{cand_id}\t{src}\t{prob:.6f}\n".encode("utf-8"))

                stats["total_pairs_scored"] += len(batch)

            stats["chunks_processed"] += 1

            # Commit chunk progress
            out_f.flush()
            os.fsync(out_f.fileno())
            write_checkpoint(state_path, stats, out_f.tell())

            if (chunk_idx + 1) % log_interval == 0 or (max_pairs and stats["total_pairs_read"] >= max_pairs):
                elapsed = time.time() - t_inference_start
                rate = stats["total_pairs_scored"] / elapsed if elapsed > 0 else 0
                print(f"      Chunk {chunk_idx + 1:4d} | "
                      f"Pairs read: {stats['total_pairs_read']:9,d} | "
                      f"Scored: {stats['total_pairs_scored']:9,d} | "
                      f"Speed: {rate:6.1f} pairs/s")
    finally:
        out_f.close()

    # Atomic rename from temporary to target file
    if temp_output_path.exists():
        os.replace(temp_output_path, output_path)
    if state_path.exists():
        state_path.unlink()

    total_time = time.time() - start_time
    stats["total_seconds"] = total_time
    stats["throughput_pairs_per_sec"] = stats["total_pairs_scored"] / (time.time() - t_inference_start) if (time.time() - t_inference_start) > 0 else 0
    stats["completed_at_utc"] = datetime.now(timezone.utc).isoformat()

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)
    print(f"Total candidate pairs read   : {stats['total_pairs_read']:,}")
    print(f"Total candidate pairs scored : {stats['total_pairs_scored']:,}")
    print(f"Total candidate pairs skipped: {stats['total_pairs_skipped']:,}")
    if stats["skipped_reasons"]:
        print(f"Skipped breakdown            : {stats['skipped_reasons']}")
    if threshold is not None:
        print(f"Predicted matches (>= {threshold:.2f}) : {stats['predicted_matches_count']:,}")
    print(f"Total elapsed time           : {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"Final output written to      : {output_path}")

    if summary_path:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Summary saved to             : {summary_path}")

    return stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Production batch/streaming inference driver for 13-feature entity matching model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score candidate pairs and output probabilities (threshold decoupled for Member 3):
  python batch_inference.py \\
      --pairs ../../../student_resource/dataset/test/candidate_pairs_internal.tsv \\
      --source1 ../../../student_resource/dataset/test/test_source1.tsv \\
      --source2 ../../../student_resource/dataset/test/test_source2.tsv \\
      --source3 ../../../student_resource/dataset/test/test_source3.tsv \\
      --output matching_results_scored.tsv

  # Score candidate pairs with an explicit threshold (e.g. 0.60):
  python batch_inference.py \\
      --pairs candidate_pairs_internal.tsv \\
      --source1 test_source1.tsv \\
      --source2 test_source2.tsv \\
      --source3 test_source3.tsv \\
      --output matching_results_thresholded.tsv \\
      --threshold 0.60
        """,
    )
    parser.add_argument("--pairs", required=True, help="Path to candidate pairs TSV (source1_entity_id, candidate_entity_id, source)")
    parser.add_argument("--source1", required=True, help="Path to Source 1 TSV (entity_id, business_name, business_address, country)")
    parser.add_argument("--source2", required=True, help="Path to Source 2 TSV (entity_id, business_name, business_address, country)")
    parser.add_argument("--source3", required=True, help="Path to Source 3 TSV (entity_id, business_name, business_address, country)")
    parser.add_argument("--output", required=True, help="Path to write output TSV")
    parser.add_argument("--threshold", type=float, default=None, help="Optional binary match threshold. If omitted, only probabilities are output.")
    parser.add_argument("--model", default=None, help="Optional path to matching_model.pkl (defaults to production models dir)")
    parser.add_argument("--tfidf", default=None, help="Optional path to tfidf_vectorizers.pkl (defaults to production models dir)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help=f"Number of pairs per batch (default: {DEFAULT_CHUNK_SIZE:,})")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional maximum number of candidate pairs to process (for testing)")
    parser.add_argument("--summary", default=None, help="Optional path to save JSON run summary")
    parser.add_argument("--log-interval", type=int, default=10, help="Print progress log every N chunks (default: 10)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint state.")
    parser.add_argument("--fresh", action="store_true", help="Clear existing partial output and checkpoint before running.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch_inference(
        pairs_path=args.pairs,
        source1_path=args.source1,
        source2_path=args.source2,
        source3_path=args.source3,
        output_path=args.output,
        model_path=args.model,
        tfidf_path=args.tfidf,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
        max_pairs=args.max_pairs,
        summary_path=args.summary,
        log_interval=args.log_interval,
        resume=args.resume,
        fresh=args.fresh,
    )
