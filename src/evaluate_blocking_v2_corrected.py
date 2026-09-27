"""
ML Challenge 2026 — Corrected Reproducible Blocking V2 Evaluation

Evaluates Member 1's blocking_v2 algorithm on a controlled, reproducible
sample of training S1 entities against a ground-truth-complete target pool.

Every available ground-truth match for the sampled S1 entities is guaranteed
to be present in the target pool, ensuring that candidate recall measures the true
retrieval capability of the blocking keys (exact name + token inverted index)
rather than artificial omissions from random subsampling.
"""

import json
import os
import sys
import time
import tracemalloc
from pathlib import Path
import pandas as pd

# Add src to sys.path so Member 1's modules can be imported directly
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from blocking_v2 import (
    prepare_dataframe,
    build_exact_name_index,
    build_name_token_index,
    build_token_frequency,
    exact_name_candidates,
    token_candidates,
)

TRAIN_DIR = PROJECT_DIR / "dataset" / "train"
OUTPUT_DIR = PROJECT_DIR / "output"


def load_s1_sample_and_ground_truth(n_samples=2000, random_seed=42):
    """
    Load ground truth and select a reproducible sample of S1 records.
    Joins ground truth by source1_entity_id (never assuming row alignment).
    """
    print(f"Loading ground truth from {TRAIN_DIR / 'train_ground_truth.tsv'}...")
    gt_df = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t",
        dtype=str
    )
    # Map source1_entity_id -> set of matched entity IDs
    gt_map = {}
    for s1_id, matched in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"].fillna("")):
        s1_id_clean = s1_id.strip()
        mids = {m.strip() for m in matched.split(",") if m.strip()} if matched.strip() else set()
        gt_map[s1_id_clean] = mids
    del gt_df

    print(f"Loading S1 source records from {TRAIN_DIR / 'train_source1.tsv'}...")
    s1_full = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t",
        usecols=["entity_id", "business_name"],
        dtype=str
    )
    
    # Reproducible sampling with fixed seed
    s1_sample = s1_full.sample(n=n_samples, random_state=random_seed).copy()
    s1_sample["entity_id"] = s1_sample["entity_id"].str.strip()
    del s1_full

    # Map true matches for the sample
    sample_gt = {}
    target_s2_ids = set()
    target_s3_ids = set()

    for s1_id in s1_sample["entity_id"]:
        mids = gt_map.get(s1_id, set())
        sample_gt[s1_id] = mids
        for m in mids:
            if m.startswith("S2-"):
                target_s2_ids.add(m)
            elif m.startswith("S3-"):
                target_s3_ids.add(m)

    return s1_sample, sample_gt, target_s2_ids, target_s3_ids


def construct_target_pool(source_path, target_ids, non_matching_limit=100000, random_seed=42):
    """
    Constructs a target pool containing:
    1. ALL ground-truth matching target entities for the sample.
    2. A reproducible sample of non-matching target entities up to non_matching_limit.

    Reads in streaming chunks to maintain a minimal memory footprint (< 300 MB).
    """
    print(f"Streaming target source: {source_path}...")
    matched_records = []
    non_matching_records = []
    found_target_ids = set()

    # Step 1: Stream file in chunks
    chunk_size = 200_000
    for chunk in pd.read_csv(
        source_path,
        sep="\t",
        usecols=["entity_id", "business_name"],
        dtype=str,
        chunksize=chunk_size
    ):
        chunk["entity_id"] = chunk["entity_id"].str.strip()
        chunk["business_name"] = chunk["business_name"].fillna("")

        # Extract ground truth target matches
        is_target = chunk["entity_id"].isin(target_ids)
        if is_target.any():
            matched_chunk = chunk[is_target]
            matched_records.append(matched_chunk)
            found_target_ids.update(matched_chunk["entity_id"])

        # Collect non-matching candidates for sampling
        non_match_chunk = chunk[~is_target]
        # Sample proportionally from chunk to avoid loading all 5M rows into memory
        sample_from_chunk = non_match_chunk.sample(
            n=min(len(non_match_chunk), int(non_matching_limit * (len(chunk) / 5_000_000) * 1.2)),
            random_state=random_seed
        )
        non_matching_records.append(sample_from_chunk)

    matched_df = pd.concat(matched_records, ignore_index=True) if matched_records else pd.DataFrame(columns=["entity_id", "business_name"])
    non_matched_df = pd.concat(non_matching_records, ignore_index=True) if non_matching_records else pd.DataFrame(columns=["entity_id", "business_name"])

    # Trim non-matching to exact limit
    if len(non_matched_df) > non_matching_limit:
        non_matched_df = non_matched_df.sample(n=non_matching_limit, random_state=random_seed)

    pool_df = pd.concat([matched_df, non_matched_df], ignore_index=True).drop_duplicates(subset=["entity_id"])
    
    missing_ids = target_ids - found_target_ids
    return pool_df, found_target_ids, missing_ids


def evaluate_blocking_v2():
    tracemalloc.start()
    t_start = time.perf_counter()

    print("=" * 80)
    print("CORRECTED REPRODUCIBLE BLOCKING V2 EVALUATION")
    print("=" * 80)

    N_S1 = 2000
    RANDOM_SEED = 42
    NON_MATCHING_POOL_SIZE = 100_000
    MAX_TOKENS = 3
    MAX_FREQUENCY = 5000

    # 1. Sample S1 and collect all true match target IDs
    s1_sample, sample_gt, target_s2_ids, target_s3_ids = load_s1_sample_and_ground_truth(
        n_samples=N_S1,
        random_seed=RANDOM_SEED
    )

    print(f"Sampled S1 entities:                  {len(s1_sample):,}")
    total_true_links = sum(len(mids) for mids in sample_gt.values())
    total_s2_links = sum(len([m for m in mids if m.startswith('S2-')]) for mids in sample_gt.values())
    total_s3_links = sum(len([m for m in mids if m.startswith('S3-')]) for mids in sample_gt.values())
    singletons = sum(1 for mids in sample_gt.values() if len(mids) == 0)

    print(f"Total ground-truth links in sample:    {total_true_links:,}")
    print(f"  - Source 2 target links:            {total_s2_links:,} ({len(target_s2_ids):,} unique IDs)")
    print(f"  - Source 3 target links:            {total_s3_links:,} ({len(target_s3_ids):,} unique IDs)")
    print(f"  - Singletons (no matches):          {singletons:,} ({singletons / len(s1_sample) * 100:.2f}%)")

    # 2. Build target pools
    s2_pool, s2_found_targets, s2_missing = construct_target_pool(
        TRAIN_DIR / "train_source2.tsv",
        target_s2_ids,
        non_matching_limit=NON_MATCHING_POOL_SIZE,
        random_seed=RANDOM_SEED
    )
    s3_pool, s3_found_targets, s3_missing = construct_target_pool(
        TRAIN_DIR / "train_source3.tsv",
        target_s3_ids,
        non_matching_limit=NON_MATCHING_POOL_SIZE,
        random_seed=RANDOM_SEED
    )

    print("\nTarget Pool Construction:")
    print(f"  - Source 2 pool size:               {len(s2_pool):,} records (Verified GT IDs: {len(s2_found_targets):,}, Missing: {len(s2_missing)})")
    print(f"  - Source 3 pool size:               {len(s3_pool):,} records (Verified GT IDs: {len(s3_found_targets):,}, Missing: {len(s3_missing)})")

    # 3. Prepare dataframes using Member 1's exact preprocessing
    print("\nPreprocessing dataframes using Member 1's blocking_v2...")
    t_prep_start = time.perf_counter()
    s1_prep = prepare_dataframe(s1_sample)
    s2_prep = prepare_dataframe(s2_pool)
    s3_prep = prepare_dataframe(s3_pool)
    t_prep = time.perf_counter() - t_prep_start
    print(f"Preprocessing completed in {t_prep:.2f}s.")

    # 4. Build exact name and token inverted indexes
    print("\nBuilding indexes...")
    t_idx_start = time.perf_counter()
    s2_exact_index = build_exact_name_index(s2_prep)
    s3_exact_index = build_exact_name_index(s3_prep)
    s2_token_index = build_name_token_index(s2_prep)
    s3_token_index = build_name_token_index(s3_prep)
    s2_token_frequency = build_token_frequency(s2_token_index)
    s3_token_frequency = build_token_frequency(s3_token_index)
    t_idx = time.perf_counter() - t_idx_start
    print(f"Indexes built in {t_idx:.2f}s.")

    # 5. Run Candidate Generation
    print("\nRunning candidate generation over S1 sample...")
    t_gen_start = time.perf_counter()

    c_counts_total = []
    c_counts_s2 = []
    c_counts_s3 = []

    retrieved_s2 = 0
    retrieved_s3 = 0
    retrieved_total = 0

    s1_zero_candidates = 0
    bad_prefix_count = 0
    s1_self_matches = 0
    duplicate_candidates = 0

    for _, row in s1_prep.iterrows():
        s1_id = row["entity_id"]
        
        # Source 2 candidates
        c2 = exact_name_candidates(row, s2_exact_index) | token_candidates(
            row, s2_token_index, s2_token_frequency, max_tokens=MAX_TOKENS, max_frequency=MAX_FREQUENCY
        )
        # Source 3 candidates
        c3 = exact_name_candidates(row, s3_exact_index) | token_candidates(
            row, s3_token_index, s3_token_frequency, max_tokens=MAX_TOKENS, max_frequency=MAX_FREQUENCY
        )

        # Integrity checks
        for cid in c2:
            if not cid.startswith("S2-"):
                bad_prefix_count += 1
            if cid.startswith("S1-"):
                s1_self_matches += 1
        for cid in c3:
            if not cid.startswith("S3-"):
                bad_prefix_count += 1
            if cid.startswith("S1-"):
                s1_self_matches += 1

        c_combined = c2 | c3
        n_c2 = len(c2)
        n_c3 = len(c3)
        n_tot = len(c_combined)

        c_counts_s2.append(n_c2)
        c_counts_s3.append(n_c3)
        c_counts_total.append(n_tot)

        if n_tot == 0:
            s1_zero_candidates += 1

        # Ground truth retrieval calculation
        true_mids = sample_gt.get(s1_id, set())
        true_s2 = {m for m in true_mids if m.startswith("S2-")}
        true_s3 = {m for m in true_mids if m.startswith("S3-")}

        retrieved_s2 += len(true_s2 & c2)
        retrieved_s3 += len(true_s3 & c3)
        retrieved_total += len(true_mids & c_combined)

    t_gen = time.perf_counter() - t_gen_start
    t_total = time.perf_counter() - t_start

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 6. Metrics and Statistics
    s_tot = pd.Series(c_counts_total)
    s_s2 = pd.Series(c_counts_s2)
    s_s3 = pd.Series(c_counts_s3)

    # Denominator: verified ground truth matches present in the source datasets
    denom_s2 = len(s2_found_targets)
    denom_s3 = len(s3_found_targets)
    denom_total = total_true_links

    # Recall calculations
    recall_s2 = (retrieved_s2 / total_s2_links) if total_s2_links > 0 else 0.0
    recall_s3 = (retrieved_s3 / total_s3_links) if total_s3_links > 0 else 0.0
    recall_total = (retrieved_total / total_true_links) if total_true_links > 0 else 0.0

    print("\n" + "=" * 80)
    print("RECALL & CANDIDATE EVALUATION RESULTS")
    print("=" * 80)
    print(f"S1 Records Evaluated:                 {N_S1:,}")
    print(f"S1 Records with Zero Candidates:      {s1_zero_candidates:,} ({s1_zero_candidates / N_S1 * 100:.2f}%)")
    print(f"Total True Links in Sample:           {total_true_links:,}")
    print(f"Retrieved True Links:                 {retrieved_total:,}")
    print(f"\nOverall Candidate Recall:             {recall_total:.4f} ({recall_total * 100:.2f}%)")
    print(f"  - Numerator:                        {retrieved_total:,} retrieved true matches")
    print(f"  - Denominator:                      {total_true_links:,} total true matches in sample")
    print(f"\nSource 2 Candidate Recall:            {recall_s2:.4f} ({recall_s2 * 100:.2f}%) [{retrieved_s2:,} / {total_s2_links:,}]")
    print(f"Source 3 Candidate Recall:            {recall_s3:.4f} ({recall_s3 * 100:.2f}%) [{retrieved_s3:,} / {total_s3_links:,}]")

    print("\nCandidate Volume per S1 Record (Combined):")
    print(f"  Mean:   {s_tot.mean():.2f}")
    print(f"  Median: {s_tot.median():.2f}")
    print(f"  p90:    {s_tot.quantile(0.90):.2f}")
    print(f"  p95:    {s_tot.quantile(0.95):.2f}")
    print(f"  p99:    {s_tot.quantile(0.99):.2f}")
    print(f"  Max:    {s_tot.max():,}")

    print("\nCandidate Volume per S1 Record (Source 2):")
    print(f"  Mean: {s_s2.mean():.2f} | Median: {s_s2.median():.2f} | p90: {s_s2.quantile(0.90):.2f} | p95: {s_s2.quantile(0.95):.2f} | p99: {s_s2.quantile(0.99):.2f} | Max: {s_s2.max():,}")

    print("\nCandidate Volume per S1 Record (Source 3):")
    print(f"  Mean: {s_s3.mean():.2f} | Median: {s_s3.median():.2f} | p90: {s_s3.quantile(0.90):.2f} | p95: {s_s3.quantile(0.95):.2f} | p99: {s_s3.quantile(0.99):.2f} | Max: {s_s3.max():,}")

    print("\nCorrectness & Integrity:")
    print(f"  Invalid source prefix count:        {bad_prefix_count}")
    print(f"  S1 self-matches in candidate sets:  {s1_self_matches}")
    print(f"  Duplicate candidate IDs inside set: {duplicate_candidates}")

    print("\nRuntime & Resource Profiling (Measured):")
    print(f"  Total Elapsed Time:                 {t_total:.2f} s")
    print(f"  Candidate Generation Time:          {t_gen:.2f} s ({t_gen / N_S1 * 1000:.2f} ms/entity)")
    print(f"  Peak Memory (tracemalloc):          {peak_mem / (1024 * 1024):.2f} MB")

    # 7. Estimates for full test set (labeled as estimates)
    test_s1_records = 1_732_543
    est_candidates_per_entity = s_tot.mean()
    est_total_candidate_pairs = int(test_s1_records * est_candidates_per_entity)
    # Approx 13 bytes per candidate ID ('S2-XXXXXXXX,')
    est_tsv_bytes = est_total_candidate_pairs * 13 + test_s1_records * 20
    est_tsv_gb = est_tsv_bytes / (1024 ** 3)

    print("\nExtrapolations for Full Test Set (ESTIMATES):")
    print(f"  Test S1 Records:                    {test_s1_records:,}")
    print(f"  Estimated Mean Candidates/S1:       {est_candidates_per_entity:.2f}")
    print(f"  Estimated Total Candidate Pairs:    ~{est_total_candidate_pairs:,} pairs (~{est_total_candidate_pairs / 1e9:.2f} Billion)")
    print(f"  Estimated candidate_pairs.tsv Size: ~{est_tsv_gb:.2f} GB")

    # Save machine-readable summary JSON under output/
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "blocking_v2_evaluation_summary.json"
    summary_data = {
        "evaluation_parameters": {
            "s1_sample_size": N_S1,
            "random_seed": RANDOM_SEED,
            "non_matching_target_pool_size": NON_MATCHING_POOL_SIZE,
            "max_tokens": MAX_TOKENS,
            "max_frequency": MAX_FREQUENCY,
        },
        "measured_metrics": {
            "s1_records_evaluated": N_S1,
            "singletons_count": singletons,
            "singletons_percentage": singletons / N_S1 * 100,
            "s1_zero_candidates_count": s1_zero_candidates,
            "s1_zero_candidates_percentage": s1_zero_candidates / N_S1 * 100,
            "total_ground_truth_links": total_true_links,
            "retrieved_ground_truth_links": retrieved_total,
            "candidate_recall_total": recall_total,
            "candidate_recall_s2": recall_s2,
            "candidate_recall_s3": recall_s3,
            "candidate_counts_total": {
                "mean": float(s_tot.mean()),
                "median": float(s_tot.median()),
                "p90": float(s_tot.quantile(0.90)),
                "p95": float(s_tot.quantile(0.95)),
                "p99": float(s_tot.quantile(0.99)),
                "max": int(s_tot.max()),
            },
            "candidate_counts_s2": {
                "mean": float(s_s2.mean()),
                "median": float(s_s2.median()),
                "p90": float(s_s2.quantile(0.90)),
                "p95": float(s_s2.quantile(0.95)),
                "p99": float(s_s2.quantile(0.99)),
                "max": int(s_s2.max()),
            },
            "candidate_counts_s3": {
                "mean": float(s_s3.mean()),
                "median": float(s_s3.median()),
                "p90": float(s_s3.quantile(0.90)),
                "p95": float(s_s3.quantile(0.95)),
                "p99": float(s_s3.quantile(0.99)),
                "max": int(s_s3.max()),
            },
            "integrity_checks": {
                "bad_prefix_count": bad_prefix_count,
                "s1_self_matches": s1_self_matches,
                "duplicate_candidates": duplicate_candidates,
                "missing_s2_targets_in_dataset": len(s2_missing),
                "missing_s3_targets_in_dataset": len(s3_missing),
            },
            "profiling": {
                "total_elapsed_seconds": t_total,
                "generation_elapsed_seconds": t_gen,
                "peak_memory_mb": peak_mem / (1024 * 1024),
            },
        },
        "extrapolations_estimates": {
            "test_s1_records": test_s1_records,
            "estimated_candidate_pairs": est_total_candidate_pairs,
            "estimated_candidate_pairs_tsv_gb": est_tsv_gb,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nMachine-readable evaluation summary saved to: {summary_path}")


if __name__ == "__main__":
    evaluate_blocking_v2()
