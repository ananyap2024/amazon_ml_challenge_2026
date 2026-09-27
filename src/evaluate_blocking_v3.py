"""
ML Challenge 2026 — Head-to-Head Evaluation: Blocking V2 vs Blocking V3

Runs a controlled, reproducible head-to-head comparison on the identical 2,000 S1 sample
(random_state=42) and ground-truth-complete target pool.

Evaluates channels independently and in combination:
- V2 Baseline (Exact Name + Name Tokens with max_freq=5000)
- Channel Ablation:
    * Exact Name Only
    * Domain-Cleaned Condensed Name Only
    * Name Tokens (Tightened max_freq=1500) Only
    * Address Tokens (max_addr_freq=250) Only
- Multi-Channel V3 Full Combination

Reports link recovery, candidate volume reductions, runtime, Python tracemalloc memory,
and Win32 OS Process Peak Working Set RAM.
"""

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import time
import tracemalloc
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import V2 baseline functions
from blocking_v2 import (
    prepare_dataframe as prepare_v2,
    build_exact_name_index as build_v2_exact,
    build_name_token_index as build_v2_token,
    build_token_frequency as build_v2_freq,
    exact_name_candidates as v2_exact_candidates,
    token_candidates as v2_token_candidates,
)

# Import V3 enhanced functions
from blocking_v3 import (
    prepare_dataframe_v3,
    build_exact_name_index as build_v3_exact,
    build_condensed_name_index,
    build_inverted_index,
    build_token_frequency as build_v3_freq,
    lookup_channel_candidates,
)

TRAIN_DIR = PROJECT_DIR / "dataset" / "train"
OUTPUT_DIR = PROJECT_DIR / "output"


# Win32 Process Memory Tracking
class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def get_process_memory_mb():
    try:
        GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
        GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
            wintypes.DWORD,
        ]
        GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        hProcess = ctypes.windll.kernel32.GetCurrentProcess()
        if GetProcessMemoryInfo(hProcess, ctypes.byref(counters), counters.cb):
            return {
                "working_set_mb": counters.WorkingSetSize / (1024 * 1024),
                "peak_working_set_mb": counters.PeakWorkingSetSize / (1024 * 1024),
            }
    except Exception:
        pass
    return {"working_set_mb": 0.0, "peak_working_set_mb": 0.0}


def load_identical_sample_and_pool(n_s1=2000, random_seed=42, non_matching_limit=100000):
    print("Loading ground truth...")
    gt_df = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str)
    gt_map = {}
    for s1_id, matched in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"].fillna("")):
        s1_id_clean = s1_id.strip()
        mids = {m.strip() for m in matched.split(",") if m.strip()} if matched.strip() else set()
        gt_map[s1_id_clean] = mids
    del gt_df

    print(f"Sampling {n_s1} S1 records with seed={random_seed}...")
    s1_full = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype=str,
    )
    s1_sample = s1_full.sample(n=n_s1, random_state=random_seed).copy()
    s1_sample["entity_id"] = s1_sample["entity_id"].str.strip()
    s1_sample["business_address"] = s1_sample["business_address"].fillna("")
    del s1_full

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

    def extract_pool(source_path, target_ids):
        print(f"Streaming target pool from {source_path.name}...")
        matched_chunks = []
        non_match_chunks = []
        found_target_ids = set()
        chunk_size = 200_000
        for chunk in pd.read_csv(
            source_path,
            sep="\t",
            usecols=["entity_id", "business_name", "business_address", "country"],
            dtype=str,
            chunksize=chunk_size,
        ):
            chunk["entity_id"] = chunk["entity_id"].str.strip()
            chunk["business_name"] = chunk["business_name"].fillna("")
            chunk["business_address"] = chunk["business_address"].fillna("")

            is_target = chunk["entity_id"].isin(target_ids)
            if is_target.any():
                match = chunk[is_target]
                matched_chunks.append(match)
                found_target_ids.update(match["entity_id"])

            non_match = chunk[~is_target]
            sample_count = min(
                len(non_match), int(non_matching_limit * (len(chunk) / 5_000_000) * 1.2)
            )
            non_match_chunks.append(
                non_match.sample(n=sample_count, random_state=random_seed)
            )

        matched_df = pd.concat(matched_chunks, ignore_index=True) if matched_chunks else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        non_matched_df = pd.concat(non_match_chunks, ignore_index=True) if non_match_chunks else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
        if len(non_matched_df) > non_matching_limit:
            non_matched_df = non_matched_df.sample(n=non_matching_limit, random_state=random_seed)

        pool_df = pd.concat([matched_df, non_matched_df], ignore_index=True).drop_duplicates(subset=["entity_id"])
        missing_ids = target_ids - found_target_ids
        return pool_df, found_target_ids, missing_ids

    s2_pool, s2_found, s2_missing = extract_pool(TRAIN_DIR / "train_source2.tsv", target_s2_ids)
    s3_pool, s3_found, s3_missing = extract_pool(TRAIN_DIR / "train_source3.tsv", target_s3_ids)

    return s1_sample, sample_gt, s2_pool, s3_pool, target_s2_ids, target_s3_ids, s2_missing, s3_missing


def run_experiment():
    tracemalloc.start()
    t_start = time.perf_counter()

    print("=" * 80)
    print("CONTROLLED HEAD-TO-HEAD EXPERIMENT: BLOCKING V2 vs BLOCKING V3")
    print("=" * 80)

    # 1. Load data
    s1_df, sample_gt, s2_df, s3_df, target_s2, target_s3, s2_miss, s3_miss = load_identical_sample_and_pool()
    total_true_links = sum(len(mids) for mids in sample_gt.values())
    total_s2_links = sum(len([m for m in mids if m.startswith("S2-")]) for mids in sample_gt.values())
    total_s3_links = sum(len([m for m in mids if m.startswith("S3-")]) for mids in sample_gt.values())

    print(f"\nEvaluated S1 records:                 {len(s1_df):,}")
    print(f"Total ground-truth links:             {total_true_links:,} (S2: {total_s2_links:,}, S3: {total_s3_links:,})")
    print(f"Missing GT target IDs in S2 source:   {len(s2_miss)}")
    print(f"Missing GT target IDs in S3 source:   {len(s3_miss)}")

    # 2. Build V2 Baseline
    print("\n--- Running V2 Baseline Evaluation ---")
    t0 = time.perf_counter()
    s1_v2 = prepare_v2(s1_df)
    s2_v2 = prepare_v2(s2_df)
    s3_v2 = prepare_v2(s3_df)

    s2_v2_exact = build_v2_exact(s2_v2)
    s3_v2_exact = build_v2_exact(s3_v2)
    s2_v2_token = build_v2_token(s2_v2)
    s3_v2_token = build_v2_token(s3_v2)
    s2_v2_freq = build_v2_freq(s2_v2_token)
    s3_v2_freq = build_v2_freq(s3_v2_token)

    v2_candidates_per_s1 = {}
    v2_retrieved_total = set()  # set of (s1_id, matched_id) pairs
    v2_counts = []

    for _, row in s1_v2.iterrows():
        s1_id = row["entity_id"]
        c2 = v2_exact_candidates(row, s2_v2_exact) | v2_token_candidates(row, s2_v2_token, s2_v2_freq, max_tokens=3, max_frequency=5000)
        c3 = v2_exact_candidates(row, s3_v2_exact) | v2_token_candidates(row, s3_v2_token, s3_v2_freq, max_tokens=3, max_frequency=5000)
        c_tot = c2 | c3
        v2_candidates_per_s1[s1_id] = c_tot
        v2_counts.append(len(c_tot))

        for mid in sample_gt.get(s1_id, set()):
            if mid in c_tot:
                v2_retrieved_total.add((s1_id, mid))

    t_v2 = time.perf_counter() - t0
    v2_recall = len(v2_retrieved_total) / total_true_links
    s_v2 = pd.Series(v2_counts)

    print(f"V2 Baseline Recall:                   {v2_recall:.4f} ({v2_recall * 100:.2f}%) [{len(v2_retrieved_total):,} / {total_true_links:,}]")
    print(f"V2 Baseline Mean Candidates:          {s_v2.mean():.2f} (Median: {s_v2.median():.2f}, Max: {s_v2.max():,})")

    # 3. Build V3 Multi-Channel Data and Indexes
    print("\n--- Building V3 Multi-Channel Indexes ---")
    t0 = time.perf_counter()
    s1_v3 = prepare_dataframe_v3(s1_df)
    s2_v3 = prepare_dataframe_v3(s2_df)
    s3_v3 = prepare_dataframe_v3(s3_df)

    # Indexes
    s2_v3_exact = build_v3_exact(s2_v3)
    s3_v3_exact = build_v3_exact(s3_v3)

    s2_condensed = build_condensed_name_index(s2_v3)
    s3_condensed = build_condensed_name_index(s3_v3)

    s2_name_tok = build_inverted_index(s2_v3, "name_tokens")
    s3_name_tok = build_inverted_index(s3_v3, "name_tokens")
    s2_name_freq = build_v3_freq(s2_name_tok)
    s3_name_freq = build_v3_freq(s3_name_tok)

    s2_addr_tok = build_inverted_index(s2_v3, "address_tokens")
    s3_addr_tok = build_inverted_index(s3_v3, "address_tokens")
    s2_addr_freq = build_v3_freq(s2_addr_tok)
    s3_addr_freq = build_v3_freq(s3_addr_tok)
    t_v3_prep = time.perf_counter() - t0
    print(f"V3 preparation & indexing complete in {t_v3_prep:.2f}s.")

    # 4. Channel Ablation and Link Recovery Tracking
    print("\n--- Evaluating Independent Channels & Full V3 Combination ---")
    # Configurations to test:
    # 1. Exact Name Only
    # 2. Domain Condensed Only
    # 3. Name Tokens Only (Tightened max_freq=1500)
    # 4. Address Tokens Only (max_addr_freq=250)
    # 5. V3 Full Combination (Exact + Domain + NameTokens[1500] + Address[250])

    all_gt_pairs = set()
    for s1_id, mids in sample_gt.items():
        for mid in mids:
            all_gt_pairs.add((s1_id, mid))

    v2_missed_pairs = all_gt_pairs - v2_retrieved_total

    results_by_config = {}
    configs = {
        "Channel: Exact Name Only": dict(use_exact=True, use_condensed=False, use_name_tokens=False, use_address=False),
        "Channel: Domain Condensed Only": dict(use_exact=False, use_condensed=True, use_name_tokens=False, use_address=False),
        "Channel: Name Tokens (1500 cap)": dict(use_exact=False, use_condensed=False, use_name_tokens=True, use_address=False, max_name_tokens=3, max_name_freq=1500),
        "Channel: Address Tokens (250 cap)": dict(use_exact=False, use_condensed=False, use_name_tokens=False, use_address=True, max_addr_tokens=2, max_addr_freq=250),
        "Combination: Exact + Domain + NameTokens(1500)": dict(use_exact=True, use_condensed=True, use_name_tokens=True, use_address=False, max_name_tokens=3, max_name_freq=1500),
        "Full V3: Exact + Domain + NameTokens(1500) + Address(250)": dict(use_exact=True, use_condensed=True, use_name_tokens=True, use_address=True, max_name_tokens=3, max_name_freq=1500, max_addr_tokens=2, max_addr_freq=250),
        "Tighter V3: Exact + Domain + NameTokens(1000) + Address(200)": dict(use_exact=True, use_condensed=True, use_name_tokens=True, use_address=True, max_name_tokens=2, max_name_freq=1000, max_addr_tokens=2, max_addr_freq=200),
    }

    for config_name, cfg in configs.items():
        retrieved_pairs = set()
        retrieved_s2 = 0
        retrieved_s3 = 0
        counts_total = []
        counts_s2 = []
        counts_s3 = []
        s1_zeros = 0

        bad_prefixes = 0
        self_matches = 0

        t_cfg_start = time.perf_counter()
        for _, row in s1_v3.iterrows():
            s1_id = row["entity_id"]
            c2, _ = lookup_channel_candidates(
                row, s2_v3_exact, s2_condensed, s2_name_tok, s2_name_freq, s2_addr_tok, s2_addr_freq, **cfg
            )
            c3, _ = lookup_channel_candidates(
                row, s3_v3_exact, s3_condensed, s3_name_tok, s3_name_freq, s3_addr_tok, s3_addr_freq, **cfg
            )

            # Checks
            for cid in c2:
                if not cid.startswith("S2-"):
                    bad_prefixes += 1
                if cid.startswith("S1-"):
                    self_matches += 1
            for cid in c3:
                if not cid.startswith("S3-"):
                    bad_prefixes += 1
                if cid.startswith("S1-"):
                    self_matches += 1

            c_tot = c2 | c3
            n2 = len(c2)
            n3 = len(c3)
            ntot = len(c_tot)

            counts_s2.append(n2)
            counts_s3.append(n3)
            counts_total.append(ntot)
            if ntot == 0:
                s1_zeros += 1

            true_mids = sample_gt.get(s1_id, set())
            for mid in true_mids:
                if mid in c_tot:
                    retrieved_pairs.add((s1_id, mid))
                    if mid.startswith("S2-"):
                        retrieved_s2 += 1
                    elif mid.startswith("S3-"):
                        retrieved_s3 += 1

        t_cfg_duration = time.perf_counter() - t_cfg_start
        s_series = pd.Series(counts_total)
        s_series_s2 = pd.Series(counts_s2)
        s_series_s3 = pd.Series(counts_s3)

        recall_tot = len(retrieved_pairs) / total_true_links
        recall_s2 = retrieved_s2 / total_s2_links if total_s2_links > 0 else 0.0
        recall_s3 = retrieved_s3 / total_s3_links if total_s3_links > 0 else 0.0

        recovered_from_v2_missed = len(retrieved_pairs & v2_missed_pairs)

        results_by_config[config_name] = {
            "recall_total": recall_tot,
            "recall_s2": recall_s2,
            "recall_s3": recall_s3,
            "retrieved_total": len(retrieved_pairs),
            "recovered_v2_missed": recovered_from_v2_missed,
            "mean_candidates": float(s_series.mean()),
            "median_candidates": float(s_series.median()),
            "p90_candidates": float(s_series.quantile(0.90)),
            "p95_candidates": float(s_series.quantile(0.95)),
            "p99_candidates": float(s_series.quantile(0.99)),
            "max_candidates": int(s_series.max()),
            "zero_candidates_pct": (s1_zeros / len(s1_df)) * 100,
            "total_candidate_pairs_in_sample": int(s_series.sum()),
            "s2_mean": float(s_series_s2.mean()),
            "s3_mean": float(s_series_s3.mean()),
            "duration_s": t_cfg_duration,
            "bad_prefixes": bad_prefixes,
            "self_matches": self_matches,
        }

    t_total = time.perf_counter() - t_start
    current_mem, peak_traced_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    proc_mem = get_process_memory_mb()

    # Print Full Comparative Table
    print("\n" + "=" * 95)
    print("HEAD-TO-HEAD EXPERIMENTAL RESULTS")
    print("=" * 95)
    print(f"{'Configuration':<45} | {'Recall':<8} | {'S2 Rec':<8} | {'S3 Rec':<8} | {'Recovered':<10} | {'Mean Cand':<10} | {'Max Cand':<9}")
    print("-" * 108)

    # Print V2 Baseline first
    print(f"{'V2 Baseline (max_freq=5000)':<45} | {v2_recall*100:>6.2f}% | {2667/3279*100:>6.2f}% | {3152/3645*100:>6.2f}% | {'0 (ref)':<10} | {s_v2.mean():>10.2f} | {s_v2.max():>9,}")

    for cname, res in results_by_config.items():
        print(f"{cname:<45} | {res['recall_total']*100:>6.2f}% | {res['recall_s2']*100:>6.2f}% | {res['recall_s3']*100:>6.2f}% | {res['recovered_v2_missed']:>10,} | {res['mean_candidates']:>10.2f} | {res['max_candidates']:>9,}")

    print("\n" + "=" * 95)
    print("RESOURCE & MEMORY MEASUREMENTS")
    print("=" * 95)
    print(f"Total Elapsed Time:                   {t_total:.2f} s")
    print(f"Python Traced Peak Memory:            {peak_traced_mem / (1024 * 1024):.2f} MB (internal Python objects)")
    print(f"OS Process Working Set Memory:        {proc_mem['working_set_mb']:.2f} MB (actual physical RAM used by python.exe)")
    print(f"OS Process Peak Working Set Memory:   {proc_mem['peak_working_set_mb']:.2f} MB (peak physical RAM consumed)")

    # Save machine-readable output under output/
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "blocking_v3_evaluation_summary.json"
    summary_data = {
        "evaluation_parameters": {
            "n_s1": len(s1_df),
            "random_seed": 42,
            "total_true_links": total_true_links,
            "total_s2_links": total_s2_links,
            "total_s3_links": total_s3_links,
        },
        "v2_baseline": {
            "recall_total": v2_recall,
            "retrieved_total": len(v2_retrieved_total),
            "mean_candidates": float(s_v2.mean()),
            "median_candidates": float(s_v2.median()),
            "p90_candidates": float(s_v2.quantile(0.90)),
            "p95_candidates": float(s_v2.quantile(0.95)),
            "p99_candidates": float(s_v2.quantile(0.99)),
            "max_candidates": int(s_v2.max()),
        },
        "v3_ablation_and_combinations": results_by_config,
        "resource_profiling": {
            "total_elapsed_seconds": t_total,
            "python_traced_peak_mb": peak_traced_mem / (1024 * 1024),
            "os_process_working_set_mb": proc_mem["working_set_mb"],
            "os_process_peak_working_set_mb": proc_mem["peak_working_set_mb"],
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nMachine-readable comparison summary saved to: {summary_path}")


if __name__ == "__main__":
    run_experiment()
