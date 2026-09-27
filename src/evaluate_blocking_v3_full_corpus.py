"""
ML Challenge 2026 — Full Corpus Evaluation (Phase 2): Blocking V2 vs V3

Evaluates 10,000 reproducible S1 records against the COMPLETE S2 and S3 target datasets
(~10.3 million records) using source-sequential, compact array-backed inverted indexing.

Compares:
A. V2 Baseline (exact normalized name + name tokens with cap 5000, max_tokens=3)
B. Full V3 (exact name + domain-cleaned name + name tokens cap 1500 + address tokens cap 250)
C. Tighter V3 (exact name + domain-cleaned name + name tokens cap 1000 + address tokens cap 200)

Preserves exact record-frequency semantics and sequential memory release.
"""

import array
import ctypes
from ctypes import wintypes
import gc
import json
import os
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Dict, List, Set, Tuple
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Original V2 implementation
from blocking_v2 import (
    prepare_dataframe as prepare_v2,
    build_exact_name_index as build_v2_exact,
    build_name_token_index as build_v2_token,
    build_token_frequency as build_v2_freq,
    exact_name_candidates as v2_exact_candidates,
    token_candidates as v2_token_candidates,
)

# Original V3 implementation
from blocking_v3 import (
    clean_domain_name,
    extract_condensed_name,
    tokenize_name_v3,
    tokenize_address_v3,
    prepare_dataframe_v3,
    build_exact_name_index as build_v3_exact,
    build_condensed_name_index as build_v3_condensed,
    build_inverted_index as build_v3_inverted,
    build_token_frequency as build_v3_freq,
    lookup_channel_candidates,
    select_distinctive_tokens,
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


# ==============================================================================
# Compact Inverted Index Data Structures
# ==============================================================================

class CompactCorpusIndex:
    """
    Compact memory-safe index for a single complete target source (S2 or S3).
    Stores posting lists as array.array('I') to avoid 80-byte Python string set overhead.
    """
    def __init__(self, source_prefix: str):
        self.source_prefix = source_prefix
        self.id_list: List[str] = []
        self.exact_name_index: Dict[str, array.array] = {}
        self.condensed_name_index: Dict[str, array.array] = {}
        self.name_token_index: Dict[str, array.array] = {}
        self.address_token_index: Dict[str, array.array] = {}
        self.name_token_frequency: Dict[str, int] = {}
        self.address_token_frequency: Dict[str, int] = {}

    def get_candidates(
        self,
        row: pd.Series,
        config: dict
    ) -> Set[str]:
        candidates: Set[str] = set()

        # 1. Exact Name
        if config.get("use_exact", True) and row.get("name_norm"):
            postings = self.exact_name_index.get(row["name_norm"])
            if postings:
                for idx in postings:
                    candidates.add(self.id_list[idx])

        # 2. Condensed Domain Name
        if config.get("use_condensed", False) and row.get("name_condensed") and len(row["name_condensed"]) >= 4:
            postings = self.condensed_name_index.get(row["name_condensed"])
            if postings:
                for idx in postings:
                    candidates.add(self.id_list[idx])

        # 3. Name Tokens
        if config.get("use_name_tokens", True) and row.get("name_tokens"):
            max_tokens = config.get("max_name_tokens", 3)
            max_freq = config.get("max_name_freq", 5000)
            selected = select_distinctive_tokens(
                row["name_tokens"],
                self.name_token_frequency,
                max_tokens=max_tokens,
                max_frequency=max_freq
            )
            for token in selected:
                postings = self.name_token_index.get(token)
                if postings:
                    for idx in postings:
                        candidates.add(self.id_list[idx])

        # 4. Address Tokens
        if config.get("use_address", False) and row.get("address_tokens"):
            max_addr_tokens = config.get("max_addr_tokens", 2)
            max_addr_freq = config.get("max_addr_freq", 250)
            selected_addr = select_distinctive_tokens(
                row["address_tokens"],
                self.address_token_frequency,
                max_tokens=max_addr_tokens,
                max_frequency=max_addr_freq
            )
            for token in selected_addr:
                postings = self.address_token_index.get(token)
                if postings:
                    for idx in postings:
                        candidates.add(self.id_list[idx])

        return candidates


def run_equivalence_test():
    """
    Rigorous test verifying that CompactCorpusIndex produces 100% identical candidate sets
    to the original blocking_v2 and blocking_v3 implementations on a real dataset fixture.
    """
    print("=" * 80)
    print("RUNNING END-TO-END EQUIVALENCE TEST (Original vs. Compact Implementation)")
    print("=" * 80)

    # Load 500 S1 records and 2,000 S2/S3 records
    s1_fixture = pd.read_csv(TRAIN_DIR / "train_source1.tsv", sep="\t", nrows=500, dtype=str)
    s2_fixture = pd.read_csv(TRAIN_DIR / "train_source2.tsv", sep="\t", nrows=2000, dtype=str)

    s1_fixture["business_name"] = s1_fixture["business_name"].fillna("")
    s1_fixture["business_address"] = s1_fixture["business_address"].fillna("")
    s2_fixture["business_name"] = s2_fixture["business_name"].fillna("")
    s2_fixture["business_address"] = s2_fixture["business_address"].fillna("")

    # 1. Test V2 Equivalence
    print("Testing V2 equivalence on fixture...")
    s1_v2 = prepare_v2(s1_fixture)
    s2_v2 = prepare_v2(s2_fixture)

    orig_exact = build_v2_exact(s2_v2)
    orig_token = build_v2_token(s2_v2)
    orig_freq = build_v2_freq(orig_token)

    # Build compact V2 index
    compact_v2 = CompactCorpusIndex("S2-")
    compact_v2.id_list = s2_v2["entity_id"].tolist()
    compact_v2.name_token_frequency = orig_freq

    for i, row in s2_v2.iterrows():
        nm = row["name_norm"]
        if nm:
            compact_v2.exact_name_index.setdefault(nm, array.array("I")).append(i)
        for tok in row["name_tokens"]:
            compact_v2.name_token_index.setdefault(tok, array.array("I")).append(i)

    v2_config = {"use_exact": True, "use_condensed": False, "use_name_tokens": True, "use_address": False, "max_name_tokens": 3, "max_name_freq": 5000}

    for _, row in s1_v2.iterrows():
        orig_cand = v2_exact_candidates(row, orig_exact) | v2_token_candidates(row, orig_token, orig_freq, max_tokens=3, max_frequency=5000)
        compact_cand = compact_v2.get_candidates(row, v2_config)
        assert orig_cand == compact_cand, f"V2 mismatch for entity {row['entity_id']}: orig={len(orig_cand)}, compact={len(compact_cand)}"

    print("  [PASS] V2 candidate generation equivalence: 100% match across 500 entities.")

    # 2. Test V3 Equivalence
    print("Testing V3 equivalence on fixture...")
    s1_v3 = prepare_dataframe_v3(s1_fixture)
    s2_v3 = prepare_dataframe_v3(s2_fixture)

    orig_v3_exact = build_v3_exact(s2_v3)
    orig_v3_condensed = build_v3_condensed(s2_v3)
    orig_v3_name_tok = build_v3_inverted(s2_v3, "name_tokens")
    orig_v3_addr_tok = build_v3_inverted(s2_v3, "address_tokens")
    orig_v3_name_freq = build_v3_freq(orig_v3_name_tok)
    orig_v3_addr_freq = build_v3_freq(orig_v3_addr_tok)

    compact_v3 = CompactCorpusIndex("S2-")
    compact_v3.id_list = s2_v3["entity_id"].tolist()
    compact_v3.name_token_frequency = orig_v3_name_freq
    compact_v3.address_token_frequency = orig_v3_addr_freq

    for i, row in s2_v3.iterrows():
        nm = clean_domain_name(row["business_name"])
        if nm:
            compact_v3.exact_name_index.setdefault(nm, array.array("I")).append(i)
        cn = row["name_condensed"]
        if cn and len(cn) >= 4:
            compact_v3.condensed_name_index.setdefault(cn, array.array("I")).append(i)
        for tok in row["name_tokens"]:
            compact_v3.name_token_index.setdefault(tok, array.array("I")).append(i)
        for tok in row["address_tokens"]:
            compact_v3.address_token_index.setdefault(tok, array.array("I")).append(i)

    v3_config = {"use_exact": True, "use_condensed": True, "use_name_tokens": True, "use_address": True, "max_name_tokens": 3, "max_name_freq": 1500, "max_addr_tokens": 2, "max_addr_freq": 250}

    for _, row in s1_v3.iterrows():
        orig_cand, _ = lookup_channel_candidates(row, orig_v3_exact, orig_v3_condensed, orig_v3_name_tok, orig_v3_name_freq, orig_v3_addr_tok, orig_v3_addr_freq, **v3_config)
        compact_cand = compact_v3.get_candidates(row, v3_config)
        assert orig_cand == compact_cand, f"V3 mismatch for entity {row['entity_id']}: orig={len(orig_cand)}, compact={len(compact_cand)}"

    print("  [PASS] V3 candidate generation equivalence: 100% match across 500 entities.")
    print("=" * 80 + "\n")


def run_full_evaluation():

    tracemalloc.start()
    t_start = time.perf_counter()

    init_mem = get_process_memory_mb()
    print("=" * 90)
    print("PHASE 2: FULL-CORPUS EVALUATION (10,000 S1 RECORDS vs COMPLETE S2 & S3)")
    print("=" * 90)
    print(f"Initial Evaluator Working Set:        {init_mem['working_set_mb']:.2f} MB")

    N_S1 = 10_000
    RANDOM_SEED = 42
    CHUNK_SIZE = 200_000

    # 1. Load Ground Truth
    print(f"\n1. Loading Ground Truth from {TRAIN_DIR / 'train_ground_truth.tsv'}...")
    t0 = time.perf_counter()
    gt_df = pd.read_csv(TRAIN_DIR / "train_ground_truth.tsv", sep="\t", dtype=str)
    gt_map = {}
    for s1_id, matched in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"].fillna("")):
        s1_id_clean = s1_id.strip()
        mids = {m.strip() for m in matched.split(",") if m.strip()} if matched.strip() else set()
        gt_map[s1_id_clean] = mids
    del gt_df
    print(f"Ground truth loaded in {time.perf_counter() - t0:.2f}s ({len(gt_map):,} total records).")

    # 2. Sample 10,000 S1 Records
    print(f"\n2. Sampling {N_S1:,} reproducible S1 records (random_state={RANDOM_SEED})...")
    s1_full = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address", "country"],
        dtype=str
    )
    s1_sample = s1_full.sample(n=N_S1, random_state=RANDOM_SEED).copy()
    s1_sample["entity_id"] = s1_sample["entity_id"].str.strip()
    s1_sample["business_address"] = s1_sample["business_address"].fillna("")
    del s1_full

    # Map sample ground truth
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

    total_true_links = sum(len(mids) for mids in sample_gt.values())
    total_s2_links = sum(len([m for m in mids if m.startswith("S2-")]) for mids in sample_gt.values())
    total_s3_links = sum(len([m for m in mids if m.startswith("S3-")]) for mids in sample_gt.values())
    singletons = sum(1 for mids in sample_gt.values() if len(mids) == 0)

    print(f"Sampled S1 entities:                  {len(s1_sample):,}")
    print(f"Total ground-truth links in sample:    {total_true_links:,}")
    print(f"  - Source 2 target links:            {total_s2_links:,} ({len(target_s2_ids):,} unique IDs)")
    print(f"  - Source 3 target links:            {total_s3_links:,} ({len(target_s3_ids):,} unique IDs)")
    print(f"  - Singletons (no matches):          {singletons:,} ({singletons / len(s1_sample) * 100:.2f}%)")

    # Prepare S1 representations once
    print("Preparing S1 sample representations...")
    s1_prep = prepare_dataframe_v3(s1_sample)

    # Configurations to evaluate
    configurations = {
        "V2 Baseline": {
            "use_exact": True,
            "use_condensed": False,
            "use_name_tokens": True,
            "use_address": False,
            "max_name_tokens": 3,
            "max_name_freq": 5000,
        },
        "Full V3": {
            "use_exact": True,
            "use_condensed": True,
            "use_name_tokens": True,
            "use_address": True,
            "max_name_tokens": 3,
            "max_name_freq": 1500,
            "max_addr_tokens": 2,
            "max_addr_freq": 250,
        },
        "Tighter V3": {
            "use_exact": True,
            "use_condensed": True,
            "use_name_tokens": True,
            "use_address": True,
            "max_name_tokens": 2,
            "max_name_freq": 1000,
            "max_addr_tokens": 2,
            "max_addr_freq": 200,
        },
    }

    # Store candidates per configuration and per source
    candidates_s2 = {cfg: {} for cfg in configurations}
    candidates_s3 = {cfg: {} for cfg in configurations}

    # Verify target IDs present in corpus
    verified_s2_ids = set()
    verified_s3_ids = set()

    timing = {
        "s2_pass1": 0.0,
        "s2_pass2": 0.0,
        "s2_query": 0.0,
        "s3_pass1": 0.0,
        "s3_pass2": 0.0,
        "s3_query": 0.0,
    }

    # ==========================================================================
    # STEP 3: Process COMPLETE Source 2 Dataset (Sequential Partition 1)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE SOURCE 2 DATASET (5,034,615 records)")
    print("=" * 80)

    # S2 Pass 1: Global Frequency Counter
    print("S2 Pass 1: Streaming global token frequencies across complete file...")
    t0 = time.perf_counter()
    s2_name_freq: Dict[str, int] = {}
    s2_addr_freq: Dict[str, int] = {}
    s2_total_records = 0

    for chunk in pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address"],
        dtype=str,
        chunksize=CHUNK_SIZE
    ):
        eids = chunk["entity_id"].tolist()
        names = chunk["business_name"].fillna("").tolist()
        addrs = chunk["business_address"].fillna("").tolist()

        for eid, name, addr in zip(eids, names, addrs):
            if eid in target_s2_ids:
                verified_s2_ids.add(eid)
            for t in tokenize_name_v3(name):
                s2_name_freq[t] = s2_name_freq.get(t, 0) + 1
            for t in tokenize_address_v3(addr):
                s2_addr_freq[t] = s2_addr_freq.get(t, 0) + 1
        s2_total_records += len(chunk)

    timing["s2_pass1"] = time.perf_counter() - t0
    mem_pass1 = get_process_memory_mb()
    print(f"S2 Pass 1 complete in {timing['s2_pass1']:.2f}s ({s2_total_records:,} records).")
    print(f"  Unique name tokens: {len(s2_name_freq):,}, address tokens: {len(s2_addr_freq):,}")
    print(f"  Verified GT target IDs found: {len(verified_s2_ids):,} / {len(target_s2_ids):,}")
    print(f"  Current Process Working Set: {mem_pass1['working_set_mb']:.2f} MB")

    # S2 Pass 2: Compact Array Inverted Indexing
    print("\nS2 Pass 2: Building compact array index (pruning tokens with freq > 5000)...")
    t0 = time.perf_counter()
    compact_s2 = CompactCorpusIndex("S2-")
    compact_s2.name_token_frequency = s2_name_freq
    compact_s2.address_token_frequency = s2_addr_freq

    row_idx = 0
    for chunk in pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address"],
        dtype=str,
        chunksize=CHUNK_SIZE
    ):
        eids = chunk["entity_id"].tolist()
        names = chunk["business_name"].fillna("").tolist()
        addrs = chunk["business_address"].fillna("").tolist()

        for eid, name, addr in zip(eids, names, addrs):
            compact_s2.id_list.append(eid)
            # Exact
            nm = clean_domain_name(name)
            if nm:
                compact_s2.exact_name_index.setdefault(nm, array.array("I")).append(row_idx)
            # Condensed
            cn = extract_condensed_name(name)
            if cn and len(cn) >= 4:
                compact_s2.condensed_name_index.setdefault(cn, array.array("I")).append(row_idx)
            # Name tokens
            for t in tokenize_name_v3(name):
                if s2_name_freq.get(t, 0) <= 5000:
                    compact_s2.name_token_index.setdefault(t, array.array("I")).append(row_idx)
            # Address tokens
            for t in tokenize_address_v3(addr):
                if s2_addr_freq.get(t, 0) <= 250:
                    compact_s2.address_token_index.setdefault(t, array.array("I")).append(row_idx)
            row_idx += 1

    timing["s2_pass2"] = time.perf_counter() - t0
    mem_pass2 = get_process_memory_mb()
    print(f"S2 Pass 2 complete in {timing['s2_pass2']:.2f}s.")
    print(f"  Postings in Name Index: {sum(len(a) for a in compact_s2.name_token_index.values()):,}")
    print(f"  Postings in Address Index: {sum(len(a) for a in compact_s2.address_token_index.values()):,}")
    print(f"  Current Process Working Set: {mem_pass2['working_set_mb']:.2f} MB")

    # S2 Querying for 10k S1 records
    print("\nQuerying S2 for 10,000 S1 records across 3 configurations...")
    t0 = time.perf_counter()
    for cfg_name, cfg in configurations.items():
        for _, row in s1_prep.iterrows():
            s1_id = row["entity_id"]
            candidates_s2[cfg_name][s1_id] = compact_s2.get_candidates(row, cfg)
    timing["s2_query"] = time.perf_counter() - t0
    print(f"S2 querying complete in {timing['s2_query']:.2f}s.")

    # Releasing S2 memory
    print("Releasing Source 2 index memory...")
    del compact_s2
    del s2_name_freq
    del s2_addr_freq
    gc.collect()
    mem_after_s2 = get_process_memory_mb()
    print(f"Memory after S2 release: {mem_after_s2['working_set_mb']:.2f} MB")

    # ==========================================================================
    # STEP 4: Process COMPLETE Source 3 Dataset (Sequential Partition 2)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE SOURCE 3 DATASET (5,285,602 records)")
    print("=" * 80)

    # S3 Pass 1: Global Frequency Counter
    print("S3 Pass 1: Streaming global token frequencies across complete file...")
    t0 = time.perf_counter()
    s3_name_freq: Dict[str, int] = {}
    s3_addr_freq: Dict[str, int] = {}
    s3_total_records = 0

    for chunk in pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address"],
        dtype=str,
        chunksize=CHUNK_SIZE
    ):
        eids = chunk["entity_id"].tolist()
        names = chunk["business_name"].fillna("").tolist()
        addrs = chunk["business_address"].fillna("").tolist()

        for eid, name, addr in zip(eids, names, addrs):
            if eid in target_s3_ids:
                verified_s3_ids.add(eid)
            for t in tokenize_name_v3(name):
                s3_name_freq[t] = s3_name_freq.get(t, 0) + 1
            for t in tokenize_address_v3(addr):
                s3_addr_freq[t] = s3_addr_freq.get(t, 0) + 1
        s3_total_records += len(chunk)

    timing["s3_pass1"] = time.perf_counter() - t0
    mem_s3_pass1 = get_process_memory_mb()
    print(f"S3 Pass 1 complete in {timing['s3_pass1']:.2f}s ({s3_total_records:,} records).")
    print(f"  Unique name tokens: {len(s3_name_freq):,}, address tokens: {len(s3_addr_freq):,}")
    print(f"  Verified GT target IDs found: {len(verified_s3_ids):,} / {len(target_s3_ids):,}")
    print(f"  Current Process Working Set: {mem_s3_pass1['working_set_mb']:.2f} MB")

    # S3 Pass 2: Compact Array Inverted Indexing
    print("\nS3 Pass 2: Building compact array index (pruning tokens with freq > 5000)...")
    t0 = time.perf_counter()
    compact_s3 = CompactCorpusIndex("S3-")
    compact_s3.name_token_frequency = s3_name_freq
    compact_s3.address_token_frequency = s3_addr_freq

    row_idx = 0
    for chunk in pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t",
        usecols=["entity_id", "business_name", "business_address"],
        dtype=str,
        chunksize=CHUNK_SIZE
    ):
        eids = chunk["entity_id"].tolist()
        names = chunk["business_name"].fillna("").tolist()
        addrs = chunk["business_address"].fillna("").tolist()

        for eid, name, addr in zip(eids, names, addrs):
            compact_s3.id_list.append(eid)
            # Exact
            nm = clean_domain_name(name)
            if nm:
                compact_s3.exact_name_index.setdefault(nm, array.array("I")).append(row_idx)
            # Condensed
            cn = extract_condensed_name(name)
            if cn and len(cn) >= 4:
                compact_s3.condensed_name_index.setdefault(cn, array.array("I")).append(row_idx)
            # Name tokens
            for t in tokenize_name_v3(name):
                if s3_name_freq.get(t, 0) <= 5000:
                    compact_s3.name_token_index.setdefault(t, array.array("I")).append(row_idx)
            # Address tokens
            for t in tokenize_address_v3(addr):
                if s3_addr_freq.get(t, 0) <= 250:
                    compact_s3.address_token_index.setdefault(t, array.array("I")).append(row_idx)
            row_idx += 1

    timing["s3_pass2"] = time.perf_counter() - t0
    mem_s3_pass2 = get_process_memory_mb()
    print(f"S3 Pass 2 complete in {timing['s3_pass2']:.2f}s.")
    print(f"  Postings in Name Index: {sum(len(a) for a in compact_s3.name_token_index.values()):,}")
    print(f"  Postings in Address Index: {sum(len(a) for a in compact_s3.address_token_index.values()):,}")
    print(f"  Current Process Working Set: {mem_s3_pass2['working_set_mb']:.2f} MB")

    # S3 Querying for 10k S1 records
    print("\nQuerying S3 for 10,000 S1 records across 3 configurations...")
    t0 = time.perf_counter()
    for cfg_name, cfg in configurations.items():
        for _, row in s1_prep.iterrows():
            s1_id = row["entity_id"]
            candidates_s3[cfg_name][s1_id] = compact_s3.get_candidates(row, cfg)
    timing["s3_query"] = time.perf_counter() - t0
    print(f"S3 querying complete in {timing['s3_query']:.2f}s.")

    # Releasing S3 memory
    print("Releasing Source 3 index memory...")
    del compact_s3
    del s3_name_freq
    del s3_addr_freq
    gc.collect()

    # ==========================================================================
    # STEP 5: Metric Evaluation & Comparison
    # ==========================================================================
    print("\n" + "=" * 90)
    print("EVALUATING METRICS & COMBINED CANDIDATE POOLS (10,000 S1 RECORDS)")
    print("=" * 90)

    results = {}
    missing_s2_targets = target_s2_ids - verified_s2_ids
    missing_s3_targets = target_s3_ids - verified_s3_ids

    for cfg_name in configurations:
        retrieved_total = 0
        retrieved_s2 = 0
        retrieved_s3 = 0
        counts_total = []
        counts_s2 = []
        counts_s3 = []
        zero_candidates_count = 0

        bad_prefix_count = 0
        self_match_count = 0

        for s1_id in s1_sample["entity_id"]:
            c2 = candidates_s2[cfg_name].get(s1_id, set())
            c3 = candidates_s3[cfg_name].get(s1_id, set())

            # Integrity checks
            for cid in c2:
                if not cid.startswith("S2-"):
                    bad_prefix_count += 1
                if cid.startswith("S1-"):
                    self_match_count += 1
            for cid in c3:
                if not cid.startswith("S3-"):
                    bad_prefix_count += 1
                if cid.startswith("S1-"):
                    self_match_count += 1

            c_combined = c2 | c3
            n2 = len(c2)
            n3 = len(c3)
            ntot = len(c_combined)

            counts_s2.append(n2)
            counts_s3.append(n3)
            counts_total.append(ntot)

            if ntot == 0:
                zero_candidates_count += 1

            true_mids = sample_gt.get(s1_id, set())
            true_s2 = {m for m in true_mids if m.startswith("S2-")}
            true_s3 = {m for m in true_mids if m.startswith("S3-")}

            retrieved_s2 += len(true_s2 & c2)
            retrieved_s3 += len(true_s3 & c3)
            retrieved_total += len(true_mids & c_combined)

        s_tot = pd.Series(counts_total)
        s_s2 = pd.Series(counts_s2)
        s_s3 = pd.Series(counts_s3)

        recall_tot = retrieved_total / total_true_links if total_true_links > 0 else 0.0
        recall_s2 = retrieved_s2 / total_s2_links if total_s2_links > 0 else 0.0
        recall_s3 = retrieved_s3 / total_s3_links if total_s3_links > 0 else 0.0

        results[cfg_name] = {
            "recall_overall": recall_tot,
            "recall_s2": recall_s2,
            "recall_s3": recall_s3,
            "retrieved_total": retrieved_total,
            "missed_total": total_true_links - retrieved_total,
            "retrieved_s2": retrieved_s2,
            "missed_s2": total_s2_links - retrieved_s2,
            "retrieved_s3": retrieved_s3,
            "missed_s3": total_s3_links - retrieved_s3,
            "zero_candidates_count": zero_candidates_count,
            "zero_candidates_pct": zero_candidates_count / N_S1 * 100,
            "total_candidate_pairs": int(s_tot.sum()),
            "candidate_distribution": {
                "mean": float(s_tot.mean()),
                "median": float(s_tot.median()),
                "p90": float(s_tot.quantile(0.90)),
                "p95": float(s_tot.quantile(0.95)),
                "p99": float(s_tot.quantile(0.99)),
                "max": int(s_tot.max()),
            },
            "candidate_distribution_s2": {
                "mean": float(s_s2.mean()),
                "median": float(s_s2.median()),
                "max": int(s_s2.max()),
            },
            "candidate_distribution_s3": {
                "mean": float(s_s3.mean()),
                "median": float(s_s3.median()),
                "max": int(s_s3.max()),
            },
            "integrity": {
                "bad_prefix_count": bad_prefix_count,
                "self_match_count": self_match_count,
            },
        }

    t_total = time.perf_counter() - t_start
    total_indexing_time = timing["s2_pass1"] + timing["s2_pass2"] + timing["s3_pass1"] + timing["s3_pass2"]
    total_querying_time = timing["s2_query"] + timing["s3_query"]

    current_traced, peak_traced = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    final_proc_mem = get_process_memory_mb()

    # Print Comparative Table
    print("\n" + "=" * 105)
    print("FULL-CORPUS REPRODUCIBLE EVALUATION RESULTS (10,000 S1 RECORDS vs COMPLETE S2 & S3)")
    print("=" * 105)
    print(f"{'Configuration':<15} | {'Recall':<8} | {'S2 Rec':<8} | {'S3 Rec':<8} | {'Retrieved / Total':<18} | {'Mean Cand':<10} | {'Median':<8} | {'Max Cand':<9}")
    print("-" * 105)
    for cfg_name, res in results.items():
        ret_str = f"{res['retrieved_total']:,} / {total_true_links:,}"
        print(
            f"{cfg_name:<15} | {res['recall_overall']*100:>6.2f}% | {res['recall_s2']*100:>6.2f}% | {res['recall_s3']*100:>6.2f}% | "
            f"{ret_str:<18} | {res['candidate_distribution']['mean']:>10.2f} | {res['candidate_distribution']['median']:>8.1f} | {res['candidate_distribution']['max']:>9,}"
        )

    print("\n" + "=" * 90)
    print("RUNTIME & MEMORY MEASUREMENTS (Measured)")
    print("=" * 90)
    print(f"Total Full-Corpus Runtime:            {t_total:.2f} s ({t_total / 60:.2f} min)")
    print(f"Total Indexing Time (10.3M records):  {total_indexing_time:.2f} s ({total_indexing_time / 60:.2f} min)")
    print(f"  - Source 2 (Pass 1 + Pass 2):       {timing['s2_pass1'] + timing['s2_pass2']:.2f} s")
    print(f"  - Source 3 (Pass 1 + Pass 2):       {timing['s3_pass1'] + timing['s3_pass2']:.2f} s")
    print(f"Total Querying Time (10k S1):         {total_querying_time:.2f} s ({total_querying_time / N_S1 * 1000:.2f} ms/entity)")
    print(f"Python Traced Peak Memory:            {peak_traced / (1024 * 1024):.2f} MB")
    print(f"OS Process Current Working Set:       {final_proc_mem['working_set_mb']:.2f} MB")
    print(f"OS Process Peak Working Set:          {final_proc_mem['peak_working_set_mb']:.2f} MB")

    print("\nIntegrity Checks:")
    print(f"  Missing S2 Ground Truth Targets:    {len(missing_s2_targets)}")
    print(f"  Missing S3 Ground Truth Targets:    {len(missing_s3_targets)}")
    for cfg_name, res in results.items():
        print(f"  {cfg_name}: Bad Prefixes={res['integrity']['bad_prefix_count']}, Self Matches={res['integrity']['self_match_count']}")

    # Save summary to output/
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "blocking_v3_full_corpus_evaluation_summary.json"
    summary_data = {
        "evaluation_parameters": {
            "n_s1": N_S1,
            "random_seed": RANDOM_SEED,
            "s2_total_records": s2_total_records,
            "s3_total_records": s3_total_records,
            "total_true_links": total_true_links,
            "total_s2_links": total_s2_links,
            "total_s3_links": total_s3_links,
            "singletons_count": singletons,
        },
        "results": results,
        "timing_and_memory": {
            "total_elapsed_seconds": t_total,
            "total_indexing_seconds": total_indexing_time,
            "total_querying_seconds": total_querying_time,
            "timing_breakdown": timing,
            "python_traced_peak_mb": peak_traced / (1024 * 1024),
            "os_process_working_set_mb": final_proc_mem["working_set_mb"],
            "os_process_peak_working_set_mb": final_proc_mem["peak_working_set_mb"],
        },
        "target_integrity": {
            "missing_s2_targets": len(missing_s2_targets),
            "missing_s3_targets": len(missing_s3_targets),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nMachine-readable full-corpus summary saved to: {summary_path}")


if __name__ == "__main__":
    # If run with --test, only run equivalence test
    if "--test" in sys.argv:
        from blocking_v3 import test_v3_components
        test_v3_components()
        run_equivalence_test()
    else:
        run_full_evaluation()
