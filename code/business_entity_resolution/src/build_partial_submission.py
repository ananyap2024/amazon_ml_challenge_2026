"""
ML Challenge 2026 — Build Partial Submission from Scored Candidates

Constructs `output/matching_results_partial.tsv` using the model-scored candidate pairs
accumulated so far by `batch_inference.py`, ensuring:
1. Strict 1-to-1 alignment with `dataset/test/test_source1.tsv` (all 1,732,544 S1s in exact order).
2. Exactly one row per test S1 ID.
3. Only candidate pairs scored with `predicted_match == 1` and present in that S1's official
   `candidate_pairs.tsv` are included as positive matches.
4. Unscored candidates and unscored S1s are NEVER fabricated or treated as matches.
5. Strict safety assertions: no self-matches, no duplicates, subset constraint verified.
"""

from collections import OrderedDict
import os
from pathlib import Path
import sys
import time

# =============================================================================
# Paths
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parents[3]

S1_PATH = PROJECT_DIR / "dataset" / "test" / "test_source1.tsv"
CANDIDATE_PATH = PROJECT_DIR / "output" / "candidate_pairs.tsv"
SCORED_TMP_PATH = PROJECT_DIR / "output" / "matching_results_scored.tsv.tmp"
OUTPUT_PATH = PROJECT_DIR / "output" / "matching_results_partial.tsv"


def main():
    start_time = time.time()
    print("=" * 72)
    print("BUILD PARTIAL SUBMISSION FROM SCORED CANDIDATE PAIRS")
    print("=" * 72)
    print(f"Project root    : {PROJECT_DIR}")
    print(f"Test Source 1   : {S1_PATH}")
    print(f"Official Cands  : {CANDIDATE_PATH}")
    print(f"Scored Pairs    : {SCORED_TMP_PATH}")
    print(f"Target Output   : {OUTPUT_PATH}")

    # Validate input file existence
    if not S1_PATH.is_file():
        raise FileNotFoundError(f"Test Source 1 file not found: {S1_PATH}")
    if not CANDIDATE_PATH.is_file():
        raise FileNotFoundError(f"Official candidate pairs file not found: {CANDIDATE_PATH}")
    if not SCORED_TMP_PATH.is_file():
        raise FileNotFoundError(f"Scored temporary file not found: {SCORED_TMP_PATH}")

    # -------------------------------------------------------------------------
    # 1. Load test Source 1 entity IDs in exact file order
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading test Source 1 entity IDs...")
    all_s1_ids = []
    with open(S1_PATH, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if "entity_id" not in header:
            raise ValueError(f"Expected 'entity_id' in {S1_PATH} header, got: {header}")
        id_idx = header.index("entity_id")

        for line in f:
            line_clean = line.rstrip("\r\n")
            if not line_clean:
                continue
            parts = line_clean.split("\t")
            if len(parts) > id_idx:
                all_s1_ids.append(parts[id_idx].strip())

    total_test_s1 = len(all_s1_ids)
    unique_test_s1_count = len(set(all_s1_ids))
    print(f"      Total test S1 rows loaded : {total_test_s1:,}")
    print(f"      Unique test S1 IDs        : {unique_test_s1_count:,}")

    if total_test_s1 != unique_test_s1_count:
        raise ValueError(
            f"Duplicate S1 IDs found in {S1_PATH}: {total_test_s1} rows vs {unique_test_s1_count} unique IDs"
        )

    # -------------------------------------------------------------------------
    # 2. Extract qualifying positive model matches from scored temporary TSV
    # -------------------------------------------------------------------------
    print("\n[2/4] Reading scored pairs from temporary inference output...")
    # Map: s1_id -> list of candidate_ids where predicted_match == 1
    scored_positive_by_s1 = OrderedDict()
    total_scored_rows_read = 0
    total_positive_scored = 0

    with open(SCORED_TMP_PATH, "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\r\n")
        header = header_line.split("\t")

        req_cols = ["source1_entity_id", "candidate_entity_id", "predicted_match"]
        for col in req_cols:
            if col not in header:
                raise ValueError(f"Required column '{col}' missing from {SCORED_TMP_PATH} header: {header}")

        s1_col_idx = header.index("source1_entity_id")
        cand_col_idx = header.index("candidate_entity_id")
        pred_col_idx = header.index("predicted_match")

        for line in f:
            line_clean = line.rstrip("\r\n")
            if not line_clean:
                continue
            parts = line_clean.split("\t")
            if len(parts) <= max(s1_col_idx, cand_col_idx, pred_col_idx):
                # Possible trailing partially flushed line if job is writing; skip cleanly
                continue

            total_scored_rows_read += 1
            pred_val = parts[pred_col_idx].strip()

            if pred_val == "1":
                total_positive_scored += 1
                s1_id = parts[s1_col_idx].strip()
                cand_id = parts[cand_col_idx].strip()

                if s1_id not in scored_positive_by_s1:
                    scored_positive_by_s1[s1_id] = []

                # Deduplicate while preserving order of appearance
                if cand_id not in scored_positive_by_s1[s1_id]:
                    scored_positive_by_s1[s1_id].append(cand_id)

    print(f"      Scored rows read from file: {total_scored_rows_read:,}")
    print(f"      Positive matches found    : {total_positive_scored:,}")
    print(f"      S1 entities with matches  : {len(scored_positive_by_s1):,}")

    # -------------------------------------------------------------------------
    # 3. Verify positive matches against official candidate_pairs.tsv
    # -------------------------------------------------------------------------
    print("\n[3/4] Validating matches against official candidate lists...")
    verified_matches_by_s1 = {}
    excluded_not_in_candidate_list = 0
    excluded_self_matches = 0
    s1_with_positive_set = set(scored_positive_by_s1.keys())

    # Stream candidate_pairs.tsv to avoid heavy RAM footprint
    with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            raise ValueError(f"Unexpected header in {CANDIDATE_PATH}: {header}")

        for line in f:
            line_clean = line.rstrip("\r\n")
            if not line_clean:
                continue
            parts = line_clean.split("\t")
            s1_id = parts[0].strip()

            if s1_id in s1_with_positive_set:
                raw_cands = parts[1].strip() if len(parts) > 1 else ""
                official_cands = set(raw_cands.split(",")) if raw_cands else set()

                valid_list = []
                for cand_id in scored_positive_by_s1[s1_id]:
                    if cand_id == s1_id:
                        excluded_self_matches += 1
                        continue
                    if cand_id in official_cands:
                        valid_list.append(cand_id)
                    else:
                        excluded_not_in_candidate_list += 1

                if valid_list:
                    verified_matches_by_s1[s1_id] = valid_list

    print(f"      Excluded self-matches     : {excluded_self_matches:,}")
    print(f"      Excluded non-candidate IDs: {excluded_not_in_candidate_list:,}")
    print(f"      Verified S1 with matches  : {len(verified_matches_by_s1):,}")

    # -------------------------------------------------------------------------
    # 4. Generate output/matching_results_partial.tsv
    # -------------------------------------------------------------------------
    print(f"\n[4/4] Writing {OUTPUT_PATH}...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = OUTPUT_PATH.with_suffix(".tmp")

    total_matched_ids_written = 0
    s1_with_at_least_one_match = 0
    output_rows_written = 0

    with open(tmp_output, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tmatched_entity_ids\n")

        for s1_id in all_s1_ids:
            matches = verified_matches_by_s1.get(s1_id, [])
            if matches:
                s1_with_at_least_one_match += 1
                total_matched_ids_written += len(matches)
                matched_str = ",".join(matches)
            else:
                matched_str = ""

            out_f.write(f"{s1_id}\t{matched_str}\n")
            output_rows_written += 1

    # Atomic rename
    if tmp_output.exists():
        os.replace(tmp_output, OUTPUT_PATH)

    output_file_size = OUTPUT_PATH.stat().st_size

    # -------------------------------------------------------------------------
    # 5. Safety Checks
    # -------------------------------------------------------------------------
    print("\n--- Running Mandatory Safety Verifications ---")

    # Check A: Every test S1 appears exactly once in the output
    if output_rows_written != total_test_s1:
        raise AssertionError(
            f"Safety check failed: output row count ({output_rows_written:,}) does not equal test S1 count ({total_test_s1:,})"
        )
    print("  [PASS] Every test S1 appears exactly once in the output.")

    # Check B: Output S1 IDs match test S1 IDs in exact order
    with open(OUTPUT_PATH, "r", encoding="utf-8") as check_f:
        check_header = check_f.readline().rstrip("\r\n").split("\t")
        assert check_header == ["source1_entity_id", "matched_entity_ids"], f"Header mismatch: {check_header}"

        for idx, (expected_s1, line) in enumerate(zip(all_s1_ids, check_f)):
            parts = line.rstrip("\r\n").split("\t")
            actual_s1 = parts[0]
            if actual_s1 != expected_s1:
                raise AssertionError(
                    f"Safety check failed at line {idx+2}: expected '{expected_s1}', got '{actual_s1}'"
                )

            # Check C & D: Verify matches format, uniqueness, and no self-matches
            if len(parts) > 1 and parts[1].strip():
                match_ids = parts[1].strip().split(",")
                if len(match_ids) != len(set(match_ids)):
                    raise AssertionError(f"Duplicate match IDs found for S1 '{actual_s1}': {match_ids}")
                if actual_s1 in match_ids:
                    raise AssertionError(f"Self-match found for S1 '{actual_s1}': {match_ids}")

    print("  [PASS] Output S1 IDs match test S1 IDs in exact file order.")
    print("  [PASS] No duplicate match IDs and no self-matches exist.")
    print("  [PASS] All output matches are verified subsets of official candidates.")

    # -------------------------------------------------------------------------
    # Summary Report
    # -------------------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    print("PARTIAL SUBMISSION SUMMARY REPORT")
    print("=" * 72)
    print(f"Number of test S1 rows              : {total_test_s1:,}")
    print(f"Number of output rows               : {output_rows_written:,}")
    print(f"Number of unique S1 IDs             : {unique_test_s1_count:,}")
    print(f"Number of S1 rows with >= 1 match   : {s1_with_at_least_one_match:,}")
    print(f"Total number of matched IDs         : {total_matched_ids_written:,}")
    print(f"Excluded positive scored rows (not in candidate list): {excluded_not_in_candidate_list:,}")
    print(f"Output file path                    : {OUTPUT_PATH}")
    print(f"Output file size                    : {output_file_size / (1024 * 1024):.2f} MB ({output_file_size:,} bytes)")
    print(f"Elapsed time                        : {elapsed:.2f} seconds")
    print("=" * 72)


if __name__ == "__main__":
    main()
