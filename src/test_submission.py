"""
ML Challenge 2026 — Submission Generator Tests & Synthetic Demo

Validates:
1. Loading S1 IDs from file or sequence.
2. Formatting and deduplication of candidate pairs.
3. Probability threshold filtering and candidate subset enforcement.
4. Correct empty-field serialization (no NaN, null, None, []).
5. Full compliance with the official utils/validate_submission.py script.
"""

import shutil
import tempfile
from pathlib import Path
import pandas as pd

from submission import (
    DELIM,
    build_submission_dataframes,
    generate_submission_files,
    load_source1_ids,
    write_submission_tsv,
    run_validator,
)


def run_synthetic_test():
    print("=" * 70)
    print("RUNNING SYNTHETIC SUBMISSION TEST & VALIDATION")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_dir = temp_path / "dataset" / "test"
        test_dir.mkdir(parents=True)
        output_dir = temp_path / "output"
        output_dir.mkdir(parents=True)

        # 1. Create synthetic test_source1.tsv with 4 entities
        # S1-00001: Has candidates and matches above threshold
        # S1-00002: Has candidates, but predictions are below threshold (should be empty match)
        # S1-00003: Has no candidates at all (singleton, both empty)
        # S1-00004: Has candidates, and a prediction has an ID not in candidates (subset test)
        synthetic_s1 = [
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n",
            "S1-00001\tAlpha Corp\t123 Main St\tUS\n",
            "S1-00002\tBeta LLC\t456 Oak Rd\tIndia\n",
            "S1-00003\tGamma Inc\t789 Pine Ave\tFrance\n",
            "S1-00004\tDelta Solutions\t101 Blvd\tUS\n",
        ]
        s1_file = test_dir / "test_source1.tsv"
        with open(s1_file, "w", encoding="utf-8") as f:
            f.writelines(synthetic_s1)

        # 1b. Create synthetic test_source2.tsv and test_source3.tsv for check_ids testing
        s2_file = test_dir / "test_source2.tsv"
        s2_file.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S2-10001\tAlpha Inc\t123 Main St\tUS\n"
            "S2-10002\tBeta LLC\t456 Oak Rd\tIndia\n",
            encoding="utf-8",
        )
        s3_file = test_dir / "test_source3.tsv"
        s3_file.write_text(
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            "S3-20001\tAlpha Co\t123 Main St\tUS\n"
            "S3-20004\tDelta Corp\t101 Blvd\tUS\n",
            encoding="utf-8",
        )

        # 2. Synthetic candidates
        # S1-00001: S2-10001, S3-20001, S2-10001 (duplicate candidate to test dedup)
        # S1-00002: S2-10002
        # S1-00003: No candidates
        # S1-00004: S3-20004
        candidates_data = [
            {"source1_entity_id": "S1-00001", "candidate_entity_id": "S2-10001", "source": "source2"},
            {"source1_entity_id": "S1-00001", "candidate_entity_id": "S3-20001", "source": "source3"},
            {"source1_entity_id": "S1-00001", "candidate_entity_id": "S2-10001", "source": "source2"},  # duplicate
            {"source1_entity_id": "S1-00002", "candidate_entity_id": "S2-10002", "source": "source2"},
            {"source1_entity_id": "S1-00004", "candidate_entity_id": "S3-20004", "source": "source3"},
        ]
        candidates_df = pd.DataFrame(candidates_data)

        # 3. Synthetic predictions with threshold = 0.5
        # S1-00001:
        #   S2-10001: prob=0.92 (>= 0.5 -> MATCH)
        #   S3-20001: prob=0.45 (< 0.5 -> NO MATCH)
        # S1-00002:
        #   S2-10002: prob=0.30 (< 0.5 -> NO MATCH)
        # S1-00004:
        #   S3-20004: prob=0.88 (>= 0.5 -> MATCH)
        #   S2-99999: prob=0.99 (>= 0.5, but S2-99999 is NOT in candidate list -> must be rejected by subset rule!)
        predictions_data = [
            {"source1_entity_id": "S1-00001", "candidate_entity_id": "S2-10001", "match_probability": 0.92},
            {"source1_entity_id": "S1-00001", "candidate_entity_id": "S3-20001", "match_probability": 0.45},
            {"source1_entity_id": "S1-00002", "candidate_entity_id": "S2-10002", "match_probability": 0.30},
            {"source1_entity_id": "S1-00004", "candidate_entity_id": "S3-20004", "match_probability": 0.88},
            {"source1_entity_id": "S1-00004", "candidate_entity_id": "S2-99999", "match_probability": 0.99},
        ]
        predictions_df = pd.DataFrame(predictions_data)

        # 4. Generate submission
        matching_path, candidate_path = generate_submission_files(
            all_source1_ids=s1_file,
            candidate_pairs=candidates_df,
            predictions=predictions_df,
            threshold=0.5,
            output_dir=output_dir,
            validate_after=False,
            test_dir=test_dir,
        )

        print("\n--- Generated candidate_pairs.tsv ---")
        cand_content = candidate_path.read_text(encoding="utf-8")
        print(cand_content)

        print("--- Generated matching_results.tsv ---")
        matching_content = matching_path.read_text(encoding="utf-8")
        print(matching_content)

        # Assertions
        assert "S1-00001\tS2-10001,S3-20001\n" in cand_content, "Candidate deduplication failed for S1-00001"
        assert "S1-00002\tS2-10002\n" in cand_content
        assert "S1-00003\t\n" in cand_content, "Empty candidate representation failed for S1-00003"
        assert "S1-00004\tS3-20004\n" in cand_content

        assert "S1-00001\tS2-10001\n" in matching_content, "High-prob match not included for S1-00001"
        assert "S1-00002\t\n" in matching_content, "Sub-threshold match not empty for S1-00002"
        assert "S1-00003\t\n" in matching_content, "Singleton match not empty for S1-00003"
        assert "S1-00004\tS3-20004\n" in matching_content, "Subset enforcement or match failed for S1-00004"
        assert "S2-99999" not in matching_content, "Non-candidate match S2-99999 was not filtered out!"

        # 5. Run the official validator on the synthetic dataset (with full --check-ids)!
        print("\n--- Executing Official Validator on Synthetic Outputs (--check-ids) ---")
        ret = run_validator(
            matching_path=matching_path,
            candidate_path=candidate_path,
            test_dir=test_dir,
            check_ids=True,
        )
        assert ret == 0, f"Official validator returned non-zero exit code: {ret}"
        print("\n[SUCCESS] All synthetic tests and official validation passed with exit code 0!")


if __name__ == "__main__":
    run_synthetic_test()
