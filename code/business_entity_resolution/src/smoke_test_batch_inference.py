"""Smoke test for batch_inference.py.

Verifies end-to-end functionality using a small sample (25 rows) from
train_candidate_pairs_sample1000.tsv:
1. Multi-chunk streaming (chunk_size=10 on 25 rows -> 3 chunks).
2. Exact ID and source preservation in input row order.
3. Probability output and bounds ([0, 1]).
4. Thresholding behavior when threshold is provided (predicted_match column).
5. Decoupled behavior when threshold is None (no predicted_match column).
6. Automatic temporary file cleanup.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parents[2]
_WORKSPACE_ROOT = _REPO_ROOT.parent

sys.path.insert(0, str(_SRC_DIR))
import batch_inference  # noqa: E402

SAMPLE_PAIRS_SOURCE = _WORKSPACE_ROOT / "student_resource" / "dataset" / "train" / "train_candidate_pairs_sample1000.tsv"
TRAIN_SOURCE1 = _WORKSPACE_ROOT / "student_resource" / "dataset" / "train" / "train_source1.tsv"
TRAIN_SOURCE2 = _WORKSPACE_ROOT / "student_resource" / "dataset" / "train" / "train_source2.tsv"
TRAIN_SOURCE3 = _WORKSPACE_ROOT / "student_resource" / "dataset" / "train" / "train_source3.tsv"

N_SAMPLE_ROWS = 25
TEST_CHUNK_SIZE = 10
TEST_THRESHOLD = 0.60


def main():
    print("=" * 70)
    print("SMOKE TEST: batch_inference.py")
    print("=" * 70)

    assert SAMPLE_PAIRS_SOURCE.is_file(), f"Sample candidate pairs file not found: {SAMPLE_PAIRS_SOURCE}"
    assert TRAIN_SOURCE1.is_file(), f"Source 1 file not found: {TRAIN_SOURCE1}"
    assert TRAIN_SOURCE2.is_file(), f"Source 2 file not found: {TRAIN_SOURCE2}"
    assert TRAIN_SOURCE3.is_file(), f"Source 3 file not found: {TRAIN_SOURCE3}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        test_pairs_path = tmp_path / "test_pairs_sample.tsv"
        test_output_thresholded = tmp_path / "test_output_thresholded.tsv"
        test_output_probabilities_only = tmp_path / "test_output_probabilities_only.tsv"
        test_summary_path = tmp_path / "test_summary.json"

        # 1. Prepare sample pairs file
        print(f"\n[Test Setup] Extracting {N_SAMPLE_ROWS} candidate rows to {test_pairs_path.name}...")
        sample_df = pd.read_csv(SAMPLE_PAIRS_SOURCE, sep="\t", nrows=N_SAMPLE_ROWS)
        sample_df.to_csv(test_pairs_path, sep="\t", index=False)
        print(f"             Sample rows written: {len(sample_df)}")

        # 2. Test Run A: Multi-chunk run with explicit threshold (0.60)
        print(f"\n[Test Run A] Running batch inference with threshold={TEST_THRESHOLD}, chunk_size={TEST_CHUNK_SIZE}...")
        stats_a = batch_inference.run_batch_inference(
            pairs_path=test_pairs_path,
            source1_path=TRAIN_SOURCE1,
            source2_path=TRAIN_SOURCE2,
            source3_path=TRAIN_SOURCE3,
            output_path=test_output_thresholded,
            threshold=TEST_THRESHOLD,
            chunk_size=TEST_CHUNK_SIZE,
            summary_path=test_summary_path,
        )

        assert test_output_thresholded.is_file(), "Output file test_output_thresholded.tsv was not created!"
        res_a = pd.read_csv(test_output_thresholded, sep="\t")

        print("\n--- Verifying Test Run A Results ---")
        print(f"  Rows output: {len(res_a)} (expected: {N_SAMPLE_ROWS})")
        assert len(res_a) == N_SAMPLE_ROWS, f"Expected {N_SAMPLE_ROWS} rows, got {len(res_a)}"

        expected_cols_a = ["source1_entity_id", "candidate_entity_id", "source", "match_probability", "predicted_match"]
        assert list(res_a.columns) == expected_cols_a, f"Expected columns {expected_cols_a}, got {list(res_a.columns)}"
        print("  Columns: OK (includes predicted_match)")

        # Verify exact alignment and ordering
        for col in ["source1_entity_id", "candidate_entity_id", "source"]:
            assert (res_a[col].to_numpy() == sample_df[col].to_numpy()).all(), f"Mismatch in {col} alignment!"
        print("  Exact ID, source, and row ordering preservation: OK")

        # Verify probability properties
        probs_a = res_a["match_probability"].to_numpy()
        assert np.isfinite(probs_a).all(), "Found non-finite probabilities!"
        assert ((probs_a >= 0.0) & (probs_a <= 1.0)).all(), "Probabilities outside [0, 1]!"
        print(f"  Probability range: min={probs_a.min():.6f}, max={probs_a.max():.6f} (all in [0, 1]): OK")

        # Verify thresholding
        preds_a = res_a["predicted_match"].to_numpy()
        expected_preds = (probs_a >= TEST_THRESHOLD).astype(int)
        assert (preds_a == expected_preds).all(), "predicted_match does not match (prob >= threshold)!"
        print(f"  Thresholding logic (>= {TEST_THRESHOLD}): OK ({preds_a.sum()} matches predicted)")

        # Verify chunks processed
        expected_chunks = (N_SAMPLE_ROWS + TEST_CHUNK_SIZE - 1) // TEST_CHUNK_SIZE
        assert stats_a["chunks_processed"] == expected_chunks, f"Expected {expected_chunks} chunks, got {stats_a['chunks_processed']}"
        print(f"  Multi-chunk streaming ({stats_a['chunks_processed']} chunks processed): OK")

        # 3. Test Run B: Run with threshold=None (probabilities only)
        print(f"\n[Test Run B] Running batch inference with threshold=None (decoupled threshold)...")
        stats_b = batch_inference.run_batch_inference(
            pairs_path=test_pairs_path,
            source1_path=TRAIN_SOURCE1,
            source2_path=TRAIN_SOURCE2,
            source3_path=TRAIN_SOURCE3,
            output_path=test_output_probabilities_only,
            threshold=None,
            chunk_size=TEST_CHUNK_SIZE,
        )

        assert test_output_probabilities_only.is_file(), "Output file was not created!"
        res_b = pd.read_csv(test_output_probabilities_only, sep="\t")

        expected_cols_b = ["source1_entity_id", "candidate_entity_id", "source", "match_probability"]
        assert list(res_b.columns) == expected_cols_b, f"Expected columns {expected_cols_b}, got {list(res_b.columns)}"
        assert len(res_b) == N_SAMPLE_ROWS
        print("  Columns: OK (predicted_match omitted, outputting probabilities only)")

        # Probabilities should be identical to Run A
        probs_b = res_b["match_probability"].to_numpy()
        assert np.allclose(probs_a, probs_b, atol=1e-5), "Probabilities differed between Run A and Run B!"
        print("  Probability reproducibility across runs: OK")

        print("\n" + "=" * 70)
        print("ALL SMOKE TESTS PASSED")
        print("=" * 70)


if __name__ == "__main__":
    main()
