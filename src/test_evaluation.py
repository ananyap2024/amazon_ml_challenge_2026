"""
ML Challenge 2026 — Comprehensive Unit Tests for Evaluation Metric Engine

Verifies:
1. Exact official F_0.5 formula from README:
   - README Example: GT=[S2-00047, S3-00812], Preds=[S2-00047, S2-00193, S3-00812] -> F0.5 = 0.7142857
2. Correct matches (precision=1.0, recall=1.0 -> F0.5 = 1.0)
3. False positives on singletons (0.0) and non-singletons
4. False negatives (missed matches -> 0.0)
5. Empty predictions on singletons (1.0)
6. S1 entities with zero candidates (evaluates to empty predictions)
7. Multi-threshold evaluation grid and optimal threshold tuning
8. String ID preservation and robustness against whitespace
"""

import math
import unittest
import pandas as pd

from evaluation import (
    compute_entity_f05,
    evaluate_matches,
    evaluate_predictions,
    evaluate_threshold_grid,
    find_best_threshold,
    load_ground_truth,
)


class TestEvaluationMetric(unittest.TestCase):

    def test_readme_official_example(self):
        """
        Verify the exact example given in README.md lines 205-210:
        GT: [S2-00047, S3-00812]
        Preds: [S2-00047, S2-00193, S3-00812]
        Precision = 2/3 = 0.6667, Recall = 2/2 = 1.0
        F_0.5 = (1.25 * 0.6667 * 1.0) / (0.25 * 0.6667 + 1.0) = 0.714
        """
        gt_set = {"S2-00047", "S3-00812"}
        pred_set = {"S2-00047", "S2-00193", "S3-00812"}

        f05, prec, rec, tp, fp, fn = compute_entity_f05(gt_set, pred_set)

        self.assertEqual(tp, 2)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)
        self.assertAlmostEqual(prec, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(rec, 1.0, places=6)
        expected_f05 = (1.25 * (2.0 / 3.0) * 1.0) / (0.25 * (2.0 / 3.0) + 1.0)
        self.assertAlmostEqual(f05, expected_f05, places=6)
        self.assertAlmostEqual(f05, 0.7142857, places=6)

    def test_singletons(self):
        """
        Verify singleton scoring rules:
        - Empty prediction earns 1.0.
        - Any prediction earns 0.0 (penalizing false merges).
        """
        gt_empty = set()

        # Correct singleton (predicted empty)
        f05, prec, rec, tp, fp, fn = compute_entity_f05(gt_empty, set())
        self.assertEqual(f05, 1.0)
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

        # False merge on singleton (predicted 1 entity)
        f05, prec, rec, tp, fp, fn = compute_entity_f05(gt_empty, {"S2-99999"})
        self.assertEqual(f05, 0.0)
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 1)

    def test_perfect_matches(self):
        """Single and multiple perfect matches score 1.0."""
        # Single match
        f05, prec, rec, tp, fp, fn = compute_entity_f05({"S2-101"}, {"S2-101"})
        self.assertEqual(f05, 1.0)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)

        # Multiple matches
        f05, prec, rec, tp, fp, fn = compute_entity_f05(
            {"S2-101", "S3-201", "S3-202"}, {"S2-101", "S3-201", "S3-202"}
        )
        self.assertEqual(f05, 1.0)
        self.assertEqual(tp, 3)
        self.assertEqual(fp, 0)
        self.assertEqual(fn, 0)

    def test_false_negatives(self):
        """Missed matches score 0.0."""
        f05, prec, rec, tp, fp, fn = compute_entity_f05({"S2-101"}, set())
        self.assertEqual(f05, 0.0)
        self.assertEqual(rec, 0.0)
        self.assertEqual(fn, 1)

    def test_disjoint_false_positives(self):
        """Predicted matches with zero overlap score 0.0."""
        f05, prec, rec, tp, fp, fn = compute_entity_f05({"S2-101"}, {"S3-999"})
        self.assertEqual(f05, 0.0)
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)

    def test_macro_averaging_and_zero_candidates(self):
        """
        Verify macro-averaging across 4 entities:
        - S1-01: Perfect match (score 1.0)
        - S1-02: True singleton, predicted empty (score 1.0)
        - S1-03: True singleton, false merge (score 0.0)
        - S1-04: Has matches, but zero candidates / zero predictions (score 0.0)
        Expected Macro F_0.5 = (1.0 + 1.0 + 0.0 + 0.0) / 4 = 0.5000
        """
        gt = {
            "S1-01": {"S2-101"},
            "S1-02": set(),
            "S1-03": set(),
            "S1-04": {"S3-401"},
        }
        # S1-04 is completely absent from predicted_matches (simulating 0 candidates)
        preds = {
            "S1-01": {"S2-101"},
            "S1-02": set(),
            "S1-03": {"S2-999"},
        }

        report = evaluate_matches(gt, preds)
        self.assertEqual(report.total_entities, 4)
        self.assertAlmostEqual(report.macro_f05, 0.5000, places=4)
        self.assertEqual(report.n_singletons, 2)
        self.assertEqual(report.singleton_correct, 1)
        self.assertEqual(report.singleton_false_positives, 1)
        self.assertAlmostEqual(report.singleton_accuracy, 0.50, places=4)
        self.assertEqual(report.n_matched_entities, 2)
        self.assertAlmostEqual(report.matched_macro_f05, 0.50, places=4)
        self.assertEqual(report.matched_zero_predictions, 1)

    def test_threshold_grid_and_tuning(self):
        """
        Test threshold sweep without retraining:
        Entity 1 (True: S2-101):
          - S2-101 has prob 0.85
          - S3-999 has prob 0.40 (FP)
        At threshold 0.3:
          Preds = [S2-101, S3-999] -> P = 1/2, R = 1.0 -> F0.5 = 5/9 = 0.5556
        At threshold 0.5:
          Preds = [S2-101] -> P = 1.0, R = 1.0 -> F0.5 = 1.0
        At threshold 0.9:
          Preds = [] -> F0.5 = 0.0
        """
        gt = {"S1-01": {"S2-101"}}
        preds_df = pd.DataFrame([
            {"source1_entity_id": "S1-01", "candidate_entity_id": "S2-101", "match_probability": 0.85},
            {"source1_entity_id": "S1-01", "candidate_entity_id": "S3-999", "match_probability": 0.40},
        ])

        thresholds = [0.30, 0.50, 0.90]
        grid_df = evaluate_threshold_grid(gt, preds_df, thresholds=thresholds)

        self.assertEqual(len(grid_df), 3)

        # Check threshold 0.30
        row_03 = grid_df[grid_df["threshold"] == 0.30].iloc[0]
        self.assertAlmostEqual(row_03["macro_f05"], 5.0 / 9.0, places=4)

        # Check threshold 0.50
        row_05 = grid_df[grid_df["threshold"] == 0.50].iloc[0]
        self.assertAlmostEqual(row_05["macro_f05"], 1.0, places=4)

        # Check threshold 0.90
        row_09 = grid_df[grid_df["threshold"] == 0.90].iloc[0]
        self.assertAlmostEqual(row_09["macro_f05"], 0.0, places=4)

        # Best threshold should be 0.50
        best_t, best_report, _ = find_best_threshold(gt, preds_df, thresholds=thresholds)
        self.assertAlmostEqual(best_t, 0.50, places=2)
        self.assertAlmostEqual(best_report.macro_f05, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
