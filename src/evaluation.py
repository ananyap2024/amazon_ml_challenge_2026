"""
ML Challenge 2026 — Model Evaluation & Metric Engine (Member 3: Pipeline, Integration & Submission)

Implements the exact official evaluation criteria specified in README.md:
- Macro-averaged F_0.5 score across all Source 1 entities:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
- Precision-heavy weighting (beta = 0.5, weighting precision 2x over recall).
- Singletons (entities with 0 ground-truth matches):
    - Predicting an empty match list earns 1.0.
    - Predicting any match earns 0.0 (penalizing false merges).
- Non-singletons (entities with >=1 ground-truth matches):
    - Predicting an empty list earns 0.0 (missed matches).
    - Predicting matches without true positive overlap earns 0.0.
    - Partial or full matches evaluated via the official F_0.5 formula.
- Unmatched / un-candidated entities:
    - S1 entities in the validation set with 0 candidates are evaluated as empty predictions (1.0 if true singleton, 0.0 otherwise).

Key Capabilities:
- Load ground-truth TSVs line-by-line without loading full text columns into memory.
- Evaluate predictions across multiple probability thresholds without retraining.
- Produce fine-grained diagnostic breakdowns (singleton accuracy, non-singleton macro precision/recall, pair-level micro metrics).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

DELIM = "\t"


@dataclass
class EntityScore:
    """Detailed score breakdown for a single Source 1 entity."""

    source1_entity_id: str
    is_singleton: bool
    true_count: int
    pred_count: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f05: float
    singleton_correct: bool


@dataclass
class EvaluationReport:
    """Complete evaluation report for a validation run."""

    # Official Macro Metric
    macro_f05: float
    total_entities: int

    # Singletons (|True| == 0)
    n_singletons: int
    singleton_correct: int
    singleton_accuracy: float
    singleton_false_positives: int

    # Matched Entities (|True| >= 1)
    n_matched_entities: int
    matched_macro_precision: float
    matched_macro_recall: float
    matched_macro_f05: float
    matched_zero_predictions: int

    # Pair-level Micro Diagnostics
    total_tp: int
    total_fp: int
    total_fn: int
    global_precision: float
    global_recall: float
    global_f05: float

    # Threshold & Metadata
    threshold: Optional[float] = None
    entity_scores: List[EntityScore] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        """Return a formatted string table of the evaluation results."""
        thresh_str = f" (Threshold: {self.threshold:.3f})" if self.threshold is not None else ""
        lines = [
            "=" * 70,
            f"OFFICIAL VALIDATION EVALUATION REPORT{thresh_str}",
            "=" * 70,
            f"Total Evaluated S1 Entities:     {self.total_entities:,}",
            f">> OFFICIAL MACRO F_0.5 SCORE:   {self.macro_f05:.4f} <<",
            "-" * 70,
            "SINGLETON ANALYSIS (|True Matches| == 0):",
            f"  Ground Truth Singletons:       {self.n_singletons:,} ({self.n_singletons / max(1, self.total_entities) * 100:.1f}%)",
            f"  Correctly Predicted Empty:     {self.singleton_correct:,} (Score = 1.0 each)",
            f"  False Merge Errors:            {self.singleton_false_positives:,} (Score = 0.0 each)",
            f"  Singleton Accuracy:            {self.singleton_accuracy * 100:.2f}%",
            "-" * 70,
            "MATCHED ENTITIES ANALYSIS (|True Matches| >= 1):",
            f"  Entities with >=1 True Match:  {self.n_matched_entities:,} ({self.n_matched_entities / max(1, self.total_entities) * 100:.1f}%)",
            f"  Matched Macro Precision:       {self.matched_macro_precision:.4f}",
            f"  Matched Macro Recall:          {self.matched_macro_recall:.4f}",
            f"  Matched Macro F_0.5:           {self.matched_macro_f05:.4f}",
            f"  Completely Missed (0 preds):   {self.matched_zero_predictions:,} ({self.matched_zero_predictions / max(1, self.n_matched_entities) * 100:.1f}%)",
            "-" * 70,
            "GLOBAL PAIR-LEVEL (MICRO) DIAGNOSTICS:",
            f"  True Positive Links:           {self.total_tp:,}",
            f"  False Positive Links:          {self.total_fp:,}",
            f"  False Negative Links:          {self.total_fn:,}",
            f"  Global Precision:              {self.global_precision:.4f}",
            f"  Global Recall:                 {self.global_recall:.4f}",
            f"  Global Micro F_0.5:            {self.global_f05:.4f}",
            "=" * 70,
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Union[float, int]]:
        """Return summary metrics as a dictionary."""
        return {
            "threshold": self.threshold if self.threshold is not None else float("nan"),
            "macro_f05": self.macro_f05,
            "total_entities": self.total_entities,
            "n_singletons": self.n_singletons,
            "singleton_correct": self.singleton_correct,
            "singleton_accuracy": self.singleton_accuracy,
            "singleton_false_positives": self.singleton_false_positives,
            "n_matched_entities": self.n_matched_entities,
            "matched_macro_precision": self.matched_macro_precision,
            "matched_macro_recall": self.matched_macro_recall,
            "matched_macro_f05": self.matched_macro_f05,
            "matched_zero_predictions": self.matched_zero_predictions,
            "total_tp": self.total_tp,
            "total_fp": self.total_fp,
            "total_fn": self.total_fn,
            "global_precision": self.global_precision,
            "global_recall": self.global_recall,
            "global_f05": self.global_f05,
        }


def compute_entity_f05(
    true_set: Set[str], pred_set: Set[str]
) -> Tuple[float, float, float, int, int, int]:
    """
    Compute official F_0.5, precision, recall, tp, fp, and fn for a single Source 1 entity.

    Rules from README:
    - If true_set is empty (singleton):
        - If pred_set is empty -> F_0.5 = 1.0 (correct singleton)
        - If pred_set is not empty -> F_0.5 = 0.0 (false merge)
    - If true_set is not empty:
        - If pred_set is empty -> F_0.5 = 0.0 (false negative)
        - Otherwise:
            precision = tp / len(pred_set)
            recall = tp / len(true_set)
            F_0.5 = (1.25 * precision * recall) / (0.25 * precision + recall)

    Returns
    -------
    Tuple[float, float, float, int, int, int]
        (f05, precision, recall, tp, fp, fn)
    """
    n_true = len(true_set)
    n_pred = len(pred_set)

    # Case 1: Singleton (0 ground-truth matches)
    if n_true == 0:
        if n_pred == 0:
            return 1.0, 1.0, 1.0, 0, 0, 0
        else:
            return 0.0, 0.0, 0.0, 0, n_pred, 0

    # Case 2: Non-singleton, empty predictions
    if n_pred == 0:
        return 0.0, 0.0, 0.0, 0, 0, n_true

    # Case 3: Non-singleton with predictions
    tp = len(true_set & pred_set)
    fp = n_pred - tp
    fn = n_true - tp

    if tp == 0:
        return 0.0, 0.0, 0.0, 0, fp, fn

    precision = tp / n_pred
    recall = tp / n_true
    denom = (0.25 * precision) + recall

    f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0
    return f05, precision, recall, tp, fp, fn


def load_ground_truth(
    gt_input: Union[str, Path, pd.DataFrame, Dict[str, Union[Set[str], List[str]]]],
    subset_s1_ids: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Load ground truth mapping {source1_entity_id -> set_of_matched_ids}.

    Preserves all entity IDs as strings. When reading from a TSV file, reads line-by-line
    to minimize memory usage and avoid loading full datasets into memory.

    Parameters
    ----------
    gt_input : Union[str, Path, pd.DataFrame, Dict]
        File path to train_ground_truth.tsv or a DataFrame/dict.
    subset_s1_ids : Optional[Set[str]]
        If provided, only loads records whose source1_entity_id is in this set.

    Returns
    -------
    Dict[str, Set[str]]
        Mapping of source1_entity_id to set of matching IDs (empty set for singletons).
    """
    gt_map: Dict[str, Set[str]] = {}

    if isinstance(gt_input, dict):
        for s1, matches in gt_input.items():
            s1_str = str(s1).strip()
            if subset_s1_ids is not None and s1_str not in subset_s1_ids:
                continue
            gt_map[s1_str] = {str(m).strip() for m in matches if str(m).strip()}

    elif isinstance(gt_input, pd.DataFrame):
        req_cols = {"source1_entity_id", "matched_entity_ids"}
        if not req_cols.issubset(gt_input.columns):
            raise ValueError(f"Ground truth DataFrame must contain columns: {req_cols}")

        s1_series = gt_input["source1_entity_id"].astype(str)
        match_series = gt_input["matched_entity_ids"].fillna("").astype(str)

        for s1, rest in zip(s1_series, match_series):
            s1_clean = s1.strip()
            if subset_s1_ids is not None and s1_clean not in subset_s1_ids:
                continue
            matches = {m.strip() for m in rest.split(",") if m.strip()}
            gt_map[s1_clean] = matches

    elif isinstance(gt_input, (str, Path)):
        path = Path(gt_input)
        if not path.is_file():
            raise FileNotFoundError(f"Ground truth file not found: {path}")

        with open(path, mode="r", encoding="utf-8") as f:
            header = f.readline()
            if not header:
                raise ValueError(f"Ground truth file is empty: {path}")

            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                s1, tab, rest = line.partition(DELIM)
                s1_clean = s1.strip()
                if subset_s1_ids is not None and s1_clean not in subset_s1_ids:
                    continue

                if not tab:
                    gt_map[s1_clean] = set()
                    continue

                rest_clean = rest.strip()
                if not rest_clean:
                    gt_map[s1_clean] = set()
                else:
                    gt_map[s1_clean] = {m.strip() for m in rest_clean.split(",") if m.strip()}
    else:
        raise TypeError(f"Unsupported type for gt_input: {type(gt_input)}")

    return gt_map


def load_predictions(
    pred_input: Union[str, Path, pd.DataFrame]
) -> pd.DataFrame:
    """
    Load model predictions DataFrame with required columns:
    [source1_entity_id, candidate_entity_id, match_probability].

    Preserves ID columns as strings and validates data types.
    """
    if isinstance(pred_input, pd.DataFrame):
        df = pred_input.copy()
    elif isinstance(pred_input, (str, Path)):
        p = Path(pred_input)
        if not p.is_file():
            raise FileNotFoundError(f"Predictions file not found: {p}")
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p, sep=DELIM, dtype={"source1_entity_id": str, "candidate_entity_id": str})
    else:
        raise TypeError(f"Unsupported type for pred_input: {type(pred_input)}")

    required_cols = {"source1_entity_id", "candidate_entity_id", "match_probability"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    df["source1_entity_id"] = df["source1_entity_id"].astype(str).str.strip()
    df["candidate_entity_id"] = df["candidate_entity_id"].astype(str).str.strip()
    df["match_probability"] = pd.to_numeric(df["match_probability"], errors="coerce").fillna(0.0)

    return df


def predictions_to_matches_map(
    predictions: pd.DataFrame,
    threshold: float = 0.5,
) -> Dict[str, Set[str]]:
    """
    Filter predictions by probability threshold and group by source1_entity_id.

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame with [source1_entity_id, candidate_entity_id, match_probability].
    threshold : float
        Decision threshold (match_probability >= threshold).

    Returns
    -------
    Dict[str, Set[str]]
        Mapping of source1_entity_id to set of predicted matching candidate IDs.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be between 0.0 and 1.0. Given: {threshold}")

    if predictions.empty:
        return {}

    qual = predictions[predictions["match_probability"] >= threshold]
    if qual.empty:
        return {}

    matches_map: Dict[str, Set[str]] = {}
    for s1, cand in zip(qual["source1_entity_id"], qual["candidate_entity_id"]):
        if s1 not in matches_map:
            matches_map[s1] = {cand}
        else:
            matches_map[s1].add(cand)

    return matches_map


def evaluate_matches(
    ground_truth: Dict[str, Set[str]],
    predicted_matches: Dict[str, Set[str]],
    all_s1_ids: Optional[Sequence[str]] = None,
    threshold: Optional[float] = None,
) -> EvaluationReport:
    """
    Evaluate predicted matches against ground truth using the official metric.

    Parameters
    ----------
    ground_truth : Dict[str, Set[str]]
        Ground truth {source1_entity_id -> set_of_true_ids}.
    predicted_matches : Dict[str, Set[str]]
        Predictions {source1_entity_id -> set_of_predicted_ids}.
    all_s1_ids : Optional[Sequence[str]]
        Universe of validation S1 entity IDs. If None, uses sorted keys of ground_truth.
    threshold : Optional[float]
        Optional threshold metadata recorded in the report.

    Returns
    -------
    EvaluationReport
        Detailed evaluation report with macro F_0.5 and diagnostics.
    """
    if not ground_truth:
        raise ValueError("Ground truth dictionary is empty.")

    eval_s1_ids = list(all_s1_ids) if all_s1_ids is not None else sorted(ground_truth.keys())
    total_entities = len(eval_s1_ids)
    if total_entities == 0:
        raise ValueError("Evaluation S1 entity list is empty.")

    entity_scores: List[EntityScore] = []

    f05_total = 0.0
    n_singletons = 0
    singleton_correct = 0
    singleton_fps = 0

    n_matched = 0
    matched_f05_total = 0.0
    matched_prec_total = 0.0
    matched_rec_total = 0.0
    matched_zero_preds = 0

    tot_tp = 0
    tot_fp = 0
    tot_fn = 0

    for s1_id in eval_s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predicted_matches.get(s1_id, set())

        f05, prec, rec, tp, fp, fn = compute_entity_f05(true_set, pred_set)
        f05_total += f05

        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

        is_sing = (len(true_set) == 0)
        sing_corr = False

        if is_sing:
            n_singletons += 1
            if len(pred_set) == 0:
                singleton_correct += 1
                sing_corr = True
            else:
                singleton_fps += 1
        else:
            n_matched += 1
            matched_f05_total += f05
            matched_prec_total += prec
            matched_rec_total += rec
            if len(pred_set) == 0:
                matched_zero_preds += 1

        entity_scores.append(
            EntityScore(
                source1_entity_id=s1_id,
                is_singleton=is_sing,
                true_count=len(true_set),
                pred_count=len(pred_set),
                tp=tp,
                fp=fp,
                fn=fn,
                precision=prec,
                recall=rec,
                f05=f05,
                singleton_correct=sing_corr,
            )
        )

    # Macro averages
    macro_f05 = f05_total / total_entities
    singleton_accuracy = (singleton_correct / n_singletons) if n_singletons > 0 else 1.0

    matched_macro_f05 = (matched_f05_total / n_matched) if n_matched > 0 else 0.0
    matched_macro_prec = (matched_prec_total / n_matched) if n_matched > 0 else 0.0
    matched_macro_rec = (matched_rec_total / n_matched) if n_matched > 0 else 0.0

    # Global Micro diagnostics
    denom_prec = tot_tp + tot_fp
    global_prec = (tot_tp / denom_prec) if denom_prec > 0 else 0.0

    denom_rec = tot_tp + tot_fn
    global_rec = (tot_tp / denom_rec) if denom_rec > 0 else 0.0

    denom_f05 = (0.25 * global_prec) + global_rec
    global_f05 = (1.25 * global_prec * global_rec) / denom_f05 if denom_f05 > 0 else 0.0

    return EvaluationReport(
        macro_f05=macro_f05,
        total_entities=total_entities,
        n_singletons=n_singletons,
        singleton_correct=singleton_correct,
        singleton_accuracy=singleton_accuracy,
        singleton_false_positives=singleton_fps,
        n_matched_entities=n_matched,
        matched_macro_precision=matched_macro_prec,
        matched_macro_recall=matched_macro_rec,
        matched_macro_f05=matched_macro_f05,
        matched_zero_predictions=matched_zero_preds,
        total_tp=tot_tp,
        total_fp=tot_fp,
        total_fn=tot_fn,
        global_precision=global_prec,
        global_recall=global_rec,
        global_f05=global_f05,
        threshold=threshold,
        entity_scores=entity_scores,
    )


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: pd.DataFrame,
    threshold: float = 0.5,
    all_s1_ids: Optional[Sequence[str]] = None,
) -> EvaluationReport:
    """
    Evaluate model predictions at a specific probability threshold.

    Parameters
    ----------
    ground_truth : Dict[str, Set[str]]
        Validation ground truth mapping.
    predictions : pd.DataFrame
        Predictions with columns [source1_entity_id, candidate_entity_id, match_probability].
    threshold : float
        Decision threshold (default 0.5).
    all_s1_ids : Optional[Sequence[str]]
        Universe of validation S1 IDs.
    """
    pred_df = load_predictions(predictions)
    matches_map = predictions_to_matches_map(pred_df, threshold=threshold)
    return evaluate_matches(
        ground_truth=ground_truth,
        predicted_matches=matches_map,
        all_s1_ids=all_s1_ids,
        threshold=threshold,
    )


def evaluate_threshold_grid(
    ground_truth: Dict[str, Set[str]],
    predictions: pd.DataFrame,
    thresholds: Sequence[float],
    all_s1_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Evaluate a sequence of probability thresholds using the same pre-computed model predictions.
    Does not retrain or re-score candidates.

    Parameters
    ----------
    ground_truth : Dict[str, Set[str]]
        Validation ground truth.
    predictions : pd.DataFrame
        Predictions DataFrame.
    thresholds : Sequence[float]
        List or array of thresholds (e.g., [0.2, 0.3, ..., 0.9]).
    all_s1_ids : Optional[Sequence[str]]
        Universe of validation S1 IDs.

    Returns
    -------
    pd.DataFrame
        Comparison table of metrics across all thresholds.
    """
    pred_df = load_predictions(predictions)
    eval_ids = list(all_s1_ids) if all_s1_ids is not None else sorted(ground_truth.keys())

    results = []
    for t in sorted(thresholds):
        report = evaluate_predictions(
            ground_truth=ground_truth,
            predictions=pred_df,
            threshold=float(t),
            all_s1_ids=eval_ids,
        )
        results.append(report.to_dict())

    df_results = pd.DataFrame(results)
    return df_results


def find_best_threshold(
    ground_truth: Dict[str, Set[str]],
    predictions: pd.DataFrame,
    thresholds: Optional[Sequence[float]] = None,
    all_s1_ids: Optional[Sequence[str]] = None,
) -> Tuple[float, EvaluationReport, pd.DataFrame]:
    """
    Find the threshold that maximizes the official Macro F_0.5 score.

    Parameters
    ----------
    ground_truth : Dict[str, Set[str]]
        Validation ground truth.
    predictions : pd.DataFrame
        Model predictions.
    thresholds : Optional[Sequence[float]]
        Threshold candidates. Defaults to np.arange(0.1, 0.95, 0.05).
    all_s1_ids : Optional[Sequence[str]]
        Universe of validation S1 IDs.

    Returns
    -------
    Tuple[float, EvaluationReport, pd.DataFrame]
        (best_threshold, best_report, grid_dataframe)
    """
    if thresholds is None:
        thresholds = [round(x, 3) for x in np.arange(0.10, 0.95, 0.05)]

    grid_df = evaluate_threshold_grid(
        ground_truth=ground_truth,
        predictions=predictions,
        thresholds=thresholds,
        all_s1_ids=all_s1_ids,
    )

    best_idx = grid_df["macro_f05"].idxmax()
    best_thresh = float(grid_df.loc[best_idx, "threshold"])

    best_report = evaluate_predictions(
        ground_truth=ground_truth,
        predictions=predictions,
        threshold=best_thresh,
        all_s1_ids=all_s1_ids,
    )

    return best_thresh, best_report, grid_df


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate entity resolution matching predictions using the official Macro F_0.5 metric."
    )
    parser.add_argument(
        "--ground-truth",
        "-g",
        required=True,
        help="Path to validation ground truth TSV (source1_entity_id, matched_entity_ids).",
    )
    parser.add_argument(
        "--predictions",
        "-p",
        help="Path to predictions TSV/Parquet (source1_entity_id, candidate_entity_id, match_probability).",
    )
    parser.add_argument(
        "--matching-results",
        "-m",
        help="Path to pre-thresholded matching_results.tsv to evaluate directly.",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.5,
        help="Decision threshold for probability scores (default: 0.5).",
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Search for best threshold maximizing official macro F_0.5.",
    )
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()

    print(f"[Evaluation] Loading ground truth from {args.ground_truth}...")
    gt = load_ground_truth(args.ground_truth)
    print(f"[Evaluation] Total ground truth entities: {len(gt):,}")

    if args.matching_results:
        print(f"[Evaluation] Evaluating matching results file: {args.matching_results}...")
        matches_df = pd.read_csv(args.matching_results, sep=DELIM, dtype=str).fillna("")
        matches_map = {
            row["source1_entity_id"].strip(): {
                m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()
            }
            for _, row in matches_df.iterrows()
        }
        report = evaluate_matches(gt, matches_map)
        print(report.summary())

    elif args.predictions:
        print(f"[Evaluation] Loading predictions from {args.predictions}...")
        preds = load_predictions(args.predictions)

        if args.tune_threshold:
            print("[Evaluation] Tuning threshold over [0.10, 0.90] grid...")
            best_t, best_report, grid = find_best_threshold(gt, preds)
            print("\nThreshold Grid Search Results:")
            print(
                grid[
                    [
                        "threshold",
                        "macro_f05",
                        "matched_macro_f05",
                        "singleton_accuracy",
                        "global_precision",
                        "global_recall",
                    ]
                ].to_string(index=False)
            )
            print(f"\nOptimal Threshold: {best_t:.3f}")
            print(best_report.summary())
        else:
            report = evaluate_predictions(gt, preds, threshold=args.threshold)
            print(report.summary())
    else:
        parser.error("Must provide either --predictions or --matching-results.")


if __name__ == "__main__":
    main()
