"""
ML Challenge 2026 — Main Execution Entrypoint (Member 3: Pipeline, Integration & Submission)

Provides the official CLI for:
1. Dry-run synthetic validation (tests full orchestration without touching full dataset).
2. End-to-end inference over test data once Member 1 & Member 2 adapters are connected.

Usage Examples:
---------------
1. Run synthetic dry-run:
   python src/main.py --dry-run

2. Run real test inference (once model adapter is plugged in):
   python src/main.py --data-dir dataset/test --output-dir output --batch-size 25000 --threshold 0.5

3. Run real test inference with precomputed candidate pairs:
   python src/main.py --data-dir dataset/test --candidates output/candidate_pairs.tsv --output-dir output
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from pipeline import (
    CandidateGeneratorAdapter,
    EntityResolutionPipeline,
    Member2ModelStubAdapter,
    Member2XGBoost13FeatureAdapter,
    ModelInferenceAdapter,
    PipelineConfig,
    PrecomputedCandidateAdapter,
    RecordCatalog,
    V3BlockingInProcessAdapter,
    run_dry_run,
)


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 — Entity Resolution End-to-End Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        default="dataset/test",
        help="Path to folder containing test_source1.tsv, test_source2.tsv, test_source3.tsv (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="output",
        help="Directory to save matching_results.tsv and candidate_pairs.tsv (default: %(default)s)",
    )
    parser.add_argument(
        "--candidates",
        "-c",
        default=None,
        help="Path to precomputed candidate pairs file (TSV/Parquet). If omitted, runs in-process candidate generation.",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=5_000,
        help="Number of candidate pairs to score per batch during inference (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.60,
        help="PROVISIONAL classification probability threshold from Member 2 (default: %(default)s)",
    )
    parser.add_argument(
        "--model-path",
        "-m",
        default="code/business_entity_resolution/models/matching_model.pkl",
        help="Path to Member 2's trained 13-feature XGBoost model pickle (default: %(default)s)",
    )
    parser.add_argument(
        "--vectorizers-path",
        default="code/business_entity_resolution/models/tfidf_vectorizers.pkl",
        help="Path to combined TF-IDF vectorizers artifact (default: %(default)s)",
    )
    parser.add_argument(
        "--name-vectorizer",
        default=None,
        help="Path to separate fitted TF-IDF name vectorizer artifact",
    )
    parser.add_argument(
        "--address-vectorizer",
        default=None,
        help="Path to separate fitted TF-IDF address vectorizer artifact",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Custom scratch directory for temporary SQLite database (useful to point to a drive with large free space)",
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help="Explicit path for SQLite accumulator database (default: temporary file in temp-dir)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run synthetic dry-run to test end-to-end orchestration without loading full dataset",
    )
    parser.add_argument(
        "--skip-validator",
        action="store_true",
        help="Skip executing utils/validate_submission.py after generation",
    )
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    if args.dry_run:
        # Dry-run synthetic execution
        dry_run_out = Path(args.output_dir) / "dry_run" if args.output_dir == "output" else Path(args.output_dir)
        matching_path, candidate_path = run_dry_run(
            output_dir=dry_run_out,
            batch_size=min(args.batch_size, 5),
            threshold=args.threshold,
        )
        print(f"\n[Dry-Run Succeeded] Verified pipeline wiring safely without touching large test dataset.")
        return

    # Real Inference Setup
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not data_dir.is_dir():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    s1_file = data_dir / "test_source1.tsv"
    if not s1_file.is_file():
        print(f"Error: test_source1.tsv not found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Select Candidate Generator Adapter
    if args.candidates:
        print(f"[Main] Using precomputed candidate pairs: {args.candidates}")
        candidate_generator: CandidateGeneratorAdapter = PrecomputedCandidateAdapter(args.candidates)
    else:
        print(f"[Main] Using Member 1 V3 blocking candidate generator...")
        candidate_generator = V3BlockingInProcessAdapter()

    # Select Model Inference Adapter
    model_path = Path(args.model_path)
    if not model_path.is_file():
        fallback_path = Path("student_resource/dataset/train/analysis_model_xgboost_13features_realistic.pkl")
        if fallback_path.is_file():
            model_path = fallback_path

    if model_path.is_file():
        print(f"[Main] Connecting Member 2 13-feature XGBoost model from: {model_path}")
        print(f"[Main] Loading business record catalog from {data_dir}...")
        catalog = RecordCatalog.from_directory(data_dir=data_dir, split="test")
        model_inference: ModelInferenceAdapter = Member2XGBoost13FeatureAdapter(
            model_path=model_path,
            vectorizers_path=args.vectorizers_path,
            name_vectorizer_path=args.name_vectorizer,
            address_vectorizer_path=args.address_vectorizer,
            records_catalog=catalog,
        )
    else:
        print(f"[Main Warning] Model artifact not found at {args.model_path}.")
        model_inference = Member2ModelStubAdapter()

    config = PipelineConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        threshold=args.threshold,
        skip_validator=args.skip_validator,
        temp_dir=Path(args.temp_dir) if args.temp_dir else None,
        db_path=Path(args.db_path) if args.db_path else None,
    )

    pipeline = EntityResolutionPipeline(
        candidate_generator=candidate_generator,
        model_inference=model_inference,
        config=config,
    )

    try:
        pipeline.run()
    except NotImplementedError as exc:
        print("\n" + "!" * 70, file=sys.stderr)
        print("PIPELINE STOPPED: PENDING COMPONENT", file=sys.stderr)
        print("!" * 70, file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print("!" * 70, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
