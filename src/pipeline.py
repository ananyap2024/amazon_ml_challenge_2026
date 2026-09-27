"""
ML Challenge 2026 — Orchestration Pipeline (Member 3: Pipeline, Integration & Submission)

Connects:
1. Test Source 1 Entity Loading (preserving IDs as strings, supporting open-set France/US/India).
2. Candidate Generation via CandidateGeneratorAdapter (using Member 1's blocking logic or precomputed candidates).
3. Batched Model Inference via ModelInferenceAdapter (streaming batches to prevent Cartesian explosion).
4. Submission Serialization via submission.py (strict official schema, empty singletons, deduplication).
5. Validation Gate via utils/validate_submission.py.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

# Internal dependencies from Member 3 modules
from submission import (
    DELIM,
    DiskBackedSubmissionAccumulator,
    build_submission_dataframes,
    load_source1_ids,
    run_validator,
    validate_candidate_pairs_dataframe,
    validate_predictions_dataframe,
    write_submission_tsv,
)
from features import (
    FEATURE_NAMES_13,
    Member2XGBoost13FeatureAdapter,
    RecordCatalog,
    compute_13_features_for_batch,
)

# Member 1 blocking components
try:
    from blocking_v3 import (
        build_condensed_name_index,
        build_exact_name_index,
        build_inverted_index,
        build_token_frequency,
        lookup_channel_candidates,
        prepare_dataframe_v3,
    )
    BLOCKING_V3_AVAILABLE = True
except ImportError:
    BLOCKING_V3_AVAILABLE = False


# =============================================================================
# 1. Candidate Generation Adapters (Member 1 Integration)
# =============================================================================

class CandidateGeneratorAdapter(abc.ABC):
    """
    Abstract interface for candidate pair generation.
    Must return DataFrame with columns: ['source1_entity_id', 'candidate_entity_id', 'source']
    where source is 'source2' or 'source3'.
    """

    @abc.abstractmethod
    def generate_candidates(
        self,
        s1_ids: Sequence[str],
        data_dir: Path,
    ) -> pd.DataFrame:
        """Generate candidate pairs for given Source 1 entity IDs."""
        pass

    def iter_candidates(
        self,
        s1_ids: Sequence[str],
        data_dir: Path,
        chunk_size: int = 25_000,
    ) -> Iterator[pd.DataFrame]:
        """
        Stream candidate pairs in batches to bound memory usage.
        Default implementation calls generate_candidates and slices in memory.
        """
        df = self.generate_candidates(s1_ids, data_dir)
        if df.empty:
            return
        n_rows = len(df)
        chunk_size = max(1, chunk_size)
        for start_idx in range(0, n_rows, chunk_size):
            yield df.iloc[start_idx : start_idx + chunk_size]


class PrecomputedCandidateAdapter(CandidateGeneratorAdapter):
    """Loads pre-generated candidate pairs from an existing TSV or Parquet file."""

    def __init__(self, candidate_file: Union[str, Path]):
        self.candidate_file = Path(candidate_file)
        if not self.candidate_file.is_file():
            raise FileNotFoundError(f"Candidate file not found: {self.candidate_file}")

    def generate_candidates(
        self,
        s1_ids: Sequence[str],
        data_dir: Path,
    ) -> pd.DataFrame:
        if self.candidate_file.suffix == ".parquet":
            df = pd.read_parquet(self.candidate_file)
        else:
            df = pd.read_csv(
                self.candidate_file,
                sep=DELIM,
                dtype={"source1_entity_id": str, "candidate_entity_id": str, "source": str},
            )
        validate_candidate_pairs_dataframe(df)
        return df

    def iter_candidates(
        self,
        s1_ids: Sequence[str],
        data_dir: Path,
        chunk_size: int = 25_000,
    ) -> Iterator[pd.DataFrame]:
        """
        Stream candidate pairs from disk in chunks without loading the entire file into RAM.
        """
        chunk_size = max(1, chunk_size)
        if self.candidate_file.suffix == ".parquet":
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(self.candidate_file)
            for batch in pf.iter_batches(batch_size=chunk_size):
                chunk_df = batch.to_pandas()
                validate_candidate_pairs_dataframe(chunk_df)
                yield chunk_df
        else:
            for chunk_df in pd.read_csv(
                self.candidate_file,
                sep=DELIM,
                chunksize=chunk_size,
                dtype={"source1_entity_id": str, "candidate_entity_id": str, "source": str},
            ):
                validate_candidate_pairs_dataframe(chunk_df)
                yield chunk_df


class V3BlockingInProcessAdapter(CandidateGeneratorAdapter):
    """
    In-process candidate generator using Member 1's blocking_v3 module.
    Ideal for dry runs, smaller splits, or when candidate generation runs within the pipeline.
    """

    def __init__(
        self,
        max_name_tokens: int = 3,
        max_name_freq: int = 1500,
        max_addr_tokens: int = 2,
        max_addr_freq: int = 250,
    ):
        if not BLOCKING_V3_AVAILABLE:
            raise ImportError(
                "blocking_v3 module is required for V3BlockingInProcessAdapter but could not be imported."
            )
        self.max_name_tokens = max_name_tokens
        self.max_name_freq = max_name_freq
        self.max_addr_tokens = max_addr_tokens
        self.max_addr_freq = max_addr_freq

    def generate_candidates_from_dataframes(
        self,
        s1_df: pd.DataFrame,
        s2_df: pd.DataFrame,
        s3_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run blocking V3 over given DataFrames and format as candidate pairs."""
        # Normalize tables
        s1_prep = prepare_dataframe_v3(s1_df)
        s2_prep = prepare_dataframe_v3(s2_df)
        s3_prep = prepare_dataframe_v3(s3_df)

        # Build S2 indices
        s2_exact = build_exact_name_index(s2_prep)
        s2_condensed = build_condensed_name_index(s2_prep)
        s2_name_tok = build_inverted_index(s2_prep, "name_tokens")
        s2_name_freq = build_token_frequency(s2_name_tok)
        s2_addr_tok = build_inverted_index(s2_prep, "address_tokens")
        s2_addr_freq = build_token_frequency(s2_addr_tok)

        # Build S3 indices
        s3_exact = build_exact_name_index(s3_prep)
        s3_condensed = build_condensed_name_index(s3_prep)
        s3_name_tok = build_inverted_index(s3_prep, "name_tokens")
        s3_name_freq = build_token_frequency(s3_name_tok)
        s3_addr_tok = build_inverted_index(s3_prep, "address_tokens")
        s3_addr_freq = build_token_frequency(s3_addr_tok)

        records: List[Dict[str, str]] = []

        for _, row in s1_prep.iterrows():
            s1_id = str(row["entity_id"]).strip()

            # Lookup in S2
            c2_set, _ = lookup_channel_candidates(
                row,
                exact_index=s2_exact,
                condensed_index=s2_condensed,
                name_token_index=s2_name_tok,
                name_token_freq=s2_name_freq,
                addr_token_index=s2_addr_tok,
                addr_token_freq=s2_addr_freq,
                max_name_tokens=self.max_name_tokens,
                max_name_freq=self.max_name_freq,
                max_addr_tokens=self.max_addr_tokens,
                max_addr_freq=self.max_addr_freq,
            )
            for c2 in c2_set:
                records.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": str(c2).strip(),
                    "source": "source2",
                })

            # Lookup in S3
            c3_set, _ = lookup_channel_candidates(
                row,
                exact_index=s3_exact,
                condensed_index=s3_condensed,
                name_token_index=s3_name_tok,
                name_token_freq=s3_name_freq,
                addr_token_index=s3_addr_tok,
                addr_token_freq=s3_addr_freq,
                max_name_tokens=self.max_name_tokens,
                max_name_freq=self.max_name_freq,
                max_addr_tokens=self.max_addr_tokens,
                max_addr_freq=self.max_addr_freq,
            )
            for c3 in c3_set:
                records.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": str(c3).strip(),
                    "source": "source3",
                })

        df = pd.DataFrame(records, columns=["source1_entity_id", "candidate_entity_id", "source"])
        validate_candidate_pairs_dataframe(df)
        return df

    def generate_candidates(
        self,
        s1_ids: Sequence[str],
        data_dir: Path,
    ) -> pd.DataFrame:
        """
        Loads source files from data_dir and generates candidates.
        Note: on full test dataset (10M+ records), running full in-memory blocking
        can require 16GB+ RAM. For full-scale inference, use PrecomputedCandidateAdapter.
        """
        s1_file = data_dir / "test_source1.tsv"
        s2_file = data_dir / "test_source2.tsv"
        s3_file = data_dir / "test_source3.tsv"

        s1_df = pd.read_csv(s1_file, sep=DELIM, dtype=str)
        s2_df = pd.read_csv(s2_file, sep=DELIM, dtype=str)
        s3_df = pd.read_csv(s3_file, sep=DELIM, dtype=str)

        return self.generate_candidates_from_dataframes(s1_df, s2_df, s3_df)


# =============================================================================
# 2. Model Inference Adapters (Member 2 Integration)
# =============================================================================

class ModelInferenceAdapter(abc.ABC):
    """
    Abstract interface for model candidate scoring.
    Given a batch of candidate pairs, outputs match probability in [0.0, 1.0].
    """

    @abc.abstractmethod
    def predict_batch(self, candidate_batch: pd.DataFrame) -> Sequence[float]:
        """
        Score a batch of candidate pairs.

        Parameters
        ----------
        candidate_batch : pd.DataFrame
            Columns: ['source1_entity_id', 'candidate_entity_id', 'source']

        Returns
        -------
        Sequence[float]
            Array or list of probability scores corresponding to rows in candidate_batch.
        """
        pass


class Member2ModelStubAdapter(ModelInferenceAdapter):
    """
    Explicit stub indicating Member 2's matching model is not yet connected.
    Raises a helpful NotImplementedError rather than generating fake predictions.
    """

    def predict_batch(self, candidate_batch: pd.DataFrame) -> Sequence[float]:
        raise NotImplementedError(
            "Member 2's matching model adapter is not yet connected.\n"
            "Please plug in Member 2's classifier (e.g., CatBoost / XGBoost / Siamese model)\n"
            "by subclassing ModelInferenceAdapter and implementing `predict_batch(candidate_batch)`."
        )


class SyntheticMockModelAdapter(ModelInferenceAdapter):
    """
    Deterministic scoring adapter used ONLY for pipeline dry-runs and integration testing.
    Assigns fixed scores based on simple string equality to allow verifying the pipeline flow.
    """

    def __init__(self, high_score: float = 0.95, low_score: float = 0.20):
        self.high_score = high_score
        self.low_score = low_score

    def predict_batch(self, candidate_batch: pd.DataFrame) -> Sequence[float]:
        # Return high score if ID numbers match or simple heuristic for dry run
        scores = []
        for _, row in candidate_batch.iterrows():
            s1 = str(row["source1_entity_id"])
            c = str(row["candidate_entity_id"])
            # Sample rule: if numeric suffix ends in same digit or contains common token -> high
            s1_num = "".join(filter(str.isdigit, s1))
            c_num = "".join(filter(str.isdigit, c))
            if s1_num and c_num and s1_num[-1] == c_num[-1]:
                scores.append(self.high_score)
            else:
                scores.append(self.low_score)
        return scores


# Register Member 2's 13-feature adapter as implementing ModelInferenceAdapter
ModelInferenceAdapter.register(Member2XGBoost13FeatureAdapter)

# =============================================================================
# 3. Main Pipeline Orchestrator
# =============================================================================

# PROVISIONAL: Uncalibrated threshold reported by Member 2; requires validation tuning before final submission
PROVISIONAL_DEFAULT_THRESHOLD: float = 0.60


@dataclass
class PipelineConfig:
    data_dir: Path
    output_dir: Path
    batch_size: int = 5_000
    threshold: float = PROVISIONAL_DEFAULT_THRESHOLD
    skip_validator: bool = False
    temp_dir: Optional[Path] = None
    db_path: Optional[Path] = None


class EntityResolutionPipeline:
    """
    Orchestrates the complete Entity Resolution process:
    1. Loads all Source 1 test IDs (preserving IDs as strings, supporting France/US/India).
    2. Generates candidate pairs via CandidateGeneratorAdapter.
    3. Scores candidates in batches via ModelInferenceAdapter (avoids Cartesian products).
    4. Writes official submission files (output/matching_results.tsv and output/candidate_pairs.tsv).
    5. Validates outputs via utils/validate_submission.py.
    """

    def __init__(
        self,
        candidate_generator: CandidateGeneratorAdapter,
        model_inference: ModelInferenceAdapter,
        config: PipelineConfig,
    ):
        self.candidate_generator = candidate_generator
        self.model_inference = model_inference
        self.config = config

    def run(self) -> Tuple[Path, Path]:
        """Execute the end-to-end pipeline and return paths to generated TSVs."""
        s1_file = self.config.data_dir / "test_source1.tsv"
        print(f"[Pipeline] Step 1: Loading Source 1 test IDs from {s1_file}...")
        all_s1_ids = load_source1_ids(s1_file)
        print(f"[Pipeline] Loaded {len(all_s1_ids):,} Source 1 test entities.")

        print(
            f"[Pipeline] Step 2 & 3: Streaming candidate chunks & scoring (batch_size={self.config.batch_size:,})..."
        )
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        matching_path = self.config.output_dir / "matching_results.tsv"
        candidate_path = self.config.output_dir / "candidate_pairs.tsv"

        accumulator = DiskBackedSubmissionAccumulator(
            db_path=self.config.db_path,
            temp_dir=self.config.temp_dir,
        )
        total_candidate_pairs = 0
        n_chunks = 0

        try:
            for chunk in self.candidate_generator.iter_candidates(
                s1_ids=all_s1_ids,
                data_dir=self.config.data_dir,
                chunk_size=self.config.batch_size,
            ):
                n_chunks += 1
                total_candidate_pairs += len(chunk)

                # Validate chunk schema and prefixes
                validate_candidate_pairs_dataframe(chunk)

                # Score batch with model inference
                probs = self.model_inference.predict_batch(chunk)
                if len(probs) != len(chunk):
                    raise ValueError(
                        f"Model inference returned {len(probs)} scores for a batch of {len(chunk)} candidates."
                    )

                accumulator.add_chunk(
                    candidate_chunk=chunk,
                    probs=probs,
                    threshold=self.config.threshold,
                )

            print(
                f"[Pipeline] Processed {n_chunks:,} chunks ({total_candidate_pairs:,} total candidate pairs)."
            )
            print(f"[Pipeline] Step 4: Writing submission files from accumulator...")
            accumulator.write_outputs(
                all_source1_ids=all_s1_ids,
                matching_path=matching_path,
                candidate_path=candidate_path,
            )
            print(f"[Pipeline] Wrote matching results: {matching_path}")
            print(f"[Pipeline] Wrote candidate pairs:  {candidate_path}")
        finally:
            accumulator.close()

        if not self.config.skip_validator:
            print("[Pipeline] Step 5: Running official submission validator...")
            ret = run_validator(
                matching_path=matching_path,
                candidate_path=candidate_path,
                test_dir=self.config.data_dir,
            )
            if ret != 0:
                print(f"[Pipeline Warning] Validator returned code {ret}.")

        return matching_path, candidate_path

    def _score_candidates_batched(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Score candidate pairs in configurable batches to maintain bounded memory footprint."""
        if candidates_df.empty:
            return pd.DataFrame(
                columns=["source1_entity_id", "candidate_entity_id", "match_probability"]
            )

        n_pairs = len(candidates_df)
        batch_size = max(1, self.config.batch_size)
        all_probs: List[float] = []

        for start_idx in range(0, n_pairs, batch_size):
            end_idx = min(start_idx + batch_size, n_pairs)
            batch = candidates_df.iloc[start_idx:end_idx]

            # Model inference on this batch
            probs = self.model_inference.predict_batch(batch)
            if len(probs) != len(batch):
                raise ValueError(
                    f"Model inference returned {len(probs)} scores for a batch of {len(batch)} candidates."
                )
            all_probs.extend(probs)

        pred_df = pd.DataFrame({
            "source1_entity_id": candidates_df["source1_entity_id"].astype(str),
            "candidate_entity_id": candidates_df["candidate_entity_id"].astype(str),
            "match_probability": all_probs,
        })
        validate_predictions_dataframe(pred_df)
        return pred_df


# =============================================================================
# 4. Dry-Run Orchestration (Synthetic Small Dataset)
# =============================================================================

def run_dry_run(
    output_dir: Path = Path("output") / "dry_run",
    batch_size: int = 2,
    threshold: float = 0.5,
) -> Tuple[Path, Path]:
    """
    Run an end-to-end synthetic dry-run to verify pipeline wiring.
    Uses synthetic in-memory data (with records from US, India, and France)
    without touching the full dataset, writing to output/dry_run/.
    """
    print("=" * 70)
    print("RUNNING PIPELINE DRY-RUN (SYNTHETIC VERIFICATION ONLY)")
    print("NOT A REAL SUBMISSION — TESTS INTEGRATION WIRING")
    print("=" * 70)

    dry_run_dir = output_dir
    dry_run_dir.mkdir(parents=True, exist_ok=True)
    synthetic_test_dir = dry_run_dir / "synthetic_test_data"
    synthetic_test_dir.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic test dataset
    s1_rows = [
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n",
        "S1-1001\tApex Logistics LLC\t100 Industry Way\tUS\n",
        "S1-1002\tApex Logistics Co\t100 Industry Way\tUS\n",
        "S1-1003\tBharat Enterprises Pvt Ltd\tPlot 45 MIDC Area\tIndia\n",
        "S1-1004\tChâteau Vignoble SARL\t12 Rue de Paris\tFrance\n",  # France entity (singleton)
    ]
    s2_rows = [
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n",
        "S2-2001\tApex Logistics Inc\t100 Industry Way\tUS\n",
        "S2-2002\tBharat Enterprises\tPlot 45 MIDC Area\tIndia\n",
    ]
    s3_rows = [
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n",
        "S3-3001\tApex Logistics Solutions\t100 Industry Way\tUS\n",
        "S3-3003\tZeta Tech Ltd\t888 Cyber City\tIndia\n",
    ]

    (synthetic_test_dir / "test_source1.tsv").write_text("".join(s1_rows), encoding="utf-8")
    (synthetic_test_dir / "test_source2.tsv").write_text("".join(s2_rows), encoding="utf-8")
    (synthetic_test_dir / "test_source3.tsv").write_text("".join(s3_rows), encoding="utf-8")

    # 2. Configure Pipeline with V3 blocking + Synthetic mock model
    candidate_generator = V3BlockingInProcessAdapter()
    model_inference = SyntheticMockModelAdapter(high_score=0.90, low_score=0.20)

    config = PipelineConfig(
        data_dir=synthetic_test_dir,
        output_dir=dry_run_dir,
        batch_size=batch_size,
        threshold=threshold,
        skip_validator=False,
    )

    pipeline = EntityResolutionPipeline(
        candidate_generator=candidate_generator,
        model_inference=model_inference,
        config=config,
    )

    matching_path, candidate_path = pipeline.run()

    print("\n[Dry-Run Complete] Output files generated:")
    print(f"  {matching_path}")
    print(f"  {candidate_path}")
    print("=" * 70)
    return matching_path, candidate_path
