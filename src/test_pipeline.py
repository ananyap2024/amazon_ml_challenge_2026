"""
ML Challenge 2026 — Pipeline & Orchestration Unit Tests

Covers:
1. Error handling when model inference adapter is not configured (Member2ModelStubAdapter).
2. Pipeline behavior with empty candidates (0 candidates generated).
3. Candidate scoring across multiple batches (batch_size < total_candidates).
4. Preserving all Source 1 IDs, including open-set records (e.g. France) and singletons.
5. Dry-run orchestration and official validator integration.
"""

from pathlib import Path
import tempfile
import unittest
import pandas as pd

from pipeline import (
    CandidateGeneratorAdapter,
    EntityResolutionPipeline,
    Member2ModelStubAdapter,
    ModelInferenceAdapter,
    PipelineConfig,
    PrecomputedCandidateAdapter,
    SyntheticMockModelAdapter,
    run_dry_run,
)
from submission import run_validator


class MockCandidateGenerator(CandidateGeneratorAdapter):
    """Custom generator for unit testing."""

    def __init__(self, candidates_df: pd.DataFrame):
        self.candidates_df = candidates_df

    def generate_candidates(self, s1_ids, data_dir) -> pd.DataFrame:
        return self.candidates_df.copy()


class ConstantModelAdapter(ModelInferenceAdapter):
    """Returns a constant score for every candidate."""

    def __init__(self, score: float = 0.85):
        self.score = score
        self.batches_received = 0

    def predict_batch(self, candidate_batch: pd.DataFrame):
        self.batches_received += 1
        return [self.score] * len(candidate_batch)


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name) / "test"
        self.test_dir.mkdir(parents=True)
        self.output_dir = Path(self.temp_dir.name) / "output"
        self.output_dir.mkdir(parents=True)

        # Create test_source1.tsv with US, India, and France entities
        s1_lines = [
            "entity_id\tbusiness_name\tbusiness_address\tcountry\n",
            "S1-100\tUS Business 1\t100 Main St\tUS\n",
            "S1-200\tIndia Business 1\t200 Park St\tIndia\n",
            "S1-300\tFrance Business 1\t300 Rue Lafayette\tFrance\n",
        ]
        (self.test_dir / "test_source1.tsv").write_text("".join(s1_lines), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_model_stub_raises_not_implemented(self):
        """Ensure unconfigured model inference raises clear NotImplementedError."""
        cand_df = pd.DataFrame([
            {"source1_entity_id": "S1-100", "candidate_entity_id": "S2-500", "source": "source2"}
        ])
        candidate_gen = MockCandidateGenerator(cand_df)
        model_stub = Member2ModelStubAdapter()

        config = PipelineConfig(
            data_dir=self.test_dir,
            output_dir=self.output_dir,
            batch_size=10,
            threshold=0.5,
            skip_validator=True,
        )
        pipeline = EntityResolutionPipeline(candidate_gen, model_stub, config)

        with self.assertRaises(NotImplementedError) as ctx:
            pipeline.run()
        self.assertIn("Member 2's matching model adapter is not yet connected", str(ctx.exception))

    def test_empty_candidates(self):
        """Pipeline must handle 0 candidates cleanly, outputting empty fields for all S1 entities."""
        empty_cands = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "source"])
        candidate_gen = MockCandidateGenerator(empty_cands)
        model_adapter = ConstantModelAdapter(score=0.99)

        config = PipelineConfig(
            data_dir=self.test_dir,
            output_dir=self.output_dir,
            batch_size=10,
            threshold=0.5,
            skip_validator=True,
        )
        pipeline = EntityResolutionPipeline(candidate_gen, model_adapter, config)
        match_p, cand_p = pipeline.run()

        self.assertTrue(match_p.is_file())
        self.assertTrue(cand_p.is_file())

        # All 3 entities should be present with empty match and candidate lists
        cand_content = cand_p.read_text(encoding="utf-8")
        match_content = match_p.read_text(encoding="utf-8")

        for s1 in ["S1-100", "S1-200", "S1-300"]:
            self.assertIn(f"{s1}\t\n", cand_content)
            self.assertIn(f"{s1}\t\n", match_content)

    def test_multiple_batches_and_preserved_ids(self):
        """
        Verify candidate pairs split across multiple batches:
        - 5 candidate pairs with batch_size=2 requires 3 batches.
        - Preserves all Source 1 test IDs in exact order, including France (S1-300).
        """
        cand_df = pd.DataFrame([
            {"source1_entity_id": "S1-100", "candidate_entity_id": "S2-501", "source": "source2"},
            {"source1_entity_id": "S1-100", "candidate_entity_id": "S3-601", "source": "source3"},
            {"source1_entity_id": "S1-200", "candidate_entity_id": "S2-502", "source": "source2"},
            {"source1_entity_id": "S1-200", "candidate_entity_id": "S3-602", "source": "source3"},
            {"source1_entity_id": "S1-200", "candidate_entity_id": "S3-603", "source": "source3"},
            # Notice S1-300 has no candidates (singleton from France)
        ])
        candidate_gen = MockCandidateGenerator(cand_df)
        model_adapter = ConstantModelAdapter(score=0.90)  # > 0.5 threshold -> all candidates match

        config = PipelineConfig(
            data_dir=self.test_dir,
            output_dir=self.output_dir,
            batch_size=2,  # 5 rows / 2 = 3 batches
            threshold=0.5,
            skip_validator=True,
        )
        pipeline = EntityResolutionPipeline(candidate_gen, model_adapter, config)
        match_p, cand_p = pipeline.run()

        self.assertEqual(model_adapter.batches_received, 3)

        cand_df_out = pd.read_csv(cand_p, sep="\t", dtype=str).fillna("")
        match_df_out = pd.read_csv(match_p, sep="\t", dtype=str).fillna("")

        # Verify ordering and coverage
        expected_s1 = ["S1-100", "S1-200", "S1-300"]
        self.assertEqual(cand_df_out["source1_entity_id"].tolist(), expected_s1)
        self.assertEqual(match_df_out["source1_entity_id"].tolist(), expected_s1)

        # France entity must be empty
        self.assertEqual(cand_df_out.loc[cand_df_out["source1_entity_id"] == "S1-300", "candidate_entity_ids"].iloc[0], "")
        self.assertEqual(match_df_out.loc[match_df_out["source1_entity_id"] == "S1-300", "matched_entity_ids"].iloc[0], "")

        # S1-100 has both candidates matched
        self.assertEqual(match_df_out.loc[match_df_out["source1_entity_id"] == "S1-100", "matched_entity_ids"].iloc[0], "S2-501,S3-601")

    def test_dry_run_execution_and_validator(self):
        """Execute run_dry_run and verify that the official validator passes."""
        dry_run_out = Path(self.temp_dir.name) / "dry_run_out"
        match_p, cand_p = run_dry_run(
            output_dir=dry_run_out,
            batch_size=2,
            threshold=0.5,
        )
        self.assertTrue(match_p.is_file())
        self.assertTrue(cand_p.is_file())

        ret = run_validator(
            matching_path=match_p,
            candidate_path=cand_p,
            test_dir=dry_run_out / "synthetic_test_data",
        )
        self.assertEqual(ret, 0, "Validator failed on dry-run output.")

    def test_precomputed_candidate_adapter_chunking(self):
        """
        Verify multi-chunk incremental candidate streaming:
        - Candidate file is read in small chunks (chunk_size=2).
        - Candidate pairs for an S1 entity cross chunk boundaries.
        - Probability association is preserved per candidate pair.
        - Entities with 0 candidates (S1-300) are preserved with empty fields.
        - Matched IDs are guaranteed to be a subset of candidate IDs.
        """
        # Write candidate file with 6 candidate pairs
        cand_tsv = self.test_dir / "test_candidates.tsv"
        cand_rows = [
            "source1_entity_id\tcandidate_entity_id\tsource\n",
            "S1-100\tS2-501\tsource2\n",  # Chunk 1
            "S1-100\tS3-601\tsource3\n",
            "S1-100\tS2-502\tsource2\n",  # Chunk 2 (S1-100 continues across chunk boundary)
            "S1-200\tS2-503\tsource2\n",
            "S1-200\tS3-604\tsource3\n",  # Chunk 3
            "S1-100\tS2-505\tsource2\n",  # S1-100 appears in Chunk 3 again
        ]
        cand_tsv.write_text("".join(cand_rows), encoding="utf-8")

        candidate_gen = PrecomputedCandidateAdapter(cand_tsv)

        # Map known scores to verify exact pair-probability association
        score_lookup = {
            ("S1-100", "S2-501"): 0.95,  # Match (>= 0.5)
            ("S1-100", "S3-601"): 0.30,  # Below threshold
            ("S1-100", "S2-502"): 0.85,  # Match (>= 0.5)
            ("S1-200", "S2-503"): 0.20,  # Below threshold
            ("S1-200", "S3-604"): 0.75,  # Match (>= 0.5)
            ("S1-100", "S2-505"): 0.10,  # Below threshold
        }

        class MapModelAdapter(ModelInferenceAdapter):
            def __init__(self):
                self.chunks_received = 0

            def predict_batch(self, candidate_batch: pd.DataFrame):
                self.chunks_received += 1
                return [
                    score_lookup[(str(row["source1_entity_id"]), str(row["candidate_entity_id"]))]
                    for _, row in candidate_batch.iterrows()
                ]

        model_adapter = MapModelAdapter()

        config = PipelineConfig(
            data_dir=self.test_dir,
            output_dir=self.output_dir,
            batch_size=2,  # 6 pairs / 2 = 3 chunks
            threshold=0.5,
            skip_validator=True,
        )

        pipeline = EntityResolutionPipeline(candidate_gen, model_adapter, config)
        match_p, cand_p = pipeline.run()

        self.assertEqual(model_adapter.chunks_received, 3)

        cand_df = pd.read_csv(cand_p, sep="\t", dtype=str).fillna("")
        match_df = pd.read_csv(match_p, sep="\t", dtype=str).fillna("")

        # Verify all S1 IDs exist in exact order
        self.assertEqual(cand_df["source1_entity_id"].tolist(), ["S1-100", "S1-200", "S1-300"])
        self.assertEqual(match_df["source1_entity_id"].tolist(), ["S1-100", "S1-200", "S1-300"])

        # S1-100: candidates from Chunk 1, Chunk 2, Chunk 3 all preserved in order
        s1_100_cands = cand_df.loc[cand_df["source1_entity_id"] == "S1-100", "candidate_entity_ids"].iloc[0]
        self.assertEqual(s1_100_cands, "S2-501,S3-601,S2-502,S2-505")

        # S1-100 matches: only those with prob >= 0.5
        s1_100_matches = match_df.loc[match_df["source1_entity_id"] == "S1-100", "matched_entity_ids"].iloc[0]
        self.assertEqual(s1_100_matches, "S2-501,S2-502")

        # S1-200
        s1_200_cands = cand_df.loc[cand_df["source1_entity_id"] == "S1-200", "candidate_entity_ids"].iloc[0]
        self.assertEqual(s1_200_cands, "S2-503,S3-604")
        s1_200_matches = match_df.loc[match_df["source1_entity_id"] == "S1-200", "matched_entity_ids"].iloc[0]
        self.assertEqual(s1_200_matches, "S3-604")

        # S1-300: 0 candidates -> empty fields
        s1_300_cands = cand_df.loc[cand_df["source1_entity_id"] == "S1-300", "candidate_entity_ids"].iloc[0]
        self.assertEqual(s1_300_cands, "")
        s1_300_matches = match_df.loc[match_df["source1_entity_id"] == "S1-300", "matched_entity_ids"].iloc[0]
        self.assertEqual(s1_300_matches, "")

    def test_duplicate_pairs_across_chunks(self):
        """
        Verify duplicate candidate pairs across chunk boundaries:
        - Exact same pair (S1-100, S2-501) arrives in Chunk 1 and Chunk 2.
        - Deduplication must keep first appearance order.
        - Neither candidate_entity_ids nor matched_entity_ids contains duplicate entries.
        """
        cand_tsv = self.test_dir / "duplicate_cands.tsv"
        cand_rows = [
            "source1_entity_id\tcandidate_entity_id\tsource\n",
            "S1-100\tS2-501\tsource2\n",  # Chunk 1
            "S1-100\tS2-502\tsource2\n",
            "S1-100\tS2-501\tsource2\n",  # Chunk 2 (Duplicate of S2-501)
            "S1-100\tS2-503\tsource2\n",
        ]
        cand_tsv.write_text("".join(cand_rows), encoding="utf-8")

        candidate_gen = PrecomputedCandidateAdapter(cand_tsv)
        model_adapter = ConstantModelAdapter(score=0.90)  # All are matches

        config = PipelineConfig(
            data_dir=self.test_dir,
            output_dir=self.output_dir,
            batch_size=2,
            threshold=0.5,
            skip_validator=True,
        )
        pipeline = EntityResolutionPipeline(candidate_gen, model_adapter, config)
        match_p, cand_p = pipeline.run()

        cand_df = pd.read_csv(cand_p, sep="\t", dtype=str).fillna("")
        match_df = pd.read_csv(match_p, sep="\t", dtype=str).fillna("")

        s1_100_cands = cand_df.loc[cand_df["source1_entity_id"] == "S1-100", "candidate_entity_ids"].iloc[0]
        s1_100_matches = match_df.loc[match_df["source1_entity_id"] == "S1-100", "matched_entity_ids"].iloc[0]

        # S2-501 must appear only ONCE, in first position
        self.assertEqual(s1_100_cands, "S2-501,S2-502,S2-503")
        self.assertEqual(s1_100_matches, "S2-501,S2-502,S2-503")

    def test_large_candidate_count_per_entity_streaming(self):
        """
        Verify that a single S1 entity with a large number of candidates (5,000 pairs)
        is processed via chunked streaming without building a huge in-memory DataFrame.
        """
        total_cands = 5000
        chunk_size = 250

        class StreamingLargeCandidateGenerator(CandidateGeneratorAdapter):
            def generate_candidates(self, s1_ids, data_dir):
                raise NotImplementedError("Use iter_candidates")

            def iter_candidates(self, s1_ids, data_dir, chunk_size=250):
                # Yields 20 small chunks of 250 rows each
                for start in range(0, total_cands, chunk_size):
                    chunk_rows = [
                        {
                            "source1_entity_id": "S1-100",
                            "candidate_entity_id": f"S2-{idx}",
                            "source": "source2",
                        }
                        for idx in range(start, start + chunk_size)
                    ]
                    yield pd.DataFrame(chunk_rows)

        candidate_gen = StreamingLargeCandidateGenerator()
        # Half match (idx % 2 == 0)
        class AlternatingModelAdapter(ModelInferenceAdapter):
            def predict_batch(self, candidate_batch: pd.DataFrame):
                scores = []
                for _, row in candidate_batch.iterrows():
                    cand = str(row["candidate_entity_id"])
                    idx = int(cand.split("-")[1])
                    scores.append(0.95 if idx % 2 == 0 else 0.10)
                return scores

        config = PipelineConfig(
            data_dir=self.test_dir,
            output_dir=self.output_dir,
            batch_size=chunk_size,
            threshold=0.5,
            skip_validator=True,
        )
        pipeline = EntityResolutionPipeline(candidate_gen, AlternatingModelAdapter(), config)
        match_p, cand_p = pipeline.run()

        cand_df = pd.read_csv(cand_p, sep="\t", dtype=str).fillna("")
        match_df = pd.read_csv(match_p, sep="\t", dtype=str).fillna("")

        s1_100_cands = cand_df.loc[cand_df["source1_entity_id"] == "S1-100", "candidate_entity_ids"].iloc[0].split(",")
        s1_100_matches = match_df.loc[match_df["source1_entity_id"] == "S1-100", "matched_entity_ids"].iloc[0].split(",")

        self.assertEqual(len(s1_100_cands), total_cands)
        self.assertEqual(s1_100_cands[0], "S2-0")
        self.assertEqual(s1_100_cands[-1], f"S2-{total_cands - 1}")

        self.assertEqual(len(s1_100_matches), total_cands // 2)
        self.assertEqual(s1_100_matches[0], "S2-0")
        self.assertEqual(s1_100_matches[-1], f"S2-{total_cands - 2}")

        # Entities with 0 candidates remain empty
        self.assertEqual(cand_df.loc[cand_df["source1_entity_id"] == "S1-300", "candidate_entity_ids"].iloc[0], "")
        self.assertEqual(match_df.loc[match_df["source1_entity_id"] == "S1-300", "matched_entity_ids"].iloc[0], "")

    def test_invalid_probabilities_rejected(self):
        """Pipeline must reject NaN, inf, and probabilities outside [0, 1]."""
        cand_df = pd.DataFrame([
            {"source1_entity_id": "S1-100", "candidate_entity_id": "S2-501", "source": "source2"}
        ])
        candidate_gen = MockCandidateGenerator(cand_df)

        for invalid_prob in [float("nan"), float("inf"), float("-inf"), -0.05, 1.05]:
            with self.subTest(invalid_prob=invalid_prob):
                class BadProbModel(ModelInferenceAdapter):
                    def predict_batch(self, candidate_batch):
                        return [invalid_prob] * len(candidate_batch)

                config = PipelineConfig(
                    data_dir=self.test_dir,
                    output_dir=self.output_dir,
                    batch_size=10,
                    threshold=0.5,
                    skip_validator=True,
                )
                pipeline = EntityResolutionPipeline(candidate_gen, BadProbModel(), config)
                with self.assertRaises(ValueError) as ctx:
                    pipeline.run()
                self.assertIn("Invalid match probability", str(ctx.exception))

    def test_illegal_delimiter_characters_rejected(self):
        """Entities containing tab, newline, or comma must be rejected."""
        # Comma in candidate ID
        bad_df = pd.DataFrame([
            {"source1_entity_id": "S1-100", "candidate_entity_id": "S2-501,bad", "source": "source2"}
        ])
        candidate_gen = MockCandidateGenerator(bad_df)
        config = PipelineConfig(
            data_dir=self.test_dir,
            output_dir=self.output_dir,
            batch_size=10,
            threshold=0.5,
            skip_validator=True,
        )
        pipeline = EntityResolutionPipeline(candidate_gen, ConstantModelAdapter(0.9), config)
        with self.assertRaises(ValueError) as ctx:
            pipeline.run()
        self.assertIn("illegal delimiter characters", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
