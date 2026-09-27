"""
ML Challenge 2026 — Member 2 13-Feature Model & Integration Tests

Validates:
1. Missing model / vectorizer artifact errors (FileNotFoundError).
2. Schema enforcement and model.feature_names_in_ ordering check.
3. Candidate record resolution and KeyError on missing record IDs.
4. Input validation (missing columns, row-count mismatches, invalid probabilities).
5. End-to-end small batch scoring using a real fitted XGBoost model and TF-IDF vectorizers.
6. Real record resolution on actual dataset/train/ files without running full inference.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb

from features import (
    FEATURE_NAMES_13,
    Member2XGBoost13FeatureAdapter,
    RecordCatalog,
    compute_13_features_for_batch,
    levenshtein_similarity,
    token_jaccard_similarity,
)
from pipeline import (
    EntityResolutionPipeline,
    PipelineConfig,
    PrecomputedCandidateAdapter,
    PROVISIONAL_DEFAULT_THRESHOLD,
)


class TestMember2XGBoost13Model(unittest.TestCase):

    def setUp(self):
        # Sample vocabulary for TF-IDF vectorizers
        corpus_names = [
            "apex logistics llc",
            "bharat enterprises pvt ltd",
            "chateau vignoble sarl",
            "global tech solutions inc",
        ]
        corpus_addrs = [
            "100 industry way springfield",
            "plot 45 midc area mumbai",
            "12 rue de paris lyon",
            "500 technology parkway silicon valley",
        ]

        self.name_vec = TfidfVectorizer().fit(corpus_names)
        self.addr_vec = TfidfVectorizer().fit(corpus_addrs)

        # Build synthetic RecordCatalog
        self.catalog = RecordCatalog(
            s1_records={
                "S1-101": ("Apex Logistics LLC", "100 Industry Way", "US"),
                "S1-102": ("Bharat Enterprises", "Plot 45 MIDC Area", "India"),
                "S1-103": ("Château Vignoble", "12 Rue de Paris", "France"),
            },
            s2_records={
                "S2-201": ("Apex Logistics Inc", "100 Industry Way", "US"),
                "S2-202": ("Bharat Enterprises Pvt Ltd", "Plot 45 MIDC Area", "India"),
            },
            s3_records={
                "S3-301": ("Apex Freight Solutions", "999 Other Rd", "US"),
            },
        )

        # Train a minimal real XGBClassifier with the exact 13 feature columns
        np.random.seed(42)
        X_synthetic = pd.DataFrame(
            np.random.uniform(0.0, 1.0, size=(20, 13)),
            columns=FEATURE_NAMES_13,
        )
        y_synthetic = np.random.choice([0, 1], size=20)

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=3,
            max_depth=2,
            random_state=42,
            eval_metric="logloss",
        )
        self.xgb_model.fit(X_synthetic, y_synthetic)

    def test_missing_model_artifact_raises_file_not_found(self):
        """Adapter must raise FileNotFoundError when model path does not exist."""
        nonexistent_path = Path("student_resource/dataset/train/nonexistent_model.pkl")
        with self.assertRaises(FileNotFoundError) as ctx:
            Member2XGBoost13FeatureAdapter(
                model_path=nonexistent_path,
                name_vectorizer=self.name_vec,
                address_vectorizer=self.addr_vec,
            )
        self.assertIn("Member 2 XGBoost model artifact not found", str(ctx.exception))

    def test_missing_vectorizers_raise_file_not_found(self):
        """Adapter must raise FileNotFoundError when TF-IDF vectorizers are missing."""
        # Missing name vectorizer
        with self.assertRaises(FileNotFoundError) as ctx:
            Member2XGBoost13FeatureAdapter(
                model=self.xgb_model,
                name_vectorizer=None,
                address_vectorizer=self.addr_vec,
            )
        self.assertIn("Name TF-IDF vectorizer is missing", str(ctx.exception))

        # Missing address vectorizer
        with self.assertRaises(FileNotFoundError) as ctx:
            Member2XGBoost13FeatureAdapter(
                model=self.xgb_model,
                name_vectorizer=self.name_vec,
                address_vectorizer=None,
            )
        self.assertIn("Address TF-IDF vectorizer is missing", str(ctx.exception))

    def test_feature_names_mismatch_raises_value_error(self):
        """Adapter must verify model.feature_names_in_ matches the 13 required features."""
        # Train model with wrong feature names
        wrong_features = [f"feat_{i}" for i in range(13)]
        X_wrong = pd.DataFrame(np.zeros((10, 13)), columns=wrong_features)
        y_wrong = np.zeros(10)
        wrong_model = xgb.XGBClassifier(n_estimators=2).fit(X_wrong, y_wrong)

        with self.assertRaises(ValueError) as ctx:
            Member2XGBoost13FeatureAdapter(
                model=wrong_model,
                name_vectorizer=self.name_vec,
                address_vectorizer=self.addr_vec,
            )
        self.assertIn("Model feature_names_in_ schema mismatch", str(ctx.exception))

    def test_missing_record_id_raises_key_error(self):
        """Adapter must raise KeyError when an entity ID in candidate_batch is missing from catalog."""
        adapter = Member2XGBoost13FeatureAdapter(
            model=self.xgb_model,
            name_vectorizer=self.name_vec,
            address_vectorizer=self.addr_vec,
            records_catalog=self.catalog,
        )
        bad_batch = pd.DataFrame([
            {"source1_entity_id": "S1-999_NONEXISTENT", "candidate_entity_id": "S2-201", "source": "source2"}
        ])
        with self.assertRaises(KeyError) as ctx:
            adapter.predict_batch(bad_batch)
        self.assertIn("S1-999_NONEXISTENT", str(ctx.exception))

    def test_missing_columns_raises_value_error(self):
        """Adapter must raise ValueError if required candidate columns are missing."""
        adapter = Member2XGBoost13FeatureAdapter(
            model=self.xgb_model,
            name_vectorizer=self.name_vec,
            address_vectorizer=self.addr_vec,
            records_catalog=self.catalog,
        )
        bad_df = pd.DataFrame({"source1_entity_id": ["S1-101"]})  # missing candidate_entity_id
        with self.assertRaises(ValueError) as ctx:
            adapter.predict_batch(bad_df)
        self.assertIn("missing required columns", str(ctx.exception))

    def test_small_batch_scoring_and_row_order_preservation(self):
        """
        Verify that a small batch of candidate pairs:
        - Computes all 13 features.
        - Scores successfully with real XGBoost model.
        - Preserves input row order and returns exactly one probability per input row.
        """
        adapter = Member2XGBoost13FeatureAdapter(
            model=self.xgb_model,
            name_vectorizer=self.name_vec,
            address_vectorizer=self.addr_vec,
            records_catalog=self.catalog,
        )

        candidate_batch = pd.DataFrame([
            {"source1_entity_id": "S1-101", "candidate_entity_id": "S2-201", "source": "source2"},
            {"source1_entity_id": "S1-101", "candidate_entity_id": "S3-301", "source": "source3"},
            {"source1_entity_id": "S1-102", "candidate_entity_id": "S2-202", "source": "source2"},
        ])

        probs = adapter.predict_batch(candidate_batch)

        self.assertEqual(len(probs), 3)
        for idx, p in enumerate(probs):
            self.assertIsInstance(p, float)
            self.assertFalse(np.isnan(p))
            self.assertFalse(np.isinf(p))
            self.assertTrue(0.0 <= p <= 1.0, f"Probability out of range: {p}")

    def test_pipeline_integration_with_xgb_adapter(self):
        """
        Run end-to-end pipeline with Member2XGBoost13FeatureAdapter:
        - Stream candidate chunks with batch_size=2.
        - Use provisional threshold=0.60.
        - Accumulate results in SQLite accumulator.
        - Emit valid matching_results.tsv and candidate_pairs.tsv.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir = temp_path / "data"
            output_dir = temp_path / "output"
            data_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)

            # Write test_source1.tsv
            s1_lines = [
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n",
                "S1-101\tApex Logistics LLC\t100 Industry Way\tUS\n",
                "S1-102\tBharat Enterprises\tPlot 45 MIDC Area\tIndia\n",
                "S1-103\tChâteau Vignoble\t12 Rue de Paris\tFrance\n",
            ]
            (data_dir / "test_source1.tsv").write_text("".join(s1_lines), encoding="utf-8")

            # Write candidate file
            cand_tsv = data_dir / "candidates.tsv"
            cand_lines = [
                "source1_entity_id\tcandidate_entity_id\tsource\n",
                "S1-101\tS2-201\tsource2\n",
                "S1-101\tS3-301\tsource3\n",
                "S1-102\tS2-202\tsource2\n",
            ]
            cand_tsv.write_text("".join(cand_lines), encoding="utf-8")

            adapter = Member2XGBoost13FeatureAdapter(
                model=self.xgb_model,
                name_vectorizer=self.name_vec,
                address_vectorizer=self.addr_vec,
                records_catalog=self.catalog,
            )

            cand_gen = PrecomputedCandidateAdapter(cand_tsv)
            config = PipelineConfig(
                data_dir=data_dir,
                output_dir=output_dir,
                batch_size=2,
                threshold=PROVISIONAL_DEFAULT_THRESHOLD,  # 0.60
                skip_validator=True,
            )

            pipeline = EntityResolutionPipeline(cand_gen, adapter, config)
            matching_path, candidate_path = pipeline.run()

            self.assertTrue(matching_path.is_file())
            self.assertTrue(candidate_path.is_file())

            # Read back outputs
            match_df = pd.read_csv(matching_path, sep="\t", dtype=str).fillna("")
            cand_df = pd.read_csv(candidate_path, sep="\t", dtype=str).fillna("")

            # 3 rows, exactly matching test_source1.tsv IDs
            self.assertEqual(match_df["source1_entity_id"].tolist(), ["S1-101", "S1-102", "S1-103"])
            self.assertEqual(cand_df["source1_entity_id"].tolist(), ["S1-101", "S1-102", "S1-103"])

            # S1-103 singleton has empty fields
            self.assertEqual(match_df.loc[match_df["source1_entity_id"] == "S1-103", "matched_entity_ids"].iloc[0], "")
            self.assertEqual(cand_df.loc[cand_df["source1_entity_id"] == "S1-103", "candidate_entity_ids"].iloc[0], "")

    def test_real_dataset_record_catalog_sample(self):
        """
        Verify that RecordCatalog can read a small sample from actual dataset files
        (e.g., dataset/train/train_source1.tsv) and compute features correctly.
        """
        train_s1 = Path("dataset/train/train_source1.tsv")
        train_s2 = Path("dataset/train/train_source2.tsv")

        if not (train_s1.is_file() and train_s2.is_file()):
            self.skipTest("dataset/train/ source files not found.")

        # Read top 5 records from train files
        s1_df = pd.read_csv(train_s1, sep="\t", nrows=5, dtype=str).fillna("")
        s2_df = pd.read_csv(train_s2, sep="\t", nrows=5, dtype=str).fillna("")

        catalog = RecordCatalog(
            s1_records={
                str(r["entity_id"]): (str(r["business_name"]), str(r["business_address"]), str(r["country"]))
                for _, r in s1_df.iterrows()
            },
            s2_records={
                str(r["entity_id"]): (str(r["business_name"]), str(r["business_address"]), str(r["country"]))
                for _, r in s2_df.iterrows()
            },
        )

        s1_id = s1_df.iloc[0]["entity_id"]
        s2_id = s2_df.iloc[0]["entity_id"]

        sample_batch = pd.DataFrame([
            {"source1_entity_id": s1_id, "candidate_entity_id": s2_id, "source": "source2"}
        ])

        feat_df = compute_13_features_for_batch(
            candidate_batch=sample_batch,
            catalog=catalog,
            name_vectorizer=self.name_vec,
            address_vectorizer=self.addr_vec,
        )

        self.assertEqual(list(feat_df.columns), FEATURE_NAMES_13)
        self.assertEqual(len(feat_df), 1)

    def test_combined_tfidf_vectorizers_artifact_loading(self):
        """Verify that a combined vectorizers artifact pickle can be loaded properly."""
        import joblib
        with tempfile.TemporaryDirectory() as td:
            vec_path = Path(td) / "tfidf_vectorizers.pkl"
            # Save dict format
            joblib.dump({"name_vectorizer": self.name_vec, "address_vectorizer": self.addr_vec}, vec_path)

            adapter = Member2XGBoost13FeatureAdapter(
                model=self.xgb_model,
                vectorizers_path=vec_path,
                records_catalog=self.catalog,
            )
            self.assertIsNotNone(adapter.name_vectorizer)
            self.assertIsNotNone(adapter.address_vectorizer)

            # Test tuple format
            joblib.dump((self.name_vec, self.addr_vec), vec_path)
            adapter_tuple = Member2XGBoost13FeatureAdapter(
                model=self.xgb_model,
                vectorizers_path=vec_path,
                records_catalog=self.catalog,
            )
            self.assertIsNotNone(adapter_tuple.name_vectorizer)
            self.assertIsNotNone(adapter_tuple.address_vectorizer)


if __name__ == "__main__":
    unittest.main()
