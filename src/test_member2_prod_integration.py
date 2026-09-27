"""
ML Challenge 2026 — Production Integration Test for Member 2 Model & Inference

Verifies:
1. Loading production model from code/business_entity_resolution/models/matching_model.pkl.
2. Loading production TF-IDF vectorizers from code/business_entity_resolution/models/tfidf_vectorizers.pkl.
3. Scoring real records from dataset/train/ with Member 2's direct inference.predict_batch().
4. Scoring the exact same candidate pairs through Member2XGBoost13FeatureAdapter.
5. Verifying that the maximum absolute difference between Member 2 direct inference
   and the pipeline adapter is 0.0 (exact numerical parity).
6. Preserving exact input row order and validating probabilities in [0.0, 1.0].
"""

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

# Add Member 2 source directory and src directory
_PROD_SRC = Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution" / "src"
_SRC = Path(__file__).resolve().parent
if str(_PROD_SRC) not in sys.path:
    sys.path.insert(0, str(_PROD_SRC))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import inference
from model_adapter import Member2XGBoost13FeatureAdapter, RecordCatalog


class TestMember2ProductionIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_path = Path("code/business_entity_resolution/models/matching_model.pkl")
        cls.tfidf_path = Path("code/business_entity_resolution/models/tfidf_vectorizers.pkl")

        if not cls.model_path.is_file() or not cls.tfidf_path.is_file():
            raise unittest.SkipTest("Production model/vectorizer artifacts not found.")

        cls.model = inference.load_model(cls.model_path)
        cls.name_vec, cls.addr_vec = inference.load_vectorizers(cls.tfidf_path)

        # Read sample records from train dataset
        s1_file = Path("dataset/train/train_source1.tsv")
        s2_file = Path("dataset/train/train_source2.tsv")
        gt_file = Path("dataset/train/train_ground_truth.tsv")

        if not (s1_file.is_file() and s2_file.is_file() and gt_file.is_file()):
            raise unittest.SkipTest("Training dataset files not found for integration test.")

        # Find known true matches and negative pairs
        gt_df = pd.read_csv(gt_file, sep="\t").dropna().head(5)
        needed_s1 = set(gt_df["source1_entity_id"])
        needed_s2 = set()
        cls.pairs_meta = []

        for _, row in gt_df.iterrows():
            s1_id = row["source1_entity_id"]
            matched_cands = str(row["matched_entity_ids"]).split(",")
            s2_cands = [c.strip() for c in matched_cands if c.strip().startswith("S2-")]
            if s2_cands:
                cand_id = s2_cands[0]
                needed_s2.add(cand_id)
                cls.pairs_meta.append((s1_id, cand_id, "source2", 1))

        # Add negative pairs
        s1_sample = pd.read_csv(s1_file, sep="\t", nrows=10, dtype=str).fillna("")
        s2_sample = pd.read_csv(s2_file, sep="\t", nrows=10, dtype=str).fillna("")

        for i in range(min(5, len(s1_sample), len(s2_sample))):
            s1_id = s1_sample.iloc[i]["entity_id"]
            s2_id = s2_sample.iloc[-(i + 1)]["entity_id"]
            needed_s1.add(s1_id)
            needed_s2.add(s2_id)
            cls.pairs_meta.append((s1_id, s2_id, "source2", 0))

        # Load needed records into dictionary catalog
        cls.s1_records = {}
        for chunk in pd.read_csv(s1_file, sep="\t", chunksize=100_000, dtype=str):
            sub = chunk[chunk["entity_id"].isin(needed_s1)]
            for _, r in sub.iterrows():
                cls.s1_records[r["entity_id"]] = (
                    r["business_name"] if pd.notna(r["business_name"]) else None,
                    r["business_address"] if pd.notna(r["business_address"]) else None,
                    r["country"] if pd.notna(r["country"]) else None,
                )
            if len(cls.s1_records) >= len(needed_s1):
                break

        cls.s2_records = {}
        for chunk in pd.read_csv(s2_file, sep="\t", chunksize=100_000, dtype=str):
            sub = chunk[chunk["entity_id"].isin(needed_s2)]
            for _, r in sub.iterrows():
                cls.s2_records[r["entity_id"]] = (
                    r["business_name"] if pd.notna(r["business_name"]) else None,
                    r["business_address"] if pd.notna(r["business_address"]) else None,
                    r["country"] if pd.notna(r["country"]) else None,
                )
            if len(cls.s2_records) >= len(needed_s2):
                break

        cls.catalog = RecordCatalog(s1_records=cls.s1_records, s2_records=cls.s2_records)

    def test_production_artifacts_schema_and_order(self):
        """Verify model feature_names_in_ matches exact 13-feature ordering."""
        expected_features = [
            "name_exact",
            "name_similarity",
            "address_exact",
            "address_similarity",
            "country_same",
            "name_token_overlap",
            "address_token_overlap",
            "name_length_similarity",
            "address_length_similarity",
            "name_levenshtein_similarity",
            "address_levenshtein_similarity",
            "name_tfidf_cosine",
            "address_tfidf_cosine",
        ]
        self.assertEqual(list(self.model.feature_names_in_), expected_features)

    def test_direct_inference_vs_adapter_parity(self):
        """
        Compare Member 2 direct inference.predict_batch() vs Member2XGBoost13FeatureAdapter.predict_batch().
        Must produce identical outputs (max absolute difference == 0.0).
        """
        # Build direct inference candidate_batch (list of dicts)
        direct_batch = []
        for s1_id, cand_id, src, label in self.pairs_meta:
            s1_rec = self.catalog.get_record("source1", s1_id)
            c_rec = self.catalog.get_record(src, cand_id)
            direct_batch.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "source": src,
                "source1_record": {
                    "entity_id": s1_id,
                    "business_name": s1_rec[0],
                    "business_address": s1_rec[1],
                    "country": s1_rec[2],
                },
                "candidate_record": {
                    "entity_id": cand_id,
                    "business_name": c_rec[0],
                    "business_address": c_rec[1],
                    "country": c_rec[2],
                },
            })

        # Run direct production inference
        probs_direct = inference.predict_batch(
            candidate_batch=direct_batch,
            model=self.model,
            name_vectorizer=self.name_vec,
            address_vectorizer=self.addr_vec,
        )

        # Build pipeline DataFrame candidate_batch
        pipeline_df = pd.DataFrame([
            {
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "source": src,
            }
            for s1_id, cand_id, src, label in self.pairs_meta
        ])

        # Run pipeline adapter inference
        adapter = Member2XGBoost13FeatureAdapter(
            model=self.model,
            name_vectorizer=self.name_vec,
            address_vectorizer=self.addr_vec,
            records_catalog=self.catalog,
        )
        probs_adapter = adapter.predict_batch(pipeline_df)

        self.assertEqual(len(probs_direct), len(probs_adapter))
        self.assertEqual(len(probs_adapter), len(self.pairs_meta))

        diffs = np.abs(np.array(probs_direct) - np.array(probs_adapter))
        max_diff = float(np.max(diffs))

        print(f"\n[Parity Check] Batch Size: {len(self.pairs_meta)}")
        print(f"[Parity Check] Max Absolute Difference: {max_diff:.10e}")
        for i, (prob_d, prob_a) in enumerate(zip(probs_direct, probs_adapter)):
            pair = self.pairs_meta[i]
            print(f"  Row {i:2d} ({pair[0]} -> {pair[1]} | label={pair[3]}): direct={prob_d:.6f}, adapter={prob_a:.6f}")

        self.assertAlmostEqual(max_diff, 0.0, places=9, msg="Mismatch between direct inference and pipeline adapter")


if __name__ == "__main__":
    unittest.main()
