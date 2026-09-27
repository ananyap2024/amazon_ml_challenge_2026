"""
ML Challenge 2026 — Member 2 Model Inference Adapter (Member 3 Pipeline Integration)

Connects Member 2's production 13-feature XGBoost matching model to the
streaming, disk-backed pipeline:
1. Loads production model from code/business_entity_resolution/models/matching_model.pkl.
2. Loads production TF-IDF vectorizers from code/business_entity_resolution/models/tfidf_vectorizers.pkl.
3. Resolves candidate pairs to business records via RecordCatalog.
4. Directly invokes Member 2's production feature engineering (features_13.py, features.py, features_extra.py)
   and inference API (inference.py).
5. Enforces exact row-order preservation, returning 1-to-1 probabilities in [0.0, 1.0].
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Path to Member 2 production source package
PROD_SRC_DIR = Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution" / "src"
PROD_MODELS_DIR = Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution" / "models"

# Add Member 2 package to path if present
if PROD_SRC_DIR.is_dir() and str(PROD_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROD_SRC_DIR))

try:
    import inference as prod_inference
    from features_13 import (
        FEATURE_NAMES_13 as PROD_FEATURE_NAMES_13,
        RECORD_COLUMNS as PROD_RECORD_COLUMNS,
        compute_pair_features as prod_compute_pair_features,
    )
    MEMBER2_PROD_AVAILABLE = True
except ImportError:
    prod_inference = None
    prod_compute_pair_features = None
    MEMBER2_PROD_AVAILABLE = False

try:
    import joblib
except ImportError:
    joblib = None

DELIM = "\t"

FEATURE_NAMES_13: List[str] = [
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


class RecordCatalog:
    """
    In-memory catalog of business records across Source 1, Source 2, and Source 3.
    Stores tuples of (business_name, business_address, country) indexed by entity_id.
    """

    def __init__(
        self,
        s1_records: Optional[Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]]] = None,
        s2_records: Optional[Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]]] = None,
        s3_records: Optional[Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]]] = None,
    ):
        self.s1: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = s1_records or {}
        self.s2: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = s2_records or {}
        self.s3: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = s3_records or {}

    @classmethod
    def from_directory(
        cls,
        data_dir: Union[str, Path],
        split: str = "test",
    ) -> RecordCatalog:
        """
        Load records from TSV files in data_dir (e.g., test_source1.tsv, etc.).
        """
        dir_path = Path(data_dir)
        catalog = cls()

        def load_tsv(file_path: Path) -> Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]]:
            records: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = {}
            if not file_path.is_file():
                raise FileNotFoundError(f"Source file not found: {file_path}")
            df = pd.read_csv(file_path, sep=DELIM, dtype=str)
            req_cols = {"entity_id", "business_name", "business_address", "country"}
            if not req_cols.issubset(df.columns):
                raise ValueError(f"File {file_path} missing required columns: {req_cols - set(df.columns)}")
            for _, row in df.iterrows():
                eid = str(row["entity_id"]).strip()
                bname = row["business_name"] if pd.notna(row["business_name"]) else None
                baddr = row["business_address"] if pd.notna(row["business_address"]) else None
                country = row["country"] if pd.notna(row["country"]) else None
                records[eid] = (bname, baddr, country)
            return records

        catalog.s1 = load_tsv(dir_path / f"{split}_source1.tsv")
        catalog.s2 = load_tsv(dir_path / f"{split}_source2.tsv")
        catalog.s3 = load_tsv(dir_path / f"{split}_source3.tsv")
        return catalog

    def get_record(self, source: str, entity_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Retrieve record tuple (business_name, business_address, country).
        Raises KeyError if entity_id is not present in the specified source catalog.
        """
        source_clean = source.strip().lower()
        if source_clean in ("source1", "s1"):
            target_dict = self.s1
        elif source_clean in ("source2", "s2"):
            target_dict = self.s2
        elif source_clean in ("source3", "s3"):
            target_dict = self.s3
        else:
            raise ValueError(f"Unknown source '{source}'. Expected source1, source2, or source3.")

        if entity_id not in target_dict:
            raise KeyError(
                f"Record ID '{entity_id}' not found in {source_clean} catalog. "
                f"Total available in {source_clean}: {len(target_dict):,}."
            )
        return target_dict[entity_id]


class Member2XGBoost13FeatureAdapter:
    """
    Adapter for Member 2's 13-feature XGBoost matching model.

    Features:
    - Loads trained model artifact via Member 2's inference.load_model() or joblib.
    - Uses model.feature_names_in_ to verify exact feature names and order.
    - Loads fitted TF-IDF vectorizers via Member 2's inference.load_vectorizers().
    - Resolves candidate pairs to business records using RecordCatalog.
    - Preserves row ordering and outputs calibrated probabilities in [0.0, 1.0].
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        name_vectorizer: Optional[Any] = None,
        address_vectorizer: Optional[Any] = None,
        vectorizers_path: Optional[Union[str, Path]] = None,
        name_vectorizer_path: Optional[Union[str, Path]] = None,
        address_vectorizer_path: Optional[Union[str, Path]] = None,
        records_catalog: Optional[RecordCatalog] = None,
        model: Optional[Any] = None,
    ):
        self.records_catalog = records_catalog

        # 1. Load or assign model
        if model is not None:
            self.model = model
        else:
            primary_path = Path(
                model_path or (PROD_MODELS_DIR / "matching_model.pkl")
            )
            fallback_path = Path("student_resource/dataset/train/analysis_model_xgboost_13features_realistic.pkl")
            m_path = primary_path if primary_path.is_file() else fallback_path

            if not m_path.is_file():
                raise FileNotFoundError(
                    f"Member 2 XGBoost model artifact not found at: '{primary_path}' "
                    f"(or fallback '{fallback_path}').\n"
                    f"Please provide the production model artifact."
                )
            if MEMBER2_PROD_AVAILABLE:
                self.model = prod_inference.load_model(m_path)
            else:
                if joblib is None:
                    raise ImportError("joblib is required to load the model artifact.")
                self.model = joblib.load(m_path)

        # 2. Verify feature names and ordering using model.feature_names_in_
        if hasattr(self.model, "feature_names_in_"):
            actual_features = list(self.model.feature_names_in_)
            if actual_features != FEATURE_NAMES_13:
                raise ValueError(
                    f"Model feature_names_in_ schema mismatch.\n"
                    f"Expected by Member 2 adapter: {FEATURE_NAMES_13}\n"
                    f"Model reported feature_names_in_: {actual_features}"
                )

        # 3. Load combined TF-IDF Vectorizers artifact if provided
        if vectorizers_path is not None or (name_vectorizer is None and name_vectorizer_path is None):
            vec_file = Path(vectorizers_path or (PROD_MODELS_DIR / "tfidf_vectorizers.pkl"))
            if vec_file.is_file():
                if MEMBER2_PROD_AVAILABLE:
                    name_vectorizer, address_vectorizer = prod_inference.load_vectorizers(vec_file)
                else:
                    if joblib is None:
                        raise ImportError("joblib is required to load vectorizer artifact.")
                    loaded_vecs = joblib.load(vec_file)
                    if isinstance(loaded_vecs, dict):
                        name_vectorizer = (
                            loaded_vecs.get("name_vectorizer")
                            or loaded_vecs.get("name")
                            or loaded_vecs.get("name_vec")
                        )
                        address_vectorizer = (
                            loaded_vecs.get("address_vectorizer")
                            or loaded_vecs.get("address")
                            or loaded_vecs.get("addr_vec")
                        )
                    elif isinstance(loaded_vecs, (tuple, list)) and len(loaded_vecs) >= 2:
                        name_vectorizer = loaded_vecs[0]
                        address_vectorizer = loaded_vecs[1]

        # 4. Load or assign Name TF-IDF Vectorizer
        if name_vectorizer is not None:
            self.name_vectorizer = name_vectorizer
        elif name_vectorizer_path is not None:
            nv_path = Path(name_vectorizer_path)
            if not nv_path.is_file():
                raise FileNotFoundError(f"Name TF-IDF vectorizer artifact not found at: {nv_path}")
            if joblib is None:
                raise ImportError("joblib is required to load vectorizer artifact.")
            self.name_vectorizer = joblib.load(nv_path)
        else:
            raise FileNotFoundError(
                "Name TF-IDF vectorizer is missing. Expected fitted vectorizer in "
                "'code/business_entity_resolution/models/tfidf_vectorizers.pkl' or via parameters."
            )

        # 5. Load or assign Address TF-IDF Vectorizer
        if address_vectorizer is not None:
            self.address_vectorizer = address_vectorizer
        elif address_vectorizer_path is not None:
            av_path = Path(address_vectorizer_path)
            if not av_path.is_file():
                raise FileNotFoundError(f"Address TF-IDF vectorizer artifact not found at: {av_path}")
            if joblib is None:
                raise ImportError("joblib is required to load vectorizer artifact.")
            self.address_vectorizer = joblib.load(av_path)
        else:
            raise FileNotFoundError(
                "Address TF-IDF vectorizer is missing. Expected fitted vectorizer in "
                "'code/business_entity_resolution/models/tfidf_vectorizers.pkl' or via parameters."
            )

    def predict_batch(self, candidate_batch: pd.DataFrame) -> Sequence[float]:
        """
        Score a batch of candidate pairs using the 13-feature XGBoost model.

        Parameters
        ----------
        candidate_batch : pd.DataFrame
            Columns: ['source1_entity_id', 'candidate_entity_id', (optional) 'source']

        Returns
        -------
        Sequence[float]
            List of match probabilities corresponding 1-to-1 with candidate_batch rows.
        """
        if candidate_batch.empty:
            return []

        if self.records_catalog is None:
            raise ValueError(
                "records_catalog is not configured. Member2XGBoost13FeatureAdapter requires "
                "a RecordCatalog to resolve candidate IDs to business records."
            )

        n_rows = len(candidate_batch)

        required_cols = {"source1_entity_id", "candidate_entity_id"}
        missing = required_cols - set(candidate_batch.columns)
        if missing:
            raise ValueError(f"candidate_batch missing required columns: {missing}")

        s1_ids = candidate_batch["source1_entity_id"].astype(str).tolist()
        cand_ids = candidate_batch["candidate_entity_id"].astype(str).tolist()
        if "source" in candidate_batch.columns:
            sources = candidate_batch["source"].astype(str).tolist()
        else:
            sources = ["source2" if cid.startswith("S2-") else "source3" for cid in cand_ids]

        # Build candidate_batch formatted for Member 2 inference
        candidate_batch_dicts = []
        for s1_id, cand_id, src in zip(s1_ids, cand_ids, sources):
            s1_rec = self.records_catalog.get_record("source1", s1_id)
            c_rec = self.records_catalog.get_record(src, cand_id)
            candidate_batch_dicts.append({
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

        if MEMBER2_PROD_AVAILABLE:
            probs_arr = prod_inference.predict_batch(
                candidate_batch=candidate_batch_dicts,
                model=self.model,
                name_vectorizer=self.name_vectorizer,
                address_vectorizer=self.address_vectorizer,
            )
            probs = probs_arr.tolist()
        else:
            raise RuntimeError("Member 2 production inference package is not available.")

        # Strict validation of outputs
        if len(probs) != n_rows:
            raise ValueError(
                f"Model predict_proba returned {len(probs)} scores for a batch of {n_rows} candidate pairs."
            )

        validated_probs: List[float] = []
        for idx, p in enumerate(probs):
            p_float = float(p)
            if math.isnan(p_float) or math.isinf(p_float) or not (0.0 <= p_float <= 1.0):
                row_pair = (
                    candidate_batch.iloc[idx]["source1_entity_id"],
                    candidate_batch.iloc[idx]["candidate_entity_id"],
                )
                raise ValueError(
                    f"Model produced invalid probability {p_float} for row {idx} {row_pair}. "
                    f"Must be a finite float in [0.0, 1.0]."
                )
            validated_probs.append(p_float)

        return validated_probs
