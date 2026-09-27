"""
ML Challenge 2026 — 13-Feature Engineering & Member 2 Model Adapter

Implements the official 13-feature representation reported by Member 2:
1.  name_exact: Binary indicator of exact normalized name match.
2.  name_similarity: Token Jaccard similarity of business names.
3.  address_exact: Binary indicator of exact normalized address match.
4.  address_similarity: Token Jaccard similarity of business addresses.
5.  country_same: Binary indicator of exact country match.
6.  name_token_overlap: Count of shared name tokens.
7.  address_token_overlap: Count of shared address tokens.
8.  name_length_similarity: Ratio of shorter to longer normalized name length.
9.  address_length_similarity: Ratio of shorter to longer normalized address length.
10. name_levenshtein_similarity: Normalized Levenshtein distance similarity for names.
11. address_levenshtein_similarity: Normalized Levenshtein distance similarity for addresses.
12. name_tfidf_cosine: Cosine similarity of fitted TF-IDF name vectors.
13. address_tfidf_cosine: Cosine similarity of fitted TF-IDF address vectors.

Guarantees:
- Strictly enforces the 13-feature schema and column ordering.
- Resolves candidate pairs against Source 1, Source 2, and Source 3 records.
- Preserves exact input row order; never sorts, drops, or deduplicates candidate rows.
- Returns exactly one calibrated probability per input row in [0.0, 1.0].
- Validates model and vectorizer artifact existence, raising clear FileNotFoundError.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import sys

try:
    import joblib
except ImportError:
    joblib = None

# Dynamically link Member 2 production package
_PROD_SRC = Path(__file__).resolve().parent.parent / "code" / "business_entity_resolution" / "src"
if _PROD_SRC.is_dir() and str(_PROD_SRC) not in sys.path:
    sys.path.insert(0, str(_PROD_SRC))

try:
    import inference as prod_inference
    import features_13 as prod_features_13
    MEMBER2_PROD_AVAILABLE = True
except ImportError:
    MEMBER2_PROD_AVAILABLE = False

from preprocessing import (
    normalize_address,
    normalize_business_name,
    tokenize,
)

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


# =============================================================================
# 1. String & Token Metric Primitives
# =============================================================================

def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein similarity between two strings:
    1.0 - (levenshtein_distance / max(len(s1), len(s2))).
    Returns 1.0 for two empty strings, 0.0 if one is empty and other is not.
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)

    # Ensure s1 is shorter to optimize DP row memory
    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    current_row = list(range(len1 + 1))
    for i, c2 in enumerate(s2, 1):
        previous_row, current_row = current_row, [i] + [0] * len1
        for j, c1 in enumerate(s1, 1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (c1 != c2)
            current_row[j] = min(insertions, deletions, substitutions)

    dist = current_row[len1]
    return max(0.0, 1.0 - (dist / max_len))


def token_jaccard_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute token Jaccard similarity |T1 ∩ T2| / |T1 ∪ T2|."""
    set1, set2 = set(tokens1), set(tokens2)
    union = set1 | set2
    if not union:
        return 0.0
    return len(set1 & set2) / len(union)


def token_overlap_count(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute token intersection count |T1 ∩ T2|."""
    return float(len(set(tokens1) & set(tokens2)))


def length_similarity(s1: str, s2: str) -> float:
    """Compute ratio of shorter string to longer string length."""
    len1, len2 = len(s1), len(s2)
    max_len = max(len1, len2)
    if max_len == 0:
        return 0.0
    return min(len1, len2) / max_len


# =============================================================================
# 2. Record Catalog for Entity Resolution
# =============================================================================

class RecordCatalog:
    """
    In-memory catalog of business records across Source 1, Source 2, and Source 3.
    Stores tuples of (business_name, business_address, country) indexed by entity_id.
    """

    def __init__(
        self,
        s1_records: Optional[Dict[str, Tuple[str, str, str]]] = None,
        s2_records: Optional[Dict[str, Tuple[str, str, str]]] = None,
        s3_records: Optional[Dict[str, Tuple[str, str, str]]] = None,
    ):
        self.s1: Dict[str, Tuple[str, str, str]] = s1_records or {}
        self.s2: Dict[str, Tuple[str, str, str]] = s2_records or {}
        self.s3: Dict[str, Tuple[str, str, str]] = s3_records or {}

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

        def load_tsv(file_path: Path) -> Dict[str, Tuple[str, str, str]]:
            records: Dict[str, Tuple[str, str, str]] = {}
            if not file_path.is_file():
                raise FileNotFoundError(f"Source file not found: {file_path}")
            df = pd.read_csv(file_path, sep=DELIM, dtype=str).fillna("")
            req_cols = {"entity_id", "business_name", "business_address", "country"}
            if not req_cols.issubset(df.columns):
                raise ValueError(f"File {file_path} missing required columns: {req_cols - set(df.columns)}")
            for _, row in df.iterrows():
                eid = str(row["entity_id"]).strip()
                bname = str(row["business_name"])
                baddr = str(row["business_address"])
                country = str(row["country"])
                records[eid] = (bname, baddr, country)
            return records

        catalog.s1 = load_tsv(dir_path / f"{split}_source1.tsv")
        catalog.s2 = load_tsv(dir_path / f"{split}_source2.tsv")
        catalog.s3 = load_tsv(dir_path / f"{split}_source3.tsv")
        return catalog

    def get_record(self, source: str, entity_id: str) -> Tuple[str, str, str]:
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


# =============================================================================
# 3. 13-Feature Matrix Generator
# =============================================================================

def compute_13_features_for_batch(
    candidate_batch: pd.DataFrame,
    catalog: RecordCatalog,
    name_vectorizer: Any,
    address_vectorizer: Any,
) -> pd.DataFrame:
    """
    Generate the exact 13-feature matrix for a batch of candidate pairs.
    Preserves input row order exactly.
    """
    required_cols = {"source1_entity_id", "candidate_entity_id"}
    missing = required_cols - set(candidate_batch.columns)
    if missing:
        raise ValueError(f"candidate_batch missing required columns: {missing}")

    if candidate_batch.empty:
        return pd.DataFrame(columns=FEATURE_NAMES_13)

    s1_ids = candidate_batch["source1_entity_id"].astype(str).tolist()
    cand_ids = candidate_batch["candidate_entity_id"].astype(str).tolist()

    if "source" in candidate_batch.columns:
        sources = candidate_batch["source"].astype(str).tolist()
    else:
        sources = ["source2" if cid.startswith("S2-") else "source3" for cid in cand_ids]

    n_rows = len(candidate_batch)

    # 1. Resolve records from catalog
    s1_names: List[str] = []
    s1_addrs: List[str] = []
    s1_countries: List[str] = []
    c_names: List[str] = []
    c_addrs: List[str] = []
    c_countries: List[str] = []

    for s1_id, cand_id, src in zip(s1_ids, cand_ids, sources):
        s1_rec = catalog.get_record("source1", s1_id)
        c_rec = catalog.get_record(src, cand_id)

        s1_names.append(s1_rec[0])
        s1_addrs.append(s1_rec[1])
        s1_countries.append(s1_rec[2])

        c_names.append(c_rec[0])
        c_addrs.append(c_rec[1])
        c_countries.append(c_rec[2])

    # 2. Preprocess strings
    norm_s1_names = [normalize_business_name(n) for n in s1_names]
    norm_c_names = [normalize_business_name(n) for n in c_names]
    norm_s1_addrs = [normalize_address(a) for a in s1_addrs]
    norm_c_addrs = [normalize_address(a) for a in c_addrs]

    # Tokenize
    tokens_s1_names = [tokenize(n) for n in norm_s1_names]
    tokens_c_names = [tokenize(n) for n in norm_c_names]
    tokens_s1_addrs = [tokenize(a) for a in norm_s1_addrs]
    tokens_c_addrs = [tokenize(a) for a in norm_c_addrs]

    # 3. Compute scalar features
    feat_name_exact = [
        1.0 if (n1 and n1 == n2) else 0.0
        for n1, n2 in zip(norm_s1_names, norm_c_names)
    ]
    feat_name_similarity = [
        token_jaccard_similarity(t1, t2)
        for t1, t2 in zip(tokens_s1_names, tokens_c_names)
    ]
    feat_address_exact = [
        1.0 if (a1 and a1 == a2) else 0.0
        for a1, a2 in zip(norm_s1_addrs, norm_c_addrs)
    ]
    feat_address_similarity = [
        token_jaccard_similarity(t1, t2)
        for t1, t2 in zip(tokens_s1_addrs, tokens_c_addrs)
    ]
    feat_country_same = [
        1.0 if (c1 and c2 and c1.strip().upper() == c2.strip().upper()) else 0.0
        for c1, c2 in zip(s1_countries, c_countries)
    ]
    feat_name_token_overlap = [
        token_overlap_count(t1, t2)
        for t1, t2 in zip(tokens_s1_names, tokens_c_names)
    ]
    feat_address_token_overlap = [
        token_overlap_count(t1, t2)
        for t1, t2 in zip(tokens_s1_addrs, tokens_c_addrs)
    ]
    feat_name_length_similarity = [
        length_similarity(n1, n2)
        for n1, n2 in zip(norm_s1_names, norm_c_names)
    ]
    feat_address_length_similarity = [
        length_similarity(a1, a2)
        for a1, a2 in zip(norm_s1_addrs, norm_c_addrs)
    ]
    feat_name_levenshtein_similarity = [
        levenshtein_similarity(n1, n2)
        for n1, n2 in zip(norm_s1_names, norm_c_names)
    ]
    feat_address_levenshtein_similarity = [
        levenshtein_similarity(a1, a2)
        for a1, a2 in zip(norm_s1_addrs, norm_c_addrs)
    ]

    # 4. Compute TF-IDF Cosine Similarities (Vectorized batch transformation)
    # Name TF-IDF
    s1_name_vecs = name_vectorizer.transform(norm_s1_names)
    c_name_vecs = name_vectorizer.transform(norm_c_names)
    # Cosine similarity for L2-normalized sparse rows is the element-wise dot product
    name_dots = s1_name_vecs.multiply(c_name_vecs).sum(axis=1)
    feat_name_tfidf_cosine = [float(val) for val in np.asarray(name_dots).ravel()]

    # Address TF-IDF
    s1_addr_vecs = address_vectorizer.transform(norm_s1_addrs)
    c_addr_vecs = address_vectorizer.transform(norm_c_addrs)
    addr_dots = s1_addr_vecs.multiply(c_addr_vecs).sum(axis=1)
    feat_address_tfidf_cosine = [float(val) for val in np.asarray(addr_dots).ravel()]

    # 5. Assemble DataFrame strictly matching FEATURE_NAMES_13 order
    feature_data = {
        "name_exact": feat_name_exact,
        "name_similarity": feat_name_similarity,
        "address_exact": feat_address_exact,
        "address_similarity": feat_address_similarity,
        "country_same": feat_country_same,
        "name_token_overlap": feat_name_token_overlap,
        "address_token_overlap": feat_address_token_overlap,
        "name_length_similarity": feat_name_length_similarity,
        "address_length_similarity": feat_address_length_similarity,
        "name_levenshtein_similarity": feat_name_levenshtein_similarity,
        "address_levenshtein_similarity": feat_address_levenshtein_similarity,
        "name_tfidf_cosine": feat_name_tfidf_cosine,
        "address_tfidf_cosine": feat_address_tfidf_cosine,
    }

    feature_df = pd.DataFrame(feature_data, columns=FEATURE_NAMES_13)
    if len(feature_df) != n_rows:
        raise ValueError(
            f"Feature matrix row count ({len(feature_df)}) does not match input candidate count ({n_rows})."
        )
    return feature_df


# =============================================================================
# 4. Member 2 XGBoost 13-Feature Model Inference Adapter
# =============================================================================

class Member2XGBoost13FeatureAdapter:
    """
    Adapter for Member 2's 13-feature XGBoost matching model.

    Features:
    - Loads trained model artifact via joblib.
    - Uses model.feature_names_in_ to verify exact feature names and order.
    - Requires fitted TF-IDF vectorizers for name and address cosine features.
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
                model_path or "code/business_entity_resolution/models/matching_model.pkl"
            )
            fallback_path = Path("student_resource/dataset/train/analysis_model_xgboost_13features_realistic.pkl")
            m_path = primary_path if primary_path.is_file() else fallback_path

            if not m_path.is_file():
                raise FileNotFoundError(
                    f"Member 2 XGBoost model artifact not found at: '{primary_path}' "
                    f"(or fallback '{fallback_path}').\n"
                    f"Please provide the production model artifact."
                )
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
            vec_file = Path(vectorizers_path or "code/business_entity_resolution/models/tfidf_vectorizers.pkl")
            if vec_file.is_file():
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
                elif hasattr(loaded_vecs, "name_vectorizer") and hasattr(loaded_vecs, "address_vectorizer"):
                    name_vectorizer = getattr(loaded_vecs, "name_vectorizer")
                    address_vectorizer = getattr(loaded_vecs, "address_vectorizer")

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

        if MEMBER2_PROD_AVAILABLE:
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
            probs_arr = prod_inference.predict_batch(
                candidate_batch=candidate_batch_dicts,
                model=self.model,
                name_vectorizer=self.name_vectorizer,
                address_vectorizer=self.address_vectorizer,
            )
            probs = probs_arr.tolist()
        else:
            # Fallback feature calculation
            feature_df = compute_13_features_for_batch(
                candidate_batch=candidate_batch,
                catalog=self.records_catalog,
                name_vectorizer=self.name_vectorizer,
                address_vectorizer=self.address_vectorizer,
            )

            # Predict match probabilities
            if not hasattr(self.model, "predict_proba"):
                raise TypeError("Model artifact does not implement predict_proba.")

            raw_probs = self.model.predict_proba(feature_df)
            if hasattr(raw_probs, "ndim") and raw_probs.ndim == 2:
                probs = raw_probs[:, 1].tolist()
            elif hasattr(raw_probs, "ndim") and raw_probs.ndim == 1:
                probs = raw_probs.tolist()
            else:
                probs = list(raw_probs)

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
