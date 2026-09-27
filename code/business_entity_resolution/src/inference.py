"""Production XGBoost inference for the 13-feature matching model.

Self-contained under code/business_entity_resolution/ - imports only its
sibling modules (features.py, features_extra.py, features_13.py) and does
NOT depend on Notebooks/, project_paths.py, student_resource/, or any
analysis script.

Public API
----------
    load_model(model_path=DEFAULT_MODEL_PATH) -> XGBClassifier
    load_vectorizers(path=DEFAULT_TFIDF_PATH) -> (name_vectorizer, address_vectorizer)
    predict_batch(candidate_batch, model=None, name_vectorizer=None,
                  address_vectorizer=None) -> numpy.ndarray of float
    apply_threshold(probabilities, threshold) -> numpy.ndarray of int

No production threshold is chosen or hard-coded here: predict_batch() returns
probabilities only. apply_threshold() is a separate, explicit step - the
caller decides the threshold.

TF-IDF vectorizers are ALWAYS loaded from the artifact fitted during
development (see tfidf_artifacts.py); this module never fits or refits a
vectorizer at inference time.

Missing values in business_name / business_address / country follow the
SAME behaviour as the verified feature-generation code (features.py /
features_extra.py, unmodified, unchanged normalisation - including the
known "nan"-as-text quirk for missing values). Nothing new is invented here.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Make the sibling modules importable regardless of the caller's working
# directory - the same pattern already used by predict_matches.py.
_SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC_DIR))

from features_13 import (  # noqa: E402
    FEATURE_NAMES_13,
    RECORD_COLUMNS,
    compute_pair_features,
)

_MODELS_DIR = _SRC_DIR.parent / "models"
DEFAULT_MODEL_PATH = _MODELS_DIR / "matching_model.pkl"
DEFAULT_TFIDF_PATH = _MODELS_DIR / "tfidf_vectorizers.pkl"

PAIR_META_FIELDS = ["source1_entity_id", "candidate_entity_id", "source"]
RECORD_FIELDS = ["entity_id"] + RECORD_COLUMNS  # entity_id, business_name, business_address, country


def load_model(model_path=DEFAULT_MODEL_PATH):
    """Load the trained XGBoost model and validate its feature schema.

    Raises ValueError if the model's feature_names_in_ differ from
    FEATURE_NAMES_13 in name, order, or length - this is checked explicitly,
    not assumed.
    """

    model = joblib.load(model_path)

    actual = list(getattr(model, "feature_names_in_", []))

    if actual != FEATURE_NAMES_13:
        raise ValueError(
            "Model feature schema mismatch.\n"
            f"  expected (FEATURE_NAMES_13): {FEATURE_NAMES_13}\n"
            f"  found (model.feature_names_in_): {actual}"
        )

    return model


def load_vectorizers(path=DEFAULT_TFIDF_PATH):
    """Load the two fitted TF-IDF vectorizers. Never fits/refits anything."""

    artifact = joblib.load(path)

    missing = {"name_vectorizer", "address_vectorizer"} - set(artifact)
    if missing:
        raise ValueError(f"{path} is missing expected key(s): {sorted(missing)}")

    return artifact["name_vectorizer"], artifact["address_vectorizer"]


def _validate_candidate_batch(candidate_batch):
    """Check every item has the required structure; raise ValueError with the
    exact item index and missing field(s) on the first problem found."""

    for i, item in enumerate(candidate_batch):
        missing_top = [f for f in PAIR_META_FIELDS if f not in item]
        if missing_top:
            raise ValueError(f"candidate_batch[{i}] is missing field(s): {missing_top}")

        for side in ("source1_record", "candidate_record"):
            if side not in item:
                raise ValueError(f"candidate_batch[{i}] is missing field: '{side}'")

            missing_fields = [f for f in RECORD_FIELDS if f not in item[side]]
            if missing_fields:
                raise ValueError(
                    f"candidate_batch[{i}]['{side}'] is missing field(s): {missing_fields}"
                )


def predict_batch(candidate_batch, model=None, name_vectorizer=None, address_vectorizer=None):
    """Score a batch of Source 1 / candidate pairs and return match probabilities.

    Parameters
    ----------
    candidate_batch : list of dict
        Each item must have exactly this structure::

            {
                "source1_entity_id": str,
                "candidate_entity_id": str,
                "source": str,            # "source2" or "source3"
                "source1_record": {
                    "entity_id": str, "business_name": str or None,
                    "business_address": str or None, "country": str or None,
                },
                "candidate_record": {
                    "entity_id": str, "business_name": str or None,
                    "business_address": str or None, "country": str or None,
                },
            }

        Every field carries its own pair's information explicitly - no
        separate lookup table or hidden global state is consulted.
    model, name_vectorizer, address_vectorizer : optional
        Already-loaded objects. If omitted, they are loaded from
        DEFAULT_MODEL_PATH / DEFAULT_TFIDF_PATH via load_model() /
        load_vectorizers() (which also re-validates the feature schema).

    Returns
    -------
    numpy.ndarray of float, shape (len(candidate_batch),)
        probabilities[i] is the match probability for candidate_batch[i],
        in exactly the same order as the input. Never reorders, never drops
        a row - every input item produces exactly one output probability.

    Raises
    ------
    ValueError
        If any item in candidate_batch is missing a required field.
    """

    if len(candidate_batch) == 0:
        return np.array([], dtype=float)

    _validate_candidate_batch(candidate_batch)

    if model is None:
        model = load_model()
    if name_vectorizer is None or address_vectorizer is None:
        name_vectorizer, address_vectorizer = load_vectorizers()

    source1_records = pd.DataFrame(
        [item["source1_record"] for item in candidate_batch]
    )[RECORD_COLUMNS]
    candidate_records = pd.DataFrame(
        [item["candidate_record"] for item in candidate_batch]
    )[RECORD_COLUMNS]

    features = compute_pair_features(source1_records, candidate_records, name_vectorizer, address_vectorizer)

    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(features)[:, positive_index]

    if len(probabilities) != len(candidate_batch):
        raise RuntimeError(
            f"internal error: produced {len(probabilities)} probabilities "
            f"for {len(candidate_batch)} input pairs"
        )

    return probabilities


def apply_threshold(probabilities, threshold):
    """Convert probabilities to binary predictions at an explicit threshold.

    No default/final production threshold is defined by this module - the
    caller must always supply one.
    """

    return (np.asarray(probabilities) >= threshold).astype(int)
