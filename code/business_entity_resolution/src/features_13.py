"""Production-oriented 13-feature computation for the realistic-distribution
XGBoost model.

Reuses (does not modify, does not reimplement) the existing modules:
  - features.py         : the 9 baseline features (create_pair_features)
  - features_extra.py   : the 4 extra features (Levenshtein, TF-IDF)

Both keep their exact existing normalisation behaviour, INCLUDING the known
"nan" text quirk for missing values (a pandas NaN becomes the literal text
"nan" after str() + normalisation) - this is left exactly as-is because the
trained model learned on features computed that way; changing it would
silently shift what every feature means for missing data.

FEATURE_NAMES_13 is the single source of truth for feature order. Any code
that trains or loads a 13-feature model should use this list (or check a
model's `feature_names_in_` against it) rather than hard-coding its own copy.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the sibling modules importable regardless of the caller's working
# directory (same pattern already used by predict_matches.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import create_pair_features  # noqa: E402
from features_extra import levenshtein_features, tfidf_features  # noqa: E402


BASELINE_FEATURE_NAMES = [
    "name_exact",
    "name_similarity",
    "address_exact",
    "address_similarity",
    "country_same",
    "name_token_overlap",
    "address_token_overlap",
    "name_length_similarity",
    "address_length_similarity",
]

LEVENSHTEIN_FEATURE_NAMES = [
    "name_levenshtein_similarity",
    "address_levenshtein_similarity",
]

TFIDF_FEATURE_NAMES = [
    "name_tfidf_cosine",
    "address_tfidf_cosine",
]

# The exact, required output order - the single source of truth for this project.
FEATURE_NAMES_13 = BASELINE_FEATURE_NAMES + LEVENSHTEIN_FEATURE_NAMES + TFIDF_FEATURE_NAMES

RECORD_COLUMNS = ["business_name", "business_address", "country"]


def compute_pair_features(source1_records, candidate_records, name_vectorizer, address_vectorizer):
    """Compute all 13 features for a batch of Source 1 / candidate pairs.

    Parameters
    ----------
    source1_records, candidate_records : pandas.DataFrame
        Two DataFrames of EQUAL LENGTH, aligned row-by-row: row i of
        source1_records is paired with row i of candidate_records. Each must
        have the columns "business_name", "business_address", "country"
        (extra columns are ignored). This mirrors the aligned-arrays shape
        predict_matches.py already uses for the 9-feature case, so an
        inference function can be extended without restructuring its data.
    name_vectorizer, address_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
        Already fitted (see tfidf_artifacts.py). NEVER refit here or per
        batch - refitting on inference data would silently change what
        features 12-13 mean relative to what the model was trained on.

    Returns
    -------
    pandas.DataFrame
        Exactly the columns in FEATURE_NAMES_13, in that order, indexed like
        source1_records / candidate_records.
    """

    if len(source1_records) != len(candidate_records):
        raise ValueError(
            f"source1_records ({len(source1_records)} rows) and candidate_records "
            f"({len(candidate_records)} rows) must be the same length."
        )

    for name, frame in [("source1_records", source1_records), ("candidate_records", candidate_records)]:
        missing = [c for c in RECORD_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{name} is missing column(s): {missing}")

    index = source1_records.index

    # --- the 9 baseline features (features.py, unmodified, unchanged normalisation) ---
    baseline_rows = [
        create_pair_features(s1_row, cand_row)
        for s1_row, cand_row in zip(
            source1_records[RECORD_COLUMNS].to_dict("records"),
            candidate_records[RECORD_COLUMNS].to_dict("records"),
        )
    ]
    baseline_df = pd.DataFrame(baseline_rows, index=index)[BASELINE_FEATURE_NAMES]

    # --- the 4 extra features (features_extra.py, unmodified) ---
    # levenshtein_features / tfidf_features expect this exact column shape.
    pair_text = pd.DataFrame(
        {
            "s1_business_name": source1_records["business_name"].to_numpy(),
            "cand_business_name": candidate_records["business_name"].to_numpy(),
            "s1_business_address": source1_records["business_address"].to_numpy(),
            "cand_business_address": candidate_records["business_address"].to_numpy(),
        },
        index=index,
    )

    lev_df = levenshtein_features(pair_text)[LEVENSHTEIN_FEATURE_NAMES]
    tfidf_df = tfidf_features(pair_text, name_vectorizer, address_vectorizer)[TFIDF_FEATURE_NAMES]

    features = pd.concat([baseline_df, lev_df, tfidf_df], axis=1)

    return features[FEATURE_NAMES_13]


def validate_feature_matrix(features_df):
    """Self-test / validation for a feature matrix produced by
    compute_pair_features(). Raises AssertionError with a specific message on
    the first failing check; returns True if every check passes.

    Checks: exactly 13 columns, in exactly the required order, all values
    finite, and no ID/label-like columns present.
    """

    columns = list(features_df.columns)

    assert len(columns) == 13, f"expected exactly 13 feature columns, found {len(columns)}: {columns}"
    assert columns == FEATURE_NAMES_13, (
        f"feature columns are not in the required order.\n  expected: {FEATURE_NAMES_13}\n  found:    {columns}"
    )

    disallowed = {"source1_entity_id", "candidate_entity_id", "source", "label", "entity_id"}
    leaked = disallowed & set(columns)
    assert not leaked, f"feature matrix must not contain ID/label columns; found: {sorted(leaked)}"

    values = features_df.to_numpy(dtype=float)
    assert np.isfinite(values).all(), (
        f"feature matrix contains non-finite values (NaN/inf) in "
        f"{int((~np.isfinite(values)).sum())} of {values.size} cells"
    )

    return True
