"""Fit, save, and load the two TF-IDF vectorizers (name, address) used by the
13-feature matching model.

The vectorizer configuration is preserved EXACTLY as used in the development
experiment (features_extra.fit_tfidf):

    TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
        norm="l2",
    )

How the vectorizers were originally fitted (for anyone reproducing them):
in the realistic 13-feature XGBoost experiment, the vectorizers were fit on
the ORIGINAL development training split's text (16,002 rows: the 80% split
of the artificial 50/50 development dataset), NOT on the realistic
2.4M-row candidate distribution and not on that experiment's own train/val
split. Each Source 1 record's and each candidate record's name/address text
was taken once (de-duplicated by entity_id), and the corpus was the
concatenation of the Source-1-side and candidate-side texts. This module
does not hard-code that corpus - fit_tfidf_vectorizers() takes whatever
text iterables the caller supplies, so it can be reused to refit later (on
a different corpus) or to reproduce the exact original one (by passing the
same de-duplicated development-split text).

Use joblib for serialization (already a project dependency for the saved
Logistic Regression / XGBoost models; no reason to introduce a second
serialization format for one more sklearn-derived object).

IMPORTANT: fit_tfidf_vectorizers() applies the same normalize_text() as
features.py/features_extra.py to every text before fitting - exactly what
features_extra.fit_tfidf() does. This is not optional preprocessing: a raw
pandas NaN is not a valid document for TfidfVectorizer, and skipping this
step (or leaving it to the caller) is exactly the mistake a first attempt at
this module made and the verification script's own crash on np.nan input
caught immediately.
"""

import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import normalize_text  # noqa: E402


def fit_tfidf_vectorizers(name_texts, address_texts):
    """Fit the name and address TF-IDF vectorizers.

    name_texts / address_texts: iterables of business name / address strings,
    exactly as they come from the source data (NaN/None are fine - each text
    is normalised via features.normalize_text() before fitting, same as
    features_extra.fit_tfidf()).

    Returns (name_vectorizer, address_vectorizer), both fitted, both using the
    exact configuration above.
    """

    def _fit_one(texts):
        vectorizer = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            lowercase=False,
            norm="l2",
        )
        vectorizer.fit([normalize_text(text) for text in texts])
        return vectorizer

    name_vectorizer = _fit_one(name_texts)
    address_vectorizer = _fit_one(address_texts)

    return name_vectorizer, address_vectorizer


def save_tfidf_vectorizers(name_vectorizer, address_vectorizer, path):
    """Persist both fitted vectorizers together in one joblib artifact.

    The artifact is a dict: {"name_vectorizer": ..., "address_vectorizer": ...}.
    A dict (not a bare tuple) is used so the artifact is self-describing when
    inspected later, and so a third vectorizer could be added in the future
    without breaking the format.
    """

    joblib.dump(
        {"name_vectorizer": name_vectorizer, "address_vectorizer": address_vectorizer},
        path,
    )


def load_tfidf_vectorizers(path):
    """Load both fitted vectorizers from a joblib artifact written by
    save_tfidf_vectorizers(). Returns (name_vectorizer, address_vectorizer).
    """

    artifact = joblib.load(path)

    missing = {"name_vectorizer", "address_vectorizer"} - set(artifact)
    if missing:
        raise ValueError(f"{path} is missing expected key(s): {sorted(missing)}")

    return artifact["name_vectorizer"], artifact["address_vectorizer"]
