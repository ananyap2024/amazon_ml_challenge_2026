"""Extra pair features for the Levenshtein / TF-IDF experiment.

Kept separate from features.py on purpose: the baseline features, and the saved
model that was trained on them, stay exactly as they are.

Every text is first normalized with the same normalize_text() as the baseline
features, so missing values behave as they do there (NaN becomes the text "nan").

The *_features() functions take a "pair text" table with the columns
    s1_business_name, cand_business_name,
    s1_business_address, cand_business_address
and return a table of new feature columns with the same index.
"""

import numpy as np
import pandas as pd
from nltk import edit_distance
from sklearn.feature_extraction.text import TfidfVectorizer

from features import normalize_text


LEVENSHTEIN_FEATURES = [
    "name_levenshtein_similarity",
    "address_levenshtein_similarity",
]

TFIDF_FEATURES = [
    "name_tfidf_cosine",
    "address_tfidf_cosine",
]


def levenshtein_similarity(a, b):
    """Normalized Levenshtein similarity: 1 - distance / max(len(a), len(b)).

    Both texts are normalized like the baseline features; 0.0 if either is empty.
    nltk's edit_distance is exact but pure Python (slow); a C implementation such
    as rapidfuzz returns identical values.
    """

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return 1 - edit_distance(a, b) / max(len(a), len(b))


def levenshtein_features(pair_text):
    """Normalized Levenshtein similarity of name and address for every pair."""

    return pd.DataFrame(
        {
            "name_levenshtein_similarity": [
                levenshtein_similarity(a, b)
                for a, b in zip(
                    pair_text["s1_business_name"],
                    pair_text["cand_business_name"],
                )
            ],
            "address_levenshtein_similarity": [
                levenshtein_similarity(a, b)
                for a, b in zip(
                    pair_text["s1_business_address"],
                    pair_text["cand_business_address"],
                )
            ],
        },
        index=pair_text.index,
    )


def fit_tfidf(texts):
    """Fit a word-level TF-IDF vectorizer on the given texts.

    Pass training-split texts only. Vectors are L2-normalized, so the dot product
    of two vectors is their cosine similarity. One-character tokens (for example
    house numbers) are kept.
    """

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
        norm="l2",
    )

    vectorizer.fit([normalize_text(text) for text in texts])

    return vectorizer


def tfidf_cosine(vectorizer, texts_a, texts_b):
    """TF-IDF cosine similarity between two aligned lists of texts (batch).

    0.0 when either text is empty. Words the vectorizer has not seen while being
    fitted are ignored.
    """

    matrix_a = vectorizer.transform([normalize_text(text) for text in texts_a])
    matrix_b = vectorizer.transform([normalize_text(text) for text in texts_b])

    cosine = np.asarray(matrix_a.multiply(matrix_b).sum(axis=1)).ravel()

    return np.clip(cosine, 0.0, 1.0)


def tfidf_features(pair_text, name_vectorizer, address_vectorizer):
    """TF-IDF cosine similarity of name and address for every pair."""

    return pd.DataFrame(
        {
            "name_tfidf_cosine": tfidf_cosine(
                name_vectorizer,
                pair_text["s1_business_name"],
                pair_text["cand_business_name"],
            ),
            "address_tfidf_cosine": tfidf_cosine(
                address_vectorizer,
                pair_text["s1_business_address"],
                pair_text["cand_business_address"],
            ),
        },
        index=pair_text.index,
    )
