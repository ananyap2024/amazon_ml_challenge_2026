import re
import unicodedata
from difflib import SequenceMatcher


def normalize_text(value):
    if value is None:
        return ""

    value = str(value)
    value = unicodedata.normalize("NFKC", value)
    value = value.lower()
    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def token_overlap(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union)


def length_similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    max_length = max(len(a), len(b))

    if max_length == 0:
        return 0.0

    return 1 - abs(len(a) - len(b)) / max_length


def create_pair_features(source1_row, candidate_row):

    name1 = normalize_text(source1_row["business_name"])
    name2 = normalize_text(candidate_row["business_name"])

    address1 = normalize_text(source1_row["business_address"])
    address2 = normalize_text(candidate_row["business_address"])

    country1 = normalize_text(source1_row["country"])
    country2 = normalize_text(candidate_row["country"])

    features = {

        # Existing features
        "name_exact": int(
            name1 == name2 and name1 != ""
        ),

        "name_similarity": similarity(
            name1,
            name2
        ),

        "address_exact": int(
            address1 == address2 and address1 != ""
        ),

        "address_similarity": similarity(
            address1,
            address2
        ),

        "country_same": int(
            country1 == country2 and country1 != ""
        ),

        # New features
        "name_token_overlap": token_overlap(
            name1,
            name2
        ),

        "address_token_overlap": token_overlap(
            address1,
            address2
        ),

        "name_length_similarity": length_similarity(
            name1,
            name2
        ),

        "address_length_similarity": length_similarity(
            address1,
            address2
        ),
    }

    return features