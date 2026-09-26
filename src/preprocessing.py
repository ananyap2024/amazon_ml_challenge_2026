import re
import unicodedata


MISSING_TOKENS = {"nan", "none", "null"}


def normalize_text(value):
    """
    Unicode-aware text normalization.

    Preserves:
    - Unicode letters
    - Unicode numbers
    - Unicode combining marks

    Removes:
    - punctuation
    - symbols
    - common missing-value tokens
    """

    if value is None:
        return ""

    value = str(value).strip()

    # Unicode normalization
    value = unicodedata.normalize("NFKC", value)

    # Lowercase
    value = value.lower()

    # Keep letters, numbers, and marks.
    # Replace punctuation/symbols with spaces.
    cleaned = []

    for char in value:
        category = unicodedata.category(char)

        if (
            category.startswith("L")  # Letter
            or category.startswith("N")  # Number
            or category.startswith("M")  # Mark / combining character
        ):
            cleaned.append(char)
        else:
            cleaned.append(" ")

    value = "".join(cleaned)

    # Normalize whitespace
    value = re.sub(r"\s+", " ", value).strip()

    # Remove common missing-value tokens
    tokens = value.split()
    tokens = [
        token
        for token in tokens
        if token not in MISSING_TOKENS
    ]

    return " ".join(tokens)


def normalize_business_name(value):
    return normalize_text(value)


def normalize_address(value):
    return normalize_text(value)


def tokenize(value):
    text = normalize_text(value)

    if not text:
        return []

    return text.split()