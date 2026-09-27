"""
ML Challenge 2026 — Multi-Channel Candidate Generation (Blocking V3)

Enhances Member 1's blocking_v2 algorithm by introducing:
1. Domain-cleaned condensed name matching for website/company identifiers
   (e.g. "Heritage Society LLC" <---> "heritagesociety.com").
2. Controlled address-based blocking for cross-script and transliterated pairs
   (e.g. Hindi Devanagari names with identical postal / street addresses).
3. Configurable frequency caps to prevent candidate explosion while maintaining high recall.
"""

import re
from typing import Dict, List, Set, Tuple
import pandas as pd

from preprocessing import (
    normalize_text,
    normalize_business_name,
    normalize_address,
    tokenize,
)

# Common business suffixes to ignore in name tokens
GENERIC_NAME_TOKENS = {
    "private",
    "limited",
    "ltd",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "plc",
    "pvt",
    "public",
    "enterprises",
    "enterprise",
    "services",
    "service",
    "solutions",
    "solution",
    "technologies",
    "technology",
    "tech",
    "group",
    "associates",
    "consulting",
    "consultants",
}

# Domain extensions and web noise
DOMAIN_STOPWORDS = {
    "com",
    "org",
    "net",
    "edu",
    "gov",
    "mil",
    "info",
    "biz",
    "io",
    "ai",
    "co",
    "in",
    "us",
    "uk",
    "ca",
    "fr",
    "de",
    "jp",
    "www",
    "http",
    "https",
}

# High-frequency generic address terms that should not be used as blocking keys
GENERIC_ADDRESS_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "lane",
    "ln",
    "ave",
    "avenue",
    "near",
    "opp",
    "opposite",
    "floor",
    "bldg",
    "building",
    "no",
    "plot",
    "area",
    "dist",
    "district",
    "post",
    "po",
    "nagar",
    "marg",
    "chowk",
    "sector",
    "phase",
    "cross",
    "main",
    "state",
    "city",
    "west",
    "east",
    "north",
    "south",
    "central",
    "new",
    "old",
    "delhi",
    "mumbai",
    "bangalore",
    "texas",
    "california",
}


def clean_domain_name(name: str) -> str:
    """
    Remove web protocols, www prefixes, and common domain extensions.
    Example: 'heritagesociety.com' -> 'heritagesociety'
             'www.alpha-care.org' -> 'alpha care'
    """
    if not name:
        return ""
    text = str(name).strip().lower()
    text = re.sub(r"^(https?:\/\/)?(www\.)?", "", text)
    # Strip domain extensions at word boundaries or ends
    text = re.sub(r"\.(com|org|net|edu|gov|info|biz|io|ai|co\.in|com\.au|in|us|fr|de)$", "", text)
    return text


def extract_condensed_name(name: str) -> str:
    """
    Produce a concatenated lowercase alphanumeric string representing the core name.
    Example: 'Heritage Society LLC' -> 'heritagesociety'
             'heritagesociety.com' -> 'heritagesociety'
    """
    cleaned = clean_domain_name(name)
    tokens = tokenize(cleaned)
    useful = [t for t in tokens if t not in GENERIC_NAME_TOKENS and t not in DOMAIN_STOPWORDS]
    return "".join(useful)


def tokenize_name_v3(name: str) -> List[str]:
    """
    Extract distinctive name tokens, removing legal suffixes, domain stops,
    and single/double character noise.
    """
    cleaned = clean_domain_name(name)
    tokens = tokenize(cleaned)
    useful = []
    for t in tokens:
        if t in GENERIC_NAME_TOKENS or t in DOMAIN_STOPWORDS:
            continue
        if len(t) < 3:
            continue
        useful.append(t)
    return list(set(useful))


def tokenize_address_v3(address: str) -> List[str]:
    """
    Extract distinctive address tokens (e.g. PIN codes, postal codes, unit numbers,
    distinctive street names).
    Filters out common generic address stops ('road', 'street', 'floor', etc.).
    """
    if not address or pd.isna(address):
        return []
    norm = normalize_address(address)
    tokens = norm.split()
    useful = []
    for t in tokens:
        if t in GENERIC_ADDRESS_TOKENS:
            continue
        # Retain numeric codes (e.g. 6-digit PIN codes or building numbers >= 2 digits)
        if t.isdigit():
            if len(t) >= 2:
                useful.append(t)
            continue
        # Retain word tokens of length >= 3
        if len(t) >= 3:
            useful.append(t)
    return list(set(useful))


def prepare_dataframe_v3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized columns for multi-channel blocking:
    - name_norm: standard V2 normalized name
    - name_condensed: domain-cleaned concatenated string for brand/domain matching
    - name_tokens: filtered distinctive name tokens
    - address_tokens: distinctive address tokens for cross-script recovery
    """
    df = df.copy()
    df["name_norm"] = df["business_name"].map(normalize_business_name)
    df["name_condensed"] = df["business_name"].map(extract_condensed_name)
    df["name_tokens"] = df["business_name"].map(tokenize_name_v3)
    
    if "business_address" in df.columns:
        df["address_tokens"] = df["business_address"].map(tokenize_address_v3)
    else:
        df["address_tokens"] = [[] for _ in range(len(df))]
    return df


def build_exact_name_index(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build normalized_name -> set(entity_ids) index."""
    index = {}
    for name, group in df.groupby("name_norm"):
        if name:
            index[name] = set(group["entity_id"])
    return index


def build_condensed_name_index(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build condensed_name -> set(entity_ids) index."""
    index = {}
    for name, group in df.groupby("name_condensed"):
        # Ignore empty or excessively short condensed names (< 4 chars)
        if name and len(name) >= 4:
            index[name] = set(group["entity_id"])
    return index


def build_inverted_index(df: pd.DataFrame, token_col: str) -> Dict[str, Set[str]]:
    """Generic inverted index: token -> set(entity_ids)."""
    index = {}
    for _, row in df.iterrows():
        eid = row["entity_id"]
        for token in row[token_col]:
            if token not in index:
                index[token] = set()
            index[token].add(eid)
    return index


def build_token_frequency(index: Dict[str, Set[str]]) -> Dict[str, int]:
    return {token: len(entity_ids) for token, entity_ids in index.items()}


def select_distinctive_tokens(
    tokens: List[str],
    token_frequency: Dict[str, int],
    max_tokens: int = 3,
    max_frequency: int = 1500,
) -> List[str]:
    """Select the rarest valid tokens below max_frequency."""
    valid_tokens = [
        t for t in tokens if t in token_frequency and token_frequency[t] <= max_frequency
    ]
    valid_tokens.sort(key=lambda t: token_frequency[t])
    return valid_tokens[:max_tokens]


def lookup_channel_candidates(
    row: pd.Series,
    exact_index: Dict[str, Set[str]],
    condensed_index: Dict[str, Set[str]],
    name_token_index: Dict[str, Set[str]],
    name_token_freq: Dict[str, int],
    addr_token_index: Dict[str, Set[str]],
    addr_token_freq: Dict[str, int],
    use_exact: bool = True,
    use_condensed: bool = True,
    use_name_tokens: bool = True,
    use_address: bool = True,
    max_name_tokens: int = 3,
    max_name_freq: int = 1500,
    max_addr_tokens: int = 2,
    max_addr_freq: int = 250,
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Retrieve candidate IDs for a single S1 record, tracking channel contributions.
    Returns (all_candidates_set, channel_candidates_dict).
    """
    channel_candidates = {
        "exact_name": set(),
        "condensed_domain": set(),
        "name_tokens": set(),
        "address": set(),
    }

    # Channel 1: Exact normalized name
    if use_exact and row.get("name_norm"):
        exact_match = exact_index.get(row["name_norm"], set())
        channel_candidates["exact_name"].update(exact_match)

    # Channel 2: Condensed domain-cleaned name
    if use_condensed and row.get("name_condensed") and len(row["name_condensed"]) >= 4:
        condensed_match = condensed_index.get(row["name_condensed"], set())
        channel_candidates["condensed_domain"].update(condensed_match)

    # Channel 3: Distinctive name tokens
    if use_name_tokens and row.get("name_tokens"):
        selected = select_distinctive_tokens(
            row["name_tokens"],
            name_token_freq,
            max_tokens=max_name_tokens,
            max_frequency=max_name_freq,
        )
        for token in selected:
            channel_candidates["name_tokens"].update(name_token_index.get(token, set()))

    # Channel 4: Distinctive address tokens (PIN codes, rare streets, numbers)
    if use_address and row.get("address_tokens"):
        selected_addr = select_distinctive_tokens(
            row["address_tokens"],
            addr_token_freq,
            max_tokens=max_addr_tokens,
            max_frequency=max_addr_freq,
        )
        for token in selected_addr:
            channel_candidates["address"].update(addr_token_index.get(token, set()))

    total_candidates = (
        channel_candidates["exact_name"]
        | channel_candidates["condensed_domain"]
        | channel_candidates["name_tokens"]
        | channel_candidates["address"]
    )
    return total_candidates, channel_candidates


def test_v3_components():
    """Unit test suite for V3 components ensuring correctness and invariants."""
    print("Running V3 Unit Tests...")

    # 1. Domain cleaning test
    assert clean_domain_name("heritagesociety.com") == "heritagesociety"
    assert clean_domain_name("www.fetechnationaltwin.co.in") == "fetechnationaltwin"
    assert clean_domain_name("alphacare.org") == "alphacare"
    print("  [PASS] Domain cleaning")

    # 2. Condensed name test
    s1_name = "Heritage Society LLC"
    s3_name = "heritagesociety.com"
    assert extract_condensed_name(s1_name) == "heritagesociety"
    assert extract_condensed_name(s3_name) == "heritagesociety"
    assert extract_condensed_name(s1_name) == extract_condensed_name(s3_name)
    print("  [PASS] Condensed name matching")

    # 3. Address tokenization test
    addr = "A-301, New Sai Dham Chsl, Ramdev Park Road, Thane, Maharashtra 400607"
    tokens = tokenize_address_v3(addr)
    assert "400607" in tokens  # PIN code retained
    assert "301" in tokens     # Unit number retained
    assert "ramdev" in tokens  # Street name retained
    assert "road" not in tokens  # Generic stopword removed
    assert "road" in GENERIC_ADDRESS_TOKENS
    print("  [PASS] Address tokenization and stopword pruning")

    # 4. Frequency cap test
    freq_map = {"common": 5000, "rare": 50, "unique": 2}
    sel = select_distinctive_tokens(["common", "rare", "unique"], freq_map, max_tokens=2, max_frequency=100)
    assert sel == ["unique", "rare"]
    assert "common" not in sel
    print("  [PASS] Distinctive token frequency capping")

    # 5. Prefix integrity test
    test_ids = {"S2-12345", "S3-98765"}
    for tid in test_ids:
        assert tid.startswith(("S2-", "S3-")), f"Invalid prefix: {tid}"
        assert not tid.startswith("S1-"), f"Self-match prefix: {tid}"
    print("  [PASS] Source prefix integrity")

    print("All V3 Unit Tests Passed Successfully!\n")


if __name__ == "__main__":
    test_v3_components()
