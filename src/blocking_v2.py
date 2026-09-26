import pandas as pd

from preprocessing import (
    normalize_business_name,
    tokenize,
)


# Tokens that are common across many business names
# and therefore are not useful as blocking keys.
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
}


def load_sample(file_path, nrows=10_000):
    """
    Load a small sample for blocking experiments.
    """

    return pd.read_csv(
        file_path,
        sep="\t",
        nrows=nrows
    )


def tokenize_name(name):
    """
    Convert a normalized business name into useful
    blocking tokens.

    Generic legal/business terms are removed.
    Very short tokens are ignored.
    """

    tokens = tokenize(name)

    useful_tokens = []

    for token in tokens:

        if token in GENERIC_NAME_TOKENS:
            continue

        if len(token) < 3:
            continue

        useful_tokens.append(token)

    return list(set(useful_tokens))


def prepare_dataframe(df):
    """
    Add normalized business name
    and name-token columns.
    """

    df = df.copy()

    df["name_norm"] = df["business_name"].map(
        normalize_business_name
    )

    df["name_tokens"] = df["name_norm"].map(
        tokenize_name
    )

    return df


def build_name_token_index(df):
    """
    Build an inverted index:

        token -> entity IDs

    Example:

        "zenith" -> {S2-123, S2-456}
        "telecom" -> {S2-123, S2-789}
    """

    index = {}

    for _, row in df.iterrows():

        entity_id = row["entity_id"]

        for token in row["name_tokens"]:

            if token not in index:
                index[token] = set()

            index[token].add(entity_id)

    return index


def build_token_frequency(index):
    """
    Calculate how many records contain each token.
    """

    return {
        token: len(entity_ids)
        for token, entity_ids in index.items()
    }


def select_distinctive_tokens(
    tokens,
    token_frequency,
    max_tokens=3,
    max_frequency=5000
):
    """
    Select up to max_tokens useful name tokens.

    A token is used only if it appears in at most
    max_frequency records.

    Among valid tokens, rarer tokens are preferred.
    """

    valid_tokens = [
        token
        for token in tokens
        if (
            token in token_frequency
            and token_frequency[token] <= max_frequency
        )
    ]

    valid_tokens.sort(
        key=lambda token: token_frequency[token]
    )

    return valid_tokens[:max_tokens]


def exact_name_candidates(
    s1_row,
    name_index
):
    """
    V1-style exact normalized-name candidates.
    """

    name = s1_row["name_norm"]

    if not name:
        return set()

    return name_index.get(
        name,
        set()
    )


def token_candidates(
    s1_row,
    token_index,
    token_frequency,
    max_tokens=3,
    max_frequency=5000
):
    """
    Generate candidates using distinctive and
    sufficiently rare business-name tokens.
    """

    candidates = set()

    selected_tokens = select_distinctive_tokens(
        s1_row["name_tokens"],
        token_frequency,
        max_tokens=max_tokens,
        max_frequency=max_frequency
    )

    for token in selected_tokens:

        candidates.update(
            token_index.get(
                token,
                set()
            )
        )

    return candidates


def build_exact_name_index(df):
    """
    Build:

        normalized_name -> entity IDs
    """

    index = {}

    for name, group in df.groupby("name_norm"):

        if not name:
            continue

        index[name] = set(
            group["entity_id"]
        )

    return index


def generate_candidates(s1_df, s2_df, s3_df, max_tokens=3, max_frequency=5000):

    print("Building S2 exact-name index...")
    s2_name_index = build_exact_name_index(s2_df)

    print("Building S3 exact-name index...")
    s3_name_index = build_exact_name_index(s3_df)

    print("Building S2 token index...")
    s2_token_index = build_name_token_index(s2_df)

    print("Building S3 token index...")
    s3_token_index = build_name_token_index(s3_df)

    print("Calculating S2 token frequencies...")
    s2_token_frequency = build_token_frequency(s2_token_index)

    print("Calculating S3 token frequencies...")
    s3_token_frequency = build_token_frequency(s3_token_index)

    print("Generating candidates for S1...")
    
    results = []

    for _, row in s1_df.iterrows():
        candidates = set()

        candidates.update(
            exact_name_candidates(row, s2_name_index)
        )

        candidates.update(
            exact_name_candidates(row, s3_name_index)
        )

        candidates.update(
            token_candidates(
                row,
                s2_token_index,
                s2_token_frequency,
                max_tokens=max_tokens,
                max_frequency=max_frequency
            )
        )

        candidates.update(
            token_candidates(
                row,
                s3_token_index,
                s3_token_frequency,
                max_tokens=max_tokens,
                max_frequency=max_frequency
            )
        )

        results.append({
            "source1_entity_id": row["entity_id"],
            "candidate_entity_ids": list(candidates)
        })

    return pd.DataFrame(results)