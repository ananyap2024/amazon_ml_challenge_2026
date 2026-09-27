import pandas as pd

from preprocessing import normalize_text, normalize_business_name, tokenize


GENERIC_NAME_TOKENS = {
    "private", "limited", "ltd", "llc", "inc",
    "incorporated", "corp", "corporation",
    "company", "co", "plc", "pvt", "public",
}

GENERIC_ADDRESS_TOKENS = {
    "road", "rd", "street", "st", "avenue", "ave",
    "lane", "ln", "building", "bldg", "floor", "fl",
    "block", "area", "district", "city", "state",
    "india", "usa", "us",
}


def prepare_dataframe(df):
    df = df.copy()

    df["name_norm"] = df["business_name"].map(
        normalize_business_name
    )

    df["address_norm"] = df["business_address"].map(
        normalize_text
    )

    df["country_norm"] = df["country"].map(
        normalize_text
    )

    df["name_tokens"] = df["name_norm"].map(
        tokenize_name
    )

    df["address_tokens"] = df["address_norm"].map(
        tokenize_address
    )

    return df


def tokenize_name(name):
    tokens = tokenize(name)

    return list({
        token
        for token in tokens
        if token not in GENERIC_NAME_TOKENS
        and len(token) >= 3
    })


def tokenize_address(address):
    tokens = tokenize(address)

    return list({
        token
        for token in tokens
        if token not in GENERIC_ADDRESS_TOKENS
        and len(token) >= 2
    })


def build_exact_name_index(df):
    index = {}

    for name, group in df.groupby("name_norm"):
        if not name:
            continue

        index[name] = set(group["entity_id"])

    return index


def build_name_token_index(df):
    index = {}

    for _, row in df.iterrows():
        entity_id = row["entity_id"]

        for token in row["name_tokens"]:
            if token not in index:
                index[token] = set()

            index[token].add(entity_id)

    return index


def build_country_address_token_index(df):
    index = {}

    for _, row in df.iterrows():

        country = row["country_norm"]
        entity_id = row["entity_id"]

        if not country:
            continue

        for token in row["address_tokens"]:

            key = (country, token)

            if key not in index:
                index[key] = set()

            index[key].add(entity_id)

    return index


def build_token_frequency(index):
    return {
        key: len(entity_ids)
        for key, entity_ids in index.items()
    }


def select_distinctive_tokens(
    tokens,
    token_frequency,
    max_tokens=3,
    max_frequency=10000
):
    valid_tokens = [
        token
        for token in tokens
        if token in token_frequency
        and token_frequency[token] <= max_frequency
    ]

    valid_tokens.sort(
        key=lambda token: token_frequency[token]
    )

    return valid_tokens[:max_tokens]


def select_distinctive_address_tokens(
    country,
    tokens,
    token_frequency,
    max_tokens=2,
    max_frequency=5000
):
    if not country:
        return []

    valid_tokens = []

    for token in tokens:

        key = (country, token)

        if (
            key in token_frequency
            and token_frequency[key] <= max_frequency
        ):
            valid_tokens.append(token)

    valid_tokens.sort(
        key=lambda token:
        token_frequency[(country, token)]
    )

    return valid_tokens[:max_tokens]


def generate_candidates(
    s1_df,
    s2_df,
    s3_df,
    name_max_tokens=3,
    name_max_frequency=10000,
    address_max_tokens=2,
    address_max_frequency=5000
):

    # =========================================================
    # NAME BLOCKING — SAME BASIC APPROACH AS V2
    # =========================================================

    print("Building S2 exact-name index...")
    s2_name_index = build_exact_name_index(s2_df)

    print("Building S3 exact-name index...")
    s3_name_index = build_exact_name_index(s3_df)

    print("Building S2 name-token index...")
    s2_name_token_index = build_name_token_index(s2_df)

    print("Building S3 name-token index...")
    s3_name_token_index = build_name_token_index(s3_df)

    print("Calculating name token frequencies...")

    s2_name_frequency = build_token_frequency(
        s2_name_token_index
    )

    s3_name_frequency = build_token_frequency(
        s3_name_token_index
    )

    # =========================================================
    # ADDRESS BLOCKING
    # Only country + address token
    # =========================================================

    print("Building S2 country-address-token index...")

    s2_address_index = build_country_address_token_index(
        s2_df
    )

    print("Building S3 country-address-token index...")

    s3_address_index = build_country_address_token_index(
        s3_df
    )

    print("Calculating address token frequencies...")

    s2_address_frequency = build_token_frequency(
        s2_address_index
    )

    s3_address_frequency = build_token_frequency(
        s3_address_index
    )

    print("All indexes built.")

    # =========================================================
    # CANDIDATE GENERATION
    # =========================================================

    results = []

    for index, (_, row) in enumerate(
        s1_df.iterrows(),
        start=1
    ):

        candidates = set()

        # -----------------------------------------------------
        # 1. Exact normalized business name
        # -----------------------------------------------------

        name = row["name_norm"]

        if name:
            candidates.update(
                s2_name_index.get(name, set())
            )

            candidates.update(
                s3_name_index.get(name, set())
            )

        # -----------------------------------------------------
        # 2. Distinctive business-name tokens
        # -----------------------------------------------------

        s2_name_tokens = select_distinctive_tokens(
            row["name_tokens"],
            s2_name_frequency,
            max_tokens=name_max_tokens,
            max_frequency=name_max_frequency
        )

        for token in s2_name_tokens:
            candidates.update(
                s2_name_token_index.get(
                    token,
                    set()
                )
            )

        s3_name_tokens = select_distinctive_tokens(
            row["name_tokens"],
            s3_name_frequency,
            max_tokens=name_max_tokens,
            max_frequency=name_max_frequency
        )

        for token in s3_name_tokens:
            candidates.update(
                s3_name_token_index.get(
                    token,
                    set()
                )
            )

        # -----------------------------------------------------
        # 3. Country + distinctive address tokens
        # -----------------------------------------------------

        country = row["country_norm"]

        s2_address_tokens = select_distinctive_address_tokens(
            country,
            row["address_tokens"],
            s2_address_frequency,
            max_tokens=address_max_tokens,
            max_frequency=address_max_frequency
        )

        for token in s2_address_tokens:
            candidates.update(
                s2_address_index.get(
                    (country, token),
                    set()
                )
            )

        s3_address_tokens = select_distinctive_address_tokens(
            country,
            row["address_tokens"],
            s3_address_frequency,
            max_tokens=address_max_tokens,
            max_frequency=address_max_frequency
        )

        for token in s3_address_tokens:
            candidates.update(
                s3_address_index.get(
                    (country, token),
                    set()
                )
            )

        results.append({
            "source1_entity_id": row["entity_id"],
            "candidate_entity_ids": list(candidates)
        })

        if index % 100 == 0:
            print(
                f"Processed "
                f"{index:,}/{len(s1_df):,} S1 rows..."
            )

    return pd.DataFrame(results)