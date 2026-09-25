import pandas as pd

from preprocessing import (
    normalize_business_name,
    normalize_address,
)


def load_sample(file_path, nrows=10_000):
    """
    Load a small sample for blocking experiments.
    """

    return pd.read_csv(
        file_path,
        sep="\t",
        nrows=nrows
    )


def prepare_dataframe(df):
    """
    Add normalized name and address columns.
    """

    df = df.copy()

    df["name_norm"] = df["business_name"].map(
        normalize_business_name
    )

    df["address_norm"] = df["business_address"].map(
        normalize_address
    )

    return df


def build_name_index(df):
    """
    Build:
        normalized_name -> entity IDs
    """

    index = {}

    for name, group in df.groupby("name_norm"):

        if not name:
            continue

        index[name] = group["entity_id"].tolist()

    return index


def exact_name_candidates(s1_row, name_index):
    """
    Return candidates having exactly the same
    normalized business name.
    """

    name = s1_row["name_norm"]

    if not name:
        return []

    return name_index.get(name, [])


def generate_candidates(s1_df, s2_df, s3_df):

    s2_name_index = build_name_index(s2_df)
    s3_name_index = build_name_index(s3_df)

    results = []

    for _, row in s1_df.iterrows():

        candidates = set()

        candidates.update(
            exact_name_candidates(
                row,
                s2_name_index
            )
        )

        candidates.update(
            exact_name_candidates(
                row,
                s3_name_index
            )
        )

        results.append({
            "source1_entity_id": row["entity_id"],
            "candidate_entity_ids": list(candidates)
        })

    return pd.DataFrame(results)