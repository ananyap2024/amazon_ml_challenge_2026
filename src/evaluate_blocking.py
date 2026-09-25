import pandas as pd
from pathlib import Path

from preprocessing import normalize_business_name


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"


def normalize_df(df):
    """
    Add normalized business-name column.
    """

    df = df.copy()

    df["name_norm"] = df["business_name"].map(
        normalize_business_name
    )

    return df


def build_name_index(df):
    """
    Build:
        normalized_name -> set of entity IDs
    """

    index = {}

    for name, group in df.groupby("name_norm"):

        if not name:
            continue

        index[name] = set(
            group["entity_id"]
        )

    return index


def parse_ground_truth(value):
    """
    Convert comma-separated matched IDs
    into a Python set.
    """

    if pd.isna(value):
        return set()

    value = str(value).strip()

    if not value:
        return set()

    return {
        entity_id.strip()
        for entity_id in value.split(",")
        if entity_id.strip()
    }


def main():

    print("=" * 70)
    print("BLOCKING RECALL TEST")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load full ground truth
    # --------------------------------------------------

    print("\nLoading ground truth...")

    ground_truth = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t"
    )

    print(
        f"Ground truth rows: "
        f"{len(ground_truth):,}"
    )

    # Create:
    #
    # S1 entity ID
    #       ↓
    # matched S2/S3 IDs
    #
    # dictionary for fast lookup.

    ground_truth_map = dict(
        zip(
            ground_truth["source1_entity_id"],
            ground_truth["matched_entity_ids"]
        )
    )

    # --------------------------------------------------
    # 2. Load S1
    # --------------------------------------------------

    print("\nLoading S1...")

    s1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t"
    )

    # Random validation sample
    #
    # We use a fixed random_state so that
    # the same 1,000 records are selected
    # every time we run the experiment.

    s1 = s1.sample(
        n=1000,
        random_state=42
    )

    print(
        f"S1 validation sample: "
        f"{len(s1):,}"
    )

    # --------------------------------------------------
    # 3. Load S2
    # --------------------------------------------------

    print("\nLoading S2...")

    s2 = pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )

    print(
        f"S2 rows: "
        f"{len(s2):,}"
    )

    # --------------------------------------------------
    # 4. Load S3
    # --------------------------------------------------

    print("\nLoading S3...")

    s3 = pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )

    print(
        f"S3 rows: "
        f"{len(s3):,}"
    )

    # --------------------------------------------------
    # 5. Normalize business names
    # --------------------------------------------------

    print("\nNormalizing business names...")

    s1 = normalize_df(s1)
    s2 = normalize_df(s2)
    s3 = normalize_df(s3)

    print("Normalization complete.")

    # --------------------------------------------------
    # 6. Build exact-name indexes
    # --------------------------------------------------

    print("\nBuilding name indexes...")

    s2_index = build_name_index(s2)

    print(
        f"S2 unique normalized names: "
        f"{len(s2_index):,}"
    )

    s3_index = build_name_index(s3)

    print(
        f"S3 unique normalized names: "
        f"{len(s3_index):,}"
    )

    # --------------------------------------------------
    # 7. Evaluate blocking recall
    # --------------------------------------------------

    print("\nEvaluating blocking...")

    total_true_matches = 0
    retrieved_matches = 0

    rows_with_missed_matches = 0

    rows_with_true_matches = 0

    rows_with_candidates = 0

    for _, row in s1.iterrows():

        s1_id = str(
            row["entity_id"]
        ).strip()

        # ----------------------------------------------
        # Get ground-truth matches
        # ----------------------------------------------

        if s1_id not in ground_truth_map:
            continue

        true_matches = parse_ground_truth(
            ground_truth_map[s1_id]
        )

        # ----------------------------------------------
        # Candidate generation
        # ----------------------------------------------

        candidates = set()

        name = row["name_norm"]

        if name:

            # S2 candidates
            candidates.update(
                s2_index.get(
                    name,
                    set()
                )
            )

            # S3 candidates
            candidates.update(
                s3_index.get(
                    name,
                    set()
                )
            )

        # ----------------------------------------------
        # Statistics
        # ----------------------------------------------

        if candidates:
            rows_with_candidates += 1

        if true_matches:
            rows_with_true_matches += 1

        # Which true matches were retrieved?
        found_matches = (
            true_matches & candidates
        )

        total_true_matches += len(
            true_matches
        )

        retrieved_matches += len(
            found_matches
        )

        if (
            true_matches
            and len(found_matches)
            < len(true_matches)
        ):
            rows_with_missed_matches += 1

    # --------------------------------------------------
    # 8. Calculate recall
    # --------------------------------------------------

    if total_true_matches > 0:

        recall = (
            retrieved_matches
            / total_true_matches
        )

    else:

        recall = 0.0

    # --------------------------------------------------
    # 9. Print results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"S1 validation rows: "
        f"{len(s1):,}"
    )

    print(
        f"S1 rows with true matches: "
        f"{rows_with_true_matches:,}"
    )

    print(
        f"S1 rows with candidates: "
        f"{rows_with_candidates:,}"
    )

    print(
        f"S1 rows with missed matches: "
        f"{rows_with_missed_matches:,}"
    )

    print()

    print(
        f"True matches:       "
        f"{total_true_matches:,}"
    )

    print(
        f"Retrieved matches:  "
        f"{retrieved_matches:,}"
    )

    print(
        f"Missed matches:     "
        f"{total_true_matches - retrieved_matches:,}"
    )

    print()

    print(
        f"Blocking recall:    "
        f"{recall:.4f}"
    )

    print(
        f"Blocking recall %:  "
        f"{recall * 100:.2f}%"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()