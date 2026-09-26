import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

GROUND_TRUTH = (
    BASE_DIR
    / "dataset"
    / "train"
    / "train_ground_truth.tsv"
)


def main():

    print("=" * 70)
    print("GROUND TRUTH ANALYSIS")
    print("=" * 70)

    print("\nLoading ground truth...")

    df = pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        usecols=[
            "source1_entity_id",
            "matched_entity_ids"
        ]
    )

    print(f"\nTotal S1 entities: {len(df):,}")

    # Count matched S2/S3 entities
    df["match_count"] = (
        df["matched_entity_ids"]
        .fillna("")
        .apply(
            lambda x: 0 if not x else len(x.split(","))
        )
    )

    print("\n" + "=" * 70)
    print("MATCH COUNT DISTRIBUTION")
    print("=" * 70)

    print(
        df["match_count"]
        .value_counts()
        .sort_index()
        .head(30)
    )

    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)

    print(df["match_count"].describe())

    zero_matches = (df["match_count"] == 0).sum()
    one_or_more = (df["match_count"] > 0).sum()

    print(
        f"\nS1 entities with ZERO matches: "
        f"{zero_matches:,}"
    )

    print(
        f"S1 entities with ONE OR MORE matches: "
        f"{one_or_more:,}"
    )

    print(
        f"Maximum matches for one S1 entity: "
        f"{df['match_count'].max()}"
    )

    print("\n" + "=" * 70)
    print("EXAMPLES WITH MULTIPLE MATCHES")
    print("=" * 70)

    multiple = df[df["match_count"] >= 3]

    print(
        multiple[
            [
                "source1_entity_id",
                "matched_entity_ids",
                "match_count"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()