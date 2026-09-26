from pathlib import Path

import pandas as pd

from blocking_v2 import (
    load_sample,
    prepare_dataframe,
    build_exact_name_index,
    build_name_token_index,
    build_token_frequency,
    exact_name_candidates,
    token_candidates,
)

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"


def main():

    print("=" * 70)
    print("BLOCKING V2 FREQUENCY EXPERIMENT")
    print("=" * 70)

    MAX_TOKENS = 3
    FREQUENCY_VALUES = [5000, 10000, 20000, 50000]

    # ---------------------------------------------------------
    # 1. Ground truth
    # ---------------------------------------------------------
    print("\nLoading ground truth...")

    ground_truth = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t"
    )

    print(f"Ground truth rows: {len(ground_truth):,}")

    ground_truth_map = dict(
        zip(
            ground_truth["source1_entity_id"],
            ground_truth["matched_entity_ids"]
        )
    )

    # ---------------------------------------------------------
    # 2. S1 validation sample
    # ---------------------------------------------------------
    print("\nLoading S1...")

    s1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t"
    )

    s1 = s1.sample(
        n=1000,
        random_state=42
    ).copy()

    print(f"S1 validation sample: {len(s1):,}")

    # ---------------------------------------------------------
    # 3. Full S2 and S3
    # ---------------------------------------------------------
    print("\nLoading S2...")

    s2 = pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t"
    )

    print(f"S2 rows: {len(s2):,}")

    print("\nLoading S3...")

    s3 = pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t"
    )

    print(f"S3 rows: {len(s3):,}")

    # ---------------------------------------------------------
    # 4. Prepare data
    # ---------------------------------------------------------
    print("\nPreparing data...")

    s1 = prepare_dataframe(s1)
    s2 = prepare_dataframe(s2)
    s3 = prepare_dataframe(s3)

    print("Preparation complete.")

    # ---------------------------------------------------------
    # 5. BUILD INDEXES ONLY ONCE
    # ---------------------------------------------------------
    print("\nBuilding indexes...")

    print("Building S2 exact-name index...")
    s2_name_index = build_exact_name_index(s2)

    print("Building S3 exact-name index...")
    s3_name_index = build_exact_name_index(s3)

    print("Building S2 token index...")
    s2_token_index = build_name_token_index(s2)

    print("Building S3 token index...")
    s3_token_index = build_name_token_index(s3)

    print("Calculating S2 token frequencies...")
    s2_token_frequency = build_token_frequency(s2_token_index)

    print("Calculating S3 token frequencies...")
    s3_token_frequency = build_token_frequency(s3_token_index)

    print("\nIndex building complete.")

    # ---------------------------------------------------------
    # 6. Run experiments
    # ---------------------------------------------------------

    results = []

    for max_frequency in FREQUENCY_VALUES:

        print("\n" + "=" * 70)
        print(f"TESTING MAX FREQUENCY = {max_frequency:,}")
        print("=" * 70)

        candidate_counts = []

        total_true_matches = 0
        retrieved_true_matches = 0
        missed_true_matches = 0

        rows_with_candidates = 0
        rows_with_missed_matches = 0

        for index, (_, row) in enumerate(s1.iterrows(), start=1):

            candidates = set()

            # Exact normalized name
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

            # Distinctive tokens
            candidates.update(
                token_candidates(
                    row,
                    s2_token_index,
                    s2_token_frequency,
                    max_tokens=MAX_TOKENS,
                    max_frequency=max_frequency
                )
            )

            candidates.update(
                token_candidates(
                    row,
                    s3_token_index,
                    s3_token_frequency,
                    max_tokens=MAX_TOKENS,
                    max_frequency=max_frequency
                )
            )

            candidate_counts.append(len(candidates))

            if candidates:
                rows_with_candidates += 1

            # Ground truth
            s1_id = row["entity_id"]

            true_matches_raw = ground_truth_map.get(
                s1_id,
                ""
            )

            if pd.isna(true_matches_raw):
                true_matches_raw = ""

            if true_matches_raw:
                true_matches = set(
                    str(true_matches_raw).split(",")
                )
            else:
                true_matches = set()

            total_true_matches += len(true_matches)

            retrieved = true_matches.intersection(candidates)

            retrieved_true_matches += len(retrieved)

            missed = true_matches - candidates

            missed_true_matches += len(missed)

            if missed:
                rows_with_missed_matches += 1

            # Progress
            if index % 100 == 0:
                print(
                    f"Processed {index:,}/{len(s1):,} S1 rows..."
                )

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        candidate_series = pd.Series(candidate_counts)

        recall = (
            retrieved_true_matches / total_true_matches
            if total_true_matches > 0
            else 0
        )

        print("\nRESULT")
        print("-" * 70)

        print(f"S1 validation rows:       {len(s1):,}")
        print(f"S1 rows with candidates:  {rows_with_candidates:,}")
        print(f"S1 rows with missed:      {rows_with_missed_matches:,}")

        print()
        print(f"True matches:              {total_true_matches:,}")
        print(f"Retrieved matches:         {retrieved_true_matches:,}")
        print(f"Missed matches:            {missed_true_matches:,}")

        print()
        print(f"Blocking recall:           {recall:.4f}")
        print(f"Blocking recall %:         {recall * 100:.2f}%")

        print()
        print(f"Mean candidates:           {candidate_series.mean():,.2f}")
        print(f"Median candidates:         {candidate_series.median():,.2f}")
        print(
            f"75th percentile:           "
            f"{candidate_series.quantile(0.75):,.2f}"
        )
        print(
            f"Maximum candidates:        "
            f"{candidate_series.max():,.0f}"
        )

        results.append({
            "max_frequency": max_frequency,
            "recall": recall,
            "recall_percent": recall * 100,
            "mean_candidates": candidate_series.mean(),
            "median_candidates": candidate_series.median(),
            "p75_candidates": candidate_series.quantile(0.75),
            "max_candidates": candidate_series.max(),
            "rows_with_candidates": rows_with_candidates,
            "rows_with_missed_matches": rows_with_missed_matches,
            "true_matches": total_true_matches,
            "retrieved_matches": retrieved_true_matches,
            "missed_matches": missed_true_matches,
        })

    # ---------------------------------------------------------
    # 7. Final comparison
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n\n")
    print("=" * 90)
    print("FINAL FREQUENCY COMPARISON")
    print("=" * 90)

    print(
        results_df[
            [
                "max_frequency",
                "recall_percent",
                "mean_candidates",
                "median_candidates",
                "max_candidates",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()