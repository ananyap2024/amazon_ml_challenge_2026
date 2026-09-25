import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "dataset" / "train"


def main():

    print("=" * 70)
    print("GROUND TRUTH ALIGNMENT CHECK")
    print("=" * 70)

    s1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t",
        nrows=10
    )

    gt = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t",
        nrows=10
    )

    print("\nS1 columns:")
    print(s1.columns.tolist())

    print("\nGround truth columns:")
    print(gt.columns.tolist())

    print("\nFirst 10 S1 IDs:")
    print(
        s1["entity_id"].to_string(index=False)
    )

    print("\nFirst 10 ground-truth S1 IDs:")
    print(
        gt["source1_entity_id"].to_string(index=False)
    )

    print("\nFirst 10 ground-truth matches:")
    print(
        gt["matched_entity_ids"].to_string(index=False)
    )

    print("\nData types:")

    print(
        "\nS1 entity_id:",
        s1["entity_id"].dtype
    )

    print(
        "GT source1_entity_id:",
        gt["source1_entity_id"].dtype
    )

    print("\nDirect row-by-row comparison:")

    for i in range(min(len(s1), len(gt))):

        s1_id = str(s1.iloc[i]["entity_id"]).strip()
        gt_id = str(
            gt.iloc[i]["source1_entity_id"]
        ).strip()

        print(
            f"{i}: "
            f"S1={s1_id} | "
            f"GT={gt_id} | "
            f"Match={s1_id == gt_id}"
        )


if __name__ == "__main__":
    main()