"""Member 2 matching stage: score candidate pairs with the trained model.

Reads the internal candidate pairs (candidate_pairs_internal.tsv), looks up the
Source 1 / Source 2 / Source 3 records, computes the features from features.py,
scores every pair with the trained Logistic Regression model and writes EVERY
scored pair (not only the predicted matches), so Member 3 can consume the file
and the threshold can be changed later without recomputing features.

Pairs are processed in chunks and the results are streamed to disk, so memory
use depends on --chunk-size, not on the size of the candidate file.

Records are read with the same pandas defaults that were used when the training
features were built, and missing values are passed to create_pair_features
unchanged, so the features match what the model was trained on.

Example (development run on the first 1,000 pairs):

    python src/predict_matches.py \
        --pairs candidate_pairs_internal.tsv \
        --source1 dataset/test/test_source1.tsv \
        --source2 dataset/test/test_source2.tsv \
        --source3 dataset/test/test_source3.tsv \
        --model models/matching_model.pkl \
        --output matching_results_internal.tsv \
        --max-pairs 1000
"""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

# Make the sibling modules importable regardless of the working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import create_pair_features  # noqa: E402
from train_model import FEATURES  # noqa: E402


DEFAULT_THRESHOLD = 0.75
DEFAULT_CHUNK_SIZE = 1_000_000
SOURCE_READ_CHUNK_SIZE = 500_000
MAX_EXAMPLES = 5

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "matching_model.pkl"
)

PAIR_COLUMNS = ["source1_entity_id", "candidate_entity_id", "source"]
OUTPUT_COLUMNS = PAIR_COLUMNS + ["match_probability", "predicted_match"]
RECORD_COLUMNS = ["business_name", "business_address", "country"]
SOURCE_COLUMNS = ["entity_id"] + RECORD_COLUMNS
SOURCE_PREFIX = {"source2": "S2-", "source3": "S3-"}


def sha256_of_file(path):
    """Return the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def load_model(model_path):
    """Load the model and check it was trained on the expected features.

    Returns the model and the column of predict_proba that holds class 1.
    """

    model = joblib.load(model_path)

    model_features = list(getattr(model, "feature_names_in_", []))

    if model_features != FEATURES:
        raise ValueError(
            f"Model features {model_features} do not match the FEATURES "
            f"in train_model.py {FEATURES}."
        )

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(f"Model classes {classes} do not include class 1.")

    return model, classes.index(1)


def empty_records():
    """Return an empty records table with the same layout as load_records()."""

    return pd.DataFrame(
        columns=RECORD_COLUMNS,
        index=pd.Index([], name="entity_id"),
    )


def load_records(path, needed_ids, read_chunk_size=SOURCE_READ_CHUNK_SIZE):
    """Read a large source TSV in chunks and keep only the needed records.

    Returns the records indexed by entity_id and the number of duplicate
    entity_ids that were dropped (the first record of each ID is kept).
    """

    if not needed_ids:
        return empty_records(), 0

    parts = []

    for chunk in pd.read_csv(path, sep="\t", chunksize=read_chunk_size):
        missing = [c for c in SOURCE_COLUMNS if c not in chunk.columns]

        if missing:
            raise ValueError(
                f"{path} is missing columns {missing}; "
                f"found {list(chunk.columns)}."
            )

        parts.append(chunk[chunk["entity_id"].isin(needed_ids)])

    if not parts:
        return empty_records(), 0

    records = pd.concat(parts)

    duplicated = records["entity_id"].duplicated()
    records = records[~duplicated].set_index("entity_id")

    return records, int(duplicated.sum())


def fetch_records(records, ids):
    """Look up ids in a records table.

    Returns a found mask and, for each record column, an object array aligned
    with ids (None where the ID was not found).
    """

    positions = records.index.get_indexer(ids)
    found = positions >= 0

    values = {}

    for column in RECORD_COLUMNS:
        column_values = records[column].to_numpy(dtype=object)

        aligned = np.full(len(ids), None, dtype=object)
        aligned[found] = column_values[positions[found]]

        values[column] = aligned

    return found, values


def find_invalid_pairs(chunk):
    """Return the reason each pair is invalid ("" when the pair is fine)."""

    source1 = chunk["source1_entity_id"]
    candidate = chunk["candidate_entity_id"]
    source = chunk["source"]

    missing_field = ((source1 == "") | (candidate == "") | (source == ""))
    invalid_source = ~source.isin(list(SOURCE_PREFIX))
    bad_source1_prefix = ~source1.str.startswith("S1-")

    prefix_mismatch = np.zeros(len(chunk), dtype=bool)

    for name, prefix in SOURCE_PREFIX.items():
        prefix_mismatch |= (
            (source == name) & ~candidate.str.startswith(prefix)
        ).to_numpy()

    return np.select(
        [
            missing_field.to_numpy(),
            invalid_source.to_numpy(),
            bad_source1_prefix.to_numpy(),
            prefix_mismatch,
        ],
        [
            "missing_field",
            "invalid_source_value",
            "source1_id_bad_prefix",
            "candidate_id_prefix_mismatch",
        ],
        default="",
    )


def record_skipped(rows, reasons, skipped, examples):
    """Count skipped pairs by reason and keep a few examples of each."""

    for reason in np.unique(reasons):
        matching = rows[reasons == reason]

        reason = str(reason)
        skipped[reason] += len(matching)

        room = MAX_EXAMPLES - len(examples.setdefault(reason, []))

        if room > 0:
            examples[reason].extend(matching.head(room).to_dict("records"))


def compute_feature_matrix(source1_values, candidate_values):
    """Run create_pair_features on aligned records; return an (n, features) array."""

    n_pairs = len(source1_values[RECORD_COLUMNS[0]])
    matrix = np.empty((n_pairs, len(FEATURES)))

    source1_rows = zip(*(source1_values[c] for c in RECORD_COLUMNS))
    candidate_rows = zip(*(candidate_values[c] for c in RECORD_COLUMNS))

    for i, (source1_row, candidate_row) in enumerate(
        zip(source1_rows, candidate_rows)
    ):
        features = create_pair_features(
            dict(zip(RECORD_COLUMNS, source1_row)),
            dict(zip(RECORD_COLUMNS, candidate_row)),
        )

        matrix[i] = [features[name] for name in FEATURES]

    return matrix


def process_chunk(chunk, source_paths, model, positive_index, threshold, stats):
    """Score one chunk of pairs.

    Returns the scored pairs (or None when nothing in the chunk could be
    scored). Invalid pairs are counted in stats and skipped.
    """

    chunk = chunk.copy()
    chunk["input_row"] = chunk.index + 1

    stats["pairs_read"] += len(chunk)
    stats["duplicate_pairs_within_chunks"] += int(
        chunk.duplicated(subset=PAIR_COLUMNS[:2]).sum()
    )

    # 1. Skip pairs that are malformed or whose source does not fit the IDs.
    reasons = find_invalid_pairs(chunk)
    invalid = reasons != ""

    record_skipped(chunk[invalid], reasons[invalid], stats["skipped"], stats["examples"])

    pairs = chunk[~invalid].reset_index(drop=True)

    if pairs.empty:
        return None

    # 2. Load only the records this chunk needs.
    source1_records, duplicates = load_records(
        source_paths["source1"], set(pairs["source1_entity_id"])
    )
    stats["duplicate_source_ids_dropped"]["source1"] += duplicates

    candidate_records = {}

    for name in SOURCE_PREFIX:
        needed = set(
            pairs.loc[pairs["source"] == name, "candidate_entity_id"]
        )

        candidate_records[name], duplicates = load_records(
            source_paths[name], needed
        )
        stats["duplicate_source_ids_dropped"][name] += duplicates

    # 3. Attach the records to each pair.
    source1_found, source1_values = fetch_records(
        source1_records, pairs["source1_entity_id"]
    )

    candidate_found = np.zeros(len(pairs), dtype=bool)
    candidate_values = {
        column: np.full(len(pairs), None, dtype=object)
        for column in RECORD_COLUMNS
    }

    for name, records in candidate_records.items():
        in_source = (pairs["source"] == name).to_numpy()
        rows = np.flatnonzero(in_source)

        found, values = fetch_records(
            records, pairs.loc[in_source, "candidate_entity_id"]
        )

        candidate_found[rows[found]] = True

        for column in RECORD_COLUMNS:
            candidate_values[column][rows[found]] = values[column][found]

    # 4. Skip pairs whose IDs are not in the source files.
    keep = source1_found & candidate_found

    reasons = np.where(
        ~source1_found, "source1_id_not_found", "candidate_id_not_found"
    )

    record_skipped(pairs[~keep], reasons[~keep], stats["skipped"], stats["examples"])

    if not keep.any():
        return None

    # 5. Features, probabilities and the threshold decision.
    features = compute_feature_matrix(
        {column: source1_values[column][keep] for column in RECORD_COLUMNS},
        {column: candidate_values[column][keep] for column in RECORD_COLUMNS},
    )

    probabilities = model.predict_proba(
        pd.DataFrame(features, columns=FEATURES)
    )[:, positive_index]

    results = pairs.loc[keep, PAIR_COLUMNS].copy()
    results["match_probability"] = probabilities
    results["predicted_match"] = (probabilities >= threshold).astype(int)

    return results


def json_default(value):
    """Convert numpy scalars so they can be written to the JSON summary."""

    return value.item() if hasattr(value, "item") else str(value)


def predict_matches(
    pairs_path,
    source1_path,
    source2_path,
    source3_path,
    model_path,
    output_path,
    threshold=DEFAULT_THRESHOLD,
    max_pairs=None,
    chunk_size=DEFAULT_CHUNK_SIZE,
    summary_path=None,
):
    """Score every valid candidate pair and write the results plus a summary.

    The results file only appears under its final name once the whole run has
    finished; while running it is written to <output>.partial.
    Returns the run summary as a dictionary.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}.")

    if max_pairs is not None and max_pairs < 1:
        raise ValueError(f"max_pairs must be at least 1, got {max_pairs}.")

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be at least 1, got {chunk_size}.")

    pairs_path = Path(pairs_path)
    model_path = Path(model_path)
    output_path = Path(output_path)

    source_paths = {
        "source1": Path(source1_path),
        "source2": Path(source2_path),
        "source3": Path(source3_path),
    }

    if summary_path is None:
        summary_path = output_path.with_suffix(".summary.json")

    summary_path = Path(summary_path)

    inputs = {"pairs": pairs_path, "model": model_path, **source_paths}

    for label, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")

    header = list(pd.read_csv(pairs_path, sep="\t", nrows=0).columns)
    missing = [c for c in PAIR_COLUMNS if c not in header]

    if missing:
        raise ValueError(
            f"{pairs_path} is missing columns {missing}; found {header}."
        )

    started = time.time()

    model, positive_index = load_model(model_path)

    stats = {
        "pairs_read": 0,
        "pairs_scored": 0,
        "predicted_matches": 0,
        "duplicate_pairs_within_chunks": 0,
        "duplicate_source_ids_dropped": Counter(),
        "skipped": Counter(),
        "examples": {},
    }

    read_size = chunk_size if max_pairs is None else min(chunk_size, max_pairs)
    max_pairs_reached = False
    chunks = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".partial")

    print(f"Scoring {pairs_path} (threshold {threshold}, chunk size {read_size:,})")

    with open(partial_path, "w", encoding="utf-8", newline="") as output:
        output.write("\t".join(OUTPUT_COLUMNS) + "\n")

        with pd.read_csv(
            pairs_path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=read_size,
        ) as reader:

            for chunk in reader:

                if max_pairs is not None:
                    chunk = chunk.iloc[: max_pairs - stats["pairs_read"]]

                chunks += 1

                results = process_chunk(
                    chunk, source_paths, model, positive_index, threshold, stats
                )

                if results is not None:
                    results.to_csv(
                        output,
                        sep="\t",
                        header=False,
                        index=False,
                        lineterminator="\n",
                    )

                    stats["pairs_scored"] += len(results)
                    stats["predicted_matches"] += int(
                        results["predicted_match"].sum()
                    )

                elapsed = max(time.time() - started, 1e-9)

                print(
                    f"chunk {chunks:>4} | read {stats['pairs_read']:>12,} "
                    f"| scored {stats['pairs_scored']:>12,} "
                    f"| skipped {sum(stats['skipped'].values()):>8,} "
                    f"| {stats['pairs_read'] / elapsed:>8,.0f} pairs/s"
                )

                if max_pairs is not None and stats["pairs_read"] >= max_pairs:
                    max_pairs_reached = True
                    break

    os.replace(partial_path, output_path)

    elapsed = time.time() - started
    skipped_total = sum(stats["skipped"].values())

    summary = {
        "completed": True,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "inputs": {
            "pairs": str(pairs_path),
            "source1": str(source_paths["source1"]),
            "source2": str(source_paths["source2"]),
            "source3": str(source_paths["source3"]),
            "model": str(model_path),
            "model_sha256": sha256_of_file(model_path),
        },
        "settings": {
            "threshold": threshold,
            "max_pairs": max_pairs,
            "max_pairs_reached": max_pairs_reached,
            "chunk_size": read_size,
            "features": FEATURES,
        },
        "counts": {
            "chunks": chunks,
            "pairs_read": stats["pairs_read"],
            "pairs_scored": stats["pairs_scored"],
            "pairs_skipped": skipped_total,
            "predicted_matches": stats["predicted_matches"],
            "predicted_non_matches": stats["pairs_scored"] - stats["predicted_matches"],
            "duplicate_pairs_within_chunks": stats["duplicate_pairs_within_chunks"],
            "duplicate_source_ids_dropped": dict(stats["duplicate_source_ids_dropped"]),
        },
        "skipped_pairs": {
            "by_reason": dict(stats["skipped"]),
            "examples": {
                reason: rows for reason, rows in stats["examples"].items() if rows
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=json_default)

    print("\n" + "=" * 60)
    print("MATCHING COMPLETE")
    print("=" * 60)
    print("Pairs read:        ", f"{stats['pairs_read']:,}")
    print("Pairs scored:      ", f"{stats['pairs_scored']:,}")
    print("Predicted matches: ", f"{stats['predicted_matches']:,}")
    print("Pairs skipped:     ", f"{skipped_total:,}")
    print("\nResults saved to:")
    print(output_path)
    print("Summary saved to:")
    print(summary_path)

    if skipped_total:
        print(
            f"\nWARNING: {skipped_total:,} pairs were skipped "
            f"({dict(stats['skipped'])}). See the summary for examples."
        )

    return summary


def parse_args(argv=None):
    """Read the command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Score candidate pairs with the trained matching model."
    )

    parser.add_argument(
        "--pairs",
        required=True,
        help="candidate_pairs_internal.tsv "
        "(source1_entity_id, candidate_entity_id, source)",
    )
    parser.add_argument("--source1", required=True, help="Source 1 records (.tsv)")
    parser.add_argument("--source2", required=True, help="Source 2 records (.tsv)")
    parser.add_argument("--source3", required=True, help="Source 3 records (.tsv)")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="trained model file (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="internal matching-results file to write (.tsv)",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="run summary JSON (default: the --output name with .summary.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="match threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="only score the first N pairs (development dry run)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="pairs processed at a time; lower it to use less memory "
        "(default: %(default)s)",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    predict_matches(
        pairs_path=args.pairs,
        source1_path=args.source1,
        source2_path=args.source2,
        source3_path=args.source3,
        model_path=args.model,
        output_path=args.output,
        threshold=args.threshold,
        max_pairs=args.max_pairs,
        chunk_size=args.chunk_size,
        summary_path=args.summary,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
