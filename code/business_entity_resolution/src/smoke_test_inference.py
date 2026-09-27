"""Smoke test for inference.py's predict_batch().

Analysis-only: reads a handful of real rows from the already-verified
realistic candidate dataset (student_resource/), builds the candidate_batch
input schema inference.predict_batch() expects, and checks that everything
runs end-to-end and returns sane, correctly-ordered output.

This script (unlike inference.py itself) is allowed to depend on
Notebooks/project_paths.py and student_resource/ - see inference.py's own
docstring for why that module stays free of those dependencies.

Does NOT choose a production threshold, does NOT modify any existing file,
does NOT touch the test candidate file, does NOT retrain anything.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
_NOTEBOOKS_DIR = Path(__file__).resolve().parents[4] / "Notebooks"
sys.path.insert(0, str(_SRC_DIR))
sys.path.insert(0, str(_NOTEBOOKS_DIR))

from project_paths import (  # noqa: E402
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_CANDIDATE_SAMPLE_1000_LABELED,
)

import inference  # noqa: E402
from features_13 import RECORD_COLUMNS  # noqa: E402

N_ROWS = 20
RANDOM_STATE = 42


def load_needed_rows(path, needed_ids, chunksize=500_000):
    parts = [c[c.entity_id.isin(needed_ids)] for c in pd.read_csv(path, sep="\t", chunksize=chunksize)]
    return pd.concat(parts).set_index("entity_id")


def build_candidate_batch(sample, source1_lookup, candidate_lookup):
    batch = []
    for row in sample.itertuples(index=False):
        s1 = source1_lookup.loc[row.source1_entity_id]
        cand = candidate_lookup.loc[row.candidate_entity_id]
        batch.append(
            {
                "source1_entity_id": row.source1_entity_id,
                "candidate_entity_id": row.candidate_entity_id,
                "source": row.source,
                "source1_record": {"entity_id": row.source1_entity_id, **{c: s1[c] for c in RECORD_COLUMNS}},
                "candidate_record": {"entity_id": row.candidate_entity_id, **{c: cand[c] for c in RECORD_COLUMNS}},
            }
        )
    return batch


def main():
    print("=" * 70)
    print("SMOKE TEST: inference.predict_batch()")
    print("=" * 70)

    print(f"\nLoading model from {inference.DEFAULT_MODEL_PATH} ...")
    model = inference.load_model()
    print("  load_model() passed feature_names_in_ validation against FEATURE_NAMES_13")

    print(f"Loading TF-IDF vectorizers from {inference.DEFAULT_TFIDF_PATH} ...")
    name_vectorizer, address_vectorizer = inference.load_vectorizers()
    print("  load_vectorizers() OK")

    print(f"\nSampling {N_ROWS} real rows (random_state={RANDOM_STATE}, stratified by label "
          f"so both classes are represented - true matches are ~0.1% of rows) from "
          f"{TRAIN_CANDIDATE_SAMPLE_1000_LABELED} ...")
    labeled = pd.read_csv(TRAIN_CANDIDATE_SAMPLE_1000_LABELED, sep="\t")
    n_pos = N_ROWS // 2
    n_neg = N_ROWS - n_pos
    positives = labeled[labeled["label"] == 1].sample(n=n_pos, random_state=RANDOM_STATE)
    negatives = labeled[labeled["label"] == 0].sample(n=n_neg, random_state=RANDOM_STATE)
    sample = pd.concat([positives, negatives]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    candidate_ids = sample["candidate_entity_id"]
    source1_lookup = load_needed_rows(TRAIN_SOURCE1, set(sample["source1_entity_id"]))
    source2_lookup = load_needed_rows(TRAIN_SOURCE2, set(candidate_ids[candidate_ids.str.startswith("S2-")]))
    source3_lookup = load_needed_rows(TRAIN_SOURCE3, set(candidate_ids[candidate_ids.str.startswith("S3-")]))
    candidate_lookup = pd.concat([source2_lookup, source3_lookup])

    candidate_batch = build_candidate_batch(sample, source1_lookup, candidate_lookup)

    print("\n--- Test 1: empty batch ---")
    empty_result = inference.predict_batch([])
    assert isinstance(empty_result, np.ndarray) and len(empty_result) == 0
    print(f"  predict_batch([]) -> {empty_result!r} (OK: empty array, no crash)")

    print("\n--- Test 2: missing-field validation ---")
    broken_item = dict(candidate_batch[0])
    broken_item = {k: v for k, v in broken_item.items() if k != "source"}
    try:
        inference.predict_batch([broken_item], model, name_vectorizer, address_vectorizer)
        print("  FAILED: expected ValueError, none was raised")
    except ValueError as e:
        print(f"  predict_batch() correctly raised ValueError: {e}")

    print(f"\n--- Test 3: real {N_ROWS}-row batch ---")
    probabilities = inference.predict_batch(candidate_batch, model, name_vectorizer, address_vectorizer)
    assert len(probabilities) == len(candidate_batch), "order/length preservation failed"
    assert np.isfinite(probabilities).all(), "non-finite probability produced"
    assert ((probabilities >= 0) & (probabilities <= 1)).all(), "probability out of [0, 1] range"
    print(f"  predict_batch() returned {len(probabilities)} probabilities, "
          f"all finite and in [0, 1]. Order/length preserved.")

    print("\n  idx  source1_entity_id   candidate_entity_id   source    label  probability")
    for i, (item, prob) in enumerate(zip(candidate_batch, probabilities)):
        true_label = sample.loc[i, "label"]
        print(f"  {i:3d}  {item['source1_entity_id']:<18} {item['candidate_entity_id']:<20} "
              f"{item['source']:<8}  {true_label:5d}  {prob:.4f}")

    pos_mask = sample["label"].to_numpy() == 1
    neg_mask = ~pos_mask
    mean_pos = float(probabilities[pos_mask].mean()) if pos_mask.any() else float("nan")
    mean_neg = float(probabilities[neg_mask].mean()) if neg_mask.any() else float("nan")
    print(f"\n  mean probability on true-label=1 rows: {mean_pos:.4f}")
    print(f"  mean probability on true-label=0 rows: {mean_neg:.4f}")
    assert mean_pos > mean_neg, (
        "model does not discriminate: mean probability on true matches is not higher "
        "than on true non-matches"
    )
    print("  OK: model assigns higher probability to true matches than to true non-matches")

    print("\n--- Test 4: apply_threshold() (illustrative only, NOT a production threshold) ---")
    illustrative_threshold = 0.5
    predictions = inference.apply_threshold(probabilities, illustrative_threshold)
    print(f"  apply_threshold(probabilities, {illustrative_threshold}) -> {predictions.tolist()}")

    print("\n--- Test 5: re-loading model/vectorizers on demand (no pre-loaded args) ---")
    probabilities_reload = inference.predict_batch(candidate_batch)
    assert np.allclose(probabilities, probabilities_reload), (
        "predict_batch() gave different results with vs without pre-loaded model/vectorizers"
    )
    print("  predict_batch(candidate_batch) with default (auto-loaded) model/vectorizers "
          "matches the pre-loaded-args result exactly")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
