# Business Entity Resolution Pipeline Package

This directory (`code/business_entity_resolution/`) contains the self-contained, reproducible codebase for our Amazon ML Challenge 2026 entity resolution solution.

---

## Directory Structure

```
code/business_entity_resolution/
├── README.md                      # Reproduction instructions (this file)
├── requirements.txt               # Pinned Python package dependencies
├── models/
│   ├── matching_model.pkl         # Serialized production XGBoost model (300 trees, depth 4)
│   └── tfidf_vectorizers.pkl      # Pre-fitted Name & Address TfidfVectorizer artifacts
└── src/
    ├── batch_inference.py         # Production streaming chunked batch inference driver
    ├── features.py                # 9 baseline string, length, overlap, and equality features
    ├── features_extra.py          # Levenshtein distance & TF-IDF cosine similarity functions
    ├── features_13.py             # Central 13-feature pipeline and matrix validation checks
    ├── inference.py               # Model loader, schema validator, and batch prediction API
    ├── tfidf_artifacts.py         # TF-IDF serialization and loading utilities
    ├── train_model.py             # Baseline model training and evaluation script
    ├── smoke_test_inference.py    # Schema verification and unit test suite
    └── smoke_test_batch_inference.py # End-to-end streaming batch integration test
```

---

## Environment Setup

Install the pinned dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Pinned Dependencies
- `numpy==2.2.6`
- `pandas==2.3.3`
- `scikit-learn==1.7.2`
- `nltk==3.9.3`
- `joblib==1.5.3`
- `xgboost==3.2.0`

---

## End-to-End Reproduction Workflow

### Step 1: Candidate Generation (Blocking - Member 1)
Run candidate blocking against test source files to produce the candidate pairs set:
- **Output candidate file:** `output/candidate_pairs.tsv`
- **Internal candidate file:** `dataset/test/candidate_pairs_internal.tsv`

### Step 2: Streaming Batch Model Inference (Matching - Member 2)
Run Member 2's streaming batch inference driver to compute match probabilities and thresholded predictions:

```bash
python src/batch_inference.py \
    --pairs ../../dataset/test/candidate_pairs_internal.tsv \
    --source1 ../../dataset/test/test_source1.tsv \
    --source2 ../../dataset/test/test_source2.tsv \
    --source3 ../../dataset/test/test_source3.tsv \
    --output ../../output/matching_results_scored.tsv \
    --threshold 0.60 \
    --chunk-size 10000
```

*Key Implementation Details:*
- Loads Source 1, Source 2, and Source 3 records once into memory hash lookups for $O(1)$ record access.
- Streams candidate pairs in configurable chunks (default 10,000), computing all 13 features and XGBoost match probabilities on the fly.
- Memory consumption remains constant (< 2 GB RAM) regardless of candidate file size.

### Step 3: Submission Formatting (Integration - Member 3)
Member 3 groups the scored predictions by `source1_entity_id`, creates comma-separated matched ID lists in `output/matching_results.tsv` (with empty strings for singletons), and verifies that every Source 1 entity from `test_source1.tsv` is represented.

### Step 4: Submission Validation
Run the competition validator to verify that `matching_results.tsv` and `candidate_pairs.tsv` comply with all submission rules:

```bash
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```

---

## Verification & Smoke Tests

Run the included smoke test suite to verify pipeline integrity:

```bash
# 1. Verify model loading, TF-IDF vectorizers, and feature schema
python src/smoke_test_inference.py

# 2. Verify end-to-end streaming batch inference, chunking, and thresholding
python src/smoke_test_batch_inference.py
```
