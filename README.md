# Amazon ML Challenge 2026: Business Entity Resolution

[![Model](https://img.shields.io/badge/Model-XGBoost-blue.svg)](https://xgboost.readthedocs.io/)
[![Validation](https://img.shields.io/badge/5--Fold%20CV%20Macro%20F0.5-0.7283-success.svg)](#offline-validation--evaluation-results)
[![License](https://img.shields.io/badge/License-MIT%2FApache%202.0%20Compatible-green.svg)](#license--fair-play-compliance)

An end-to-end, high-precision Machine Learning solution for the **Amazon ML Challenge 2026: Business Entity Resolution**.

This repository links noisy, unstandardized business identity records from two independent sources (**Source 2** and **Source 3**) to a deduplicated reference source (**Source 1**) across diverse international geographies under severe scale, extreme class imbalance, and strict precision constraints.

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Team Structure & Roles](#team-structure--roles)
3. [Repository Layout](#repository-layout)
4. [Methodology & Architecture](#methodology--architecture)
   - [Stage 1: Candidate Generation (Blocking)](#stage-1-candidate-generation-blocking)
   - [Stage 2: 13-Feature Engineering](#stage-2-13-feature-engineering)
   - [Stage 3: XGBoost Pairwise Classifier](#stage-3-xgboost-pairwise-classifier)
   - [Stage 4: Streaming/Chunked Batch Inference](#stage-4-streamingchunked-batch-inference)
5. [Offline Validation & Evaluation Results](#offline-validation--evaluation-results)
6. [Reproducibility & Execution Guide](#reproducibility--execution-guide)
   - [1. Environment Setup](#1-environment-setup)
   - [2. Running Candidate Generation](#2-running-candidate-generation)
   - [3. Running Streaming Batch Inference](#3-running-streaming-batch-inference)
   - [4. Formatting & Validating Submissions](#4-formatting--validating-submissions)
7. [Smoke Tests & Artifact Verification](#smoke-tests--artifact-verification)
8. [License & Fair Play Compliance](#license--fair-play-compliance)

---

## Executive Summary

- **Decoupled Two-Stage Architecture:** Combines token-frequency-calibrated inverted index blocking (reducing over 50 trillion candidate pairs down to a high-recall candidate set) with a 13-feature gradient-boosted decision tree (XGBoost) classifier.
- **Offline Validation Performance:** Achieved an **OFFLINE 5-fold GroupKFold Macro Entity-Level $F_{0.5}$ validation score of 0.7283 (72.83%)** with a Macro Entity Precision of **0.7495 (74.95%)** on 2,408,157 realistic candidate pairs (evaluated at the primary validation-supported threshold candidate of **0.60**).
- **Constant-Memory Streaming Batch Inference:** Designed a streaming batch inference driver (`batch_inference.py`) capable of processing large candidate spaces in configurable chunks (default 10,000) using $O(1)$ memory lookups of pre-indexed source records.
- **Open-Set Geographic Safety:** Accommodates country distribution shifts (such as the unseen `France` test set label) via dynamic string equality comparison (`country_same`) rather than brittle closed-world encodings.
- **Strict Fair Play & Open Source Compliance:** Uses only MIT/Apache 2.0 open-source libraries; completely offline with zero external lookups, geocoding APIs, or commercial web services.

---

## Team Structure & Roles

**Team Name:** GMPA

- **Pappu Ananya (Candidate Generation & Preprocessing):** Unicode NFKC text normalization, generic business stop-token purging, inverted token indexing with token-frequency thresholds (`max_frequency = 5000`), exact normalized name indexing, and candidate pair generation.
- **Gorla Mamatha Latha (Feature Engineering, Modeling & Batch Inference):** Unified 13-feature pairwise feature pipeline (`features_13.py`), TF-IDF corpus vectorizer training (`tfidf_artifacts.py`), XGBoost model training and 5-fold `GroupKFold` cross-validation, and the production streaming batch inference engine (`batch_inference.py`).
- **Talla Likither (Pipeline Integration & Submission Packaging):** Full pipeline orchestration, test set coverage verification, singleton formatting, threshold tuning on candidate scores, and final submission archive packaging.

---

## Repository Layout

```
.
├── Documentation_template.md             # Challenge methodology write-up for final submission
├── README.md                             # Project overview and reproduction instructions
├── output/                               # Final competition output directory
│   ├── matching_results.tsv              # Final entity matches (scored on leaderboard)
│   └── candidate_pairs.tsv               # Final blocking candidate pairs fed into model
├── code/
│   └── business_entity_resolution/       # Self-contained, runnable production package
│       ├── README.md                     # Package-level reproduction instructions
│       ├── requirements.txt              # Pinned production environment dependencies
│       ├── models/
│       │   ├── matching_model.pkl        # Serialized production XGBoost model (300 trees, depth 4)
│       │   └── tfidf_vectorizers.pkl     # Pre-fitted Name & Address TfidfVectorizer artifacts
│       └── src/
│           ├── batch_inference.py        # Production streaming/chunked batch inference driver
│           ├── features.py               # 9 baseline lexical, length, overlap & equality features
│           ├── features_extra.py         # Levenshtein edit distance & TF-IDF cosine similarity
│           ├── features_13.py            # Central 13-feature pipeline & feature schema validator
│           ├── inference.py              # Model loader, schema validator & batch scoring API
│           ├── tfidf_artifacts.py        # TF-IDF serialization & loading utilities
│           ├── train_model.py            # Baseline model training and evaluation script
│           ├── smoke_test_inference.py   # Model schema and inference sanity test suite
│           └── smoke_test_batch_inference.py # End-to-end streaming batch integration test
└── utils/
    └── validate_submission.py            # Challenge submission format and schema validator
```

---

## Methodology & Architecture

### System Flowchart

```
+---------------------------------------------------------------------------------------------------+
|                                 END-TO-END RESOLUTION PIPELINE                                    |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [Source 1: Reference]      [Source 2: Candidates]      [Source 3: Candidates]                    |
|             |                          |                          |                               |
|             +--------------------------+--------------------------+                               |
|                                        |                                                          |
|                                        v                                                          |
|                  STAGE 1: CANDIDATE GENERATION (Member 1)                                         |
|                  * Unicode NFKC Normalization & Lowercasing                                       |
|                  * Corporate Stop-Token Filtering (pvt, ltd, corp, inc, llc...)                   |
|                  * Exact Normalized Name Inverted Index                                           |
|                  * Distinctive Token Inverted Index (Frequency <= 5000, Top-3 Rarest)             |
|                                        |                                                          |
|                                        v                                                          |
|                        [Candidate Pairs (TSV Stream)]                                             |
|                                        |                                                          |
|                                        v                                                          |
|                  STAGE 2: FEATURE EXTRACTION & INFERENCE (Member 2)                               |
|                  * 13 Pairwise Features:                                                          |
|                    - Name/Address Exact Match Indicators                                          |
|                    - SequenceMatcher Ratios (Character Blocks)                                    |
|                    - Token Jaccard Overlap Sets                                                   |
|                    - Length Ratios                                                                |
|                    - Normalized Levenshtein Edit Distance                                         |
|                    - Pre-fitted Word TF-IDF Cosine Similarities                                   |
|                    - Open-Set Country Equality Check (country_same)                               |
|                  * XGBoost Pairwise Classifier (300 estimators, max_depth=4)                      |
|                  * Constant-Memory Streaming Chunk Driver (batch_inference.py)                    |
|                                        |                                                          |
|                                        v                                                          |
|                   [Candidate Matches with Match Probabilities]                                    |
|                                        |                                                          |
|                                        v                                                          |
|                  STAGE 3: POST-PROCESSING & INTEGRATION (Member 3)                                |
|                  * Threshold Application (Primary Validation Candidate: tau = 0.60)               |
|                  * Grouping & Comma-Separated Match Aggregation                                   |
|                  * Full Test Set Coverage & Singleton Formatting                                  |
|                  * Output Verification via validate_submission.py                                 |
|                                        |                                                          |
|                                        v                                                          |
|                       [Final: matching_results.tsv]                                               |
+---------------------------------------------------------------------------------------------------+
```

### Stage 1: Candidate Generation (Blocking)
- **Problem:** Exhaustive pairwise comparison across sources requires $> 2.3 \times 10^{13}$ operations.
- **Solution:** A two-fold inverted index:
  1. *Exact Normalized Name Match:* Quickly groups records whose cleaned names match identically.
  2. *Token Inverted Indexing:* Maps distinctive words to entity IDs. Common words (exceeding `max_frequency = 5000`) and corporate suffixes (`GENERIC_NAME_TOKENS`) are filtered out. Up to 3 rarest distinctive tokens are queried per entity.
- **Output:** Generated 161,521,156 candidate pairs across 9,966 test Source 1 entities (`candidate_pairs_internal.tsv`).

### Stage 2: 13-Feature Engineering
All 13 features are computed deterministically in `features_13.py` and validated by `validate_feature_matrix()`:

| # | Feature Name | Description |
|---|---|---|
| 1 | `name_exact` | Exact normalized string match indicator (1 or 0) |
| 2 | `name_similarity` | Character-level `difflib.SequenceMatcher` ratio ($2M / (T_1 + T_2)$) |
| 3 | `address_exact` | Exact normalized address match indicator (1 or 0) |
| 4 | `address_similarity` | Character-level `difflib.SequenceMatcher` ratio for addresses |
| 5 | `country_same` | Relational binary equality (open-set safe for `US`, `India`, `France`) |
| 6 | `name_token_overlap` | Token-level Jaccard index: $\|A \cap B\| / \|A \cup B\|$ |
| 7 | `address_token_overlap` | Token-level Jaccard index on whitespace-delimited address tokens |
| 8 | `name_length_similarity` | Relative string length ratio: $1 - \frac{\|len(A) - len(B)\|}{\max(len(A), len(B))}$ |
| 9 | `address_length_similarity` | Relative address length ratio |
| 10 | `name_levenshtein_similarity` | Normalized edit distance: $1 - \frac{\text{Levenshtein}(A, B)}{\max(len(A), len(B))}$ |
| 11 | `address_levenshtein_similarity` | Normalized Levenshtein similarity on normalized addresses |
| 12 | `name_tfidf_cosine` | Cosine similarity of L2-normalized word-level TF-IDF vectors |
| 13 | `address_tfidf_cosine` | Cosine similarity of L2-normalized word-level TF-IDF vectors |

*Note:* Missing values (`NaN`) follow the established normalization convention (converted to `"nan"` prior to processing) ensuring exact feature alignment between development and production.

### Stage 3: XGBoost Pairwise Classifier
- **Model:** `XGBClassifier` (`xgboost==3.2.0`).
- **Configuration:** `n_estimators=300`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`, `random_state=42`.
- **Strengths:** Captures non-linear feature interactions (e.g., compensating moderate name similarity with high address TF-IDF cosine) while remaining highly robust against overfitting on imbalanced data.

### Stage 4: Streaming/Chunked Batch Inference
- Implemented in `batch_inference.py`.
- Source records are loaded once into hash maps for $O(1)$ lookup per candidate.
- Candidate pairs are streamed in chunks (default 10,000), computing features and probabilities on the fly.
- Memory consumption remains constant (< 2 GB RAM) regardless of candidate file size.

---

## Offline Validation & Evaluation Results

Validation was conducted on a realistic candidate evaluation dataset of **2,408,157 candidate pairs** generated from a sample of 1,000 Source 1 entities (`random_state=42`, positive rate $\approx 0.0991\%$), split using **5-Fold `GroupKFold`** grouped on `source1_entity_id`:

### 5-Fold Cross-Validation Performance (XGBoost)

| Candidate Threshold ($\tau$) | Macro Entity-Level $F_{0.5}$ | Macro Entity Precision | Macro Entity Recall | Pair-Level $F_{0.5}$ |
|---|---|---|---|---|
| $\tau = 0.50$ | $0.7200 \pm 0.0228$ | $0.7338 \pm 0.0203$ | $0.7189 \pm 0.0202$ | $0.8561 \pm 0.0267$ |
| $\tau = 0.55$ | $0.7253 \pm 0.0240$ | $0.7419 \pm 0.0210$ | $0.7124 \pm 0.0221$ | $0.8657 \pm 0.0246$ |
| **$\tau = 0.60$ \*** | **$0.7283 \pm 0.0232$** | **$0.7495 \pm 0.0208$** | **$0.7032 \pm 0.0234$** | **$0.8708 \pm 0.0258$** |

*\* Primary validation-supported threshold candidate (not a guaranteed final leaderboard threshold).*

### Model Comparison on Realistic Distribution

| Metric | Logistic Regression (13 feat, $\tau=0.50$) | XGBoost (13 feat, $\tau=0.60$) | Delta / Gain |
|---|---|---|---|
| **PR-AUC (Average Precision)** | 0.8673 | **0.9218** | **+5.45% AP gain** |
| **Macro Entity-Level $F_{0.5}$** | 0.6655 | **0.7283** | **+6.28% $F_{0.5}$ gain** |
| **Macro Entity Precision** | 0.6932 | **0.7495** | **+5.63% Precision gain** |
| **Pair-Level $F_{0.5}$** | 0.8366 | **0.8708** | **+3.42% Pair $F_{0.5}$ gain** |

> **Important Metric Clarifications:**
> 1. **Macro $F_{0.5}$, Not Accuracy:** 0.7283 is the offline Macro Entity-Level $F_{0.5}$ validation score, weighting precision 2× over recall ($\beta = 0.5$) and accounting for singletons. Describing this value as accuracy is incorrect.
> 2. **Offline Validation vs. Leaderboard:** 0.7283 reflects offline 5-fold cross-validation on real training candidate pairs. It is not the competition leaderboard score, which is evaluated on hidden test data.
> 3. **Collaborative Pipeline:** Performance reflects the combined execution of Member 1's blocking, Member 2's modeling, and Member 3's integration.

---

## Reproducibility & Execution Guide

### 1. Environment Setup

Ensure Python 3.10+ is available. Install the pinned dependencies:

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

Pinned versions:
- `numpy==2.2.6`
- `pandas==2.3.3`
- `scikit-learn==1.7.2`
- `nltk==3.9.3`
- `joblib==1.5.3`
- `xgboost==3.2.0`

### 2. Running Candidate Generation
Run Member 1's blocking pipeline over test records to generate candidate pairs:
- Output: `output/candidate_pairs.tsv`
- Internal: `dataset/test/candidate_pairs_internal.tsv`

### 3. Running Streaming Batch Inference
Execute streaming batch scoring with Member 2's inference driver:

```bash
python code/business_entity_resolution/src/batch_inference.py \
    --pairs dataset/test/candidate_pairs_internal.tsv \
    --source1 dataset/test/test_source1.tsv \
    --source2 dataset/test/test_source2.tsv \
    --source3 dataset/test/test_source3.tsv \
    --output output/matching_results_scored.tsv \
    --threshold 0.60 \
    --chunk-size 10000
```

### 4. Formatting & Validating Submissions
Member 3 aggregates scored matches by `source1_entity_id`, outputs comma-separated ID lists in `output/matching_results.tsv` (with empty lists for singletons), and verifies file compliance:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## Smoke Tests & Artifact Verification

The code package includes built-in verification tests to guarantee schema stability and pipeline integrity:

1. **Inference Schema Smoke Test:**
   ```bash
   python code/business_entity_resolution/src/smoke_test_inference.py
   ```
   *Verifies feature names, order, and output probabilities.*

2. **Streaming Batch Inference Smoke Test:**
   ```bash
   python code/business_entity_resolution/src/smoke_test_batch_inference.py
   ```
   *Verifies chunked execution, ID alignment, probability bounds [0, 1], and thresholding logic.*

---

## License & Fair Play Compliance

- **Permissible Licenses:** All model architectures and dependencies are licensed under Apache 2.0 or MIT.
- **Fair Play Guarantee:** The pipeline operates completely offline. No commercial entity resolution APIs, government registration lookups, geocoding web services, or internet-based data augmentations were used.
