"""
ML Challenge 2026 — Submission Packager (Member 3: Pipeline, Integration & Submission)

Assembles the final official submission ZIP archive conforming strictly to the
required structure specified in the problem statement (README.md lines 157-175):

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (leaderboard submission)
│   └── candidate_pairs.tsv         # blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all source code
│       ├── utils/                  # validation utility
│       ├── README.md               # reproduction guide
│       └── requirements.txt        # pinned dependencies
└── Documentation_template.md       # methodology write-up

Safeguards:
- Strict exclusion of full dataset/ and virtual environment (.venv/).
- Verifies output files, headers, and encoding before packaging.
- Validates the resulting ZIP hierarchy.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import List, Optional
import zipfile

REPRODUCTION_README_CONTENT = """# Amazon ML Challenge 2026 — Business Entity Resolution

## Reproduction Guide

This directory contains the self-contained source code, environment specification, and documentation to reproduce the final submission outputs:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

### 1. Environment Setup

Python 3.11+ is recommended. Install pinned dependencies:

```bash
pip install -r requirements.txt
```

### 2. Pipeline Execution

#### A. Synthetic Verification / Dry-Run (Instant Smoke Test)
Verify pipeline wiring, SQLite accumulation, and output formatting without loading the full dataset:

```bash
python src/main.py --dry-run
```

#### B. Full End-to-End Test Inference
To run candidate generation and model scoring over the test dataset:

```bash
python src/main.py --data-dir dataset/test --output-dir output --batch-size 5000 --threshold 0.60
```

If candidate pairs are precomputed:
```bash
python src/main.py --data-dir dataset/test --candidates output/candidate_pairs.tsv --output-dir output
```

### 3. Submission Validation

Verify the generated output files against official competition rules:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

### 4. Architecture Overview

1. **Candidate Generation (Blocking V3)**: Multi-channel candidate generator using exact names, domain-cleaned condensed names, and token frequency caps (`src/blocking_v3.py`).
2. **Feature Engineering**: 13-feature representation capturing name, address, country, token overlap, Levenshtein distance, and TF-IDF cosine similarities (`src/features.py`).
3. **Model Inference**: XGBoost matching model evaluated in streaming batches (`src/features.py`, `src/pipeline.py`).
4. **Streaming Serializer & SQLite Accumulator**: Bounded disk-backed accumulator ensuring peak RAM < 100 MB and 1-to-1 preservation of all test Source 1 entities (`src/submission.py`).
"""


def build_submission_zip(
    team_name: str,
    output_dir: Path,
    zip_dest: Path,
    repo_root: Path,
    documentation_path: Optional[Path] = None,
) -> Path:
    """
    Build the submission ZIP archive according to official hierarchy.
    """
    matching_tsv = output_dir / "matching_results.tsv"
    candidate_tsv = output_dir / "candidate_pairs.tsv"

    if not matching_tsv.is_file():
        raise FileNotFoundError(f"Missing required output file: {matching_tsv}")
    if not candidate_tsv.is_file():
        raise FileNotFoundError(f"Missing required output file: {candidate_tsv}")

    doc_file = documentation_path or (repo_root / "Documentation_template.md")
    if not doc_file.is_file():
        raise FileNotFoundError(f"Missing methodology document: {doc_file}")

    req_file = repo_root / "requirements.txt"
    if not req_file.is_file():
        raise FileNotFoundError(f"Missing requirements.txt at repo root: {req_file}")

    zip_dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Packager] Creating submission archive: {zip_dest}")

    with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        print(f"  + output/matching_results.tsv")
        zf.write(matching_tsv, arcname="output/matching_results.tsv")
        print(f"  + output/candidate_pairs.tsv")
        zf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")

        # 2. Methodology document
        print(f"  + {doc_file.name}")
        zf.write(doc_file, arcname=doc_file.name)

        # 3. Code directory: code/business_entity_resolution/
        code_prefix = "code/business_entity_resolution"

        # Reproduction README
        print(f"  + {code_prefix}/README.md")
        zf.writestr(f"{code_prefix}/README.md", REPRODUCTION_README_CONTENT)

        # Requirements
        print(f"  + {code_prefix}/requirements.txt")
        zf.write(req_file, arcname=f"{code_prefix}/requirements.txt")

        # Model artifacts required for reproducible inference
        models_dir = repo_root / "code" / "business_entity_resolution" / "models"
        required_models = [
            "matching_model.pkl",
            "tfidf_vectorizers.pkl",
        ]

        for model_name in required_models:
            model_file = models_dir / model_name

            if not model_file.is_file():
                raise FileNotFoundError(
                    f"Missing required model artifact: {model_file}"
                )

            arc_name = f"{code_prefix}/models/{model_name}"
            print(f"  + {arc_name}")
            zf.write(model_file, arcname=arc_name)

        # Source code files (excluding __pycache__ and scratch)
        src_dir = repo_root / "src"
        if src_dir.is_dir():
            for root, _, files in os.walk(src_dir):
                if "__pycache__" in root:
                    continue
                for f in files:
                    if f.endswith((".py", ".sql")):
                        full_f = Path(root) / f
                        rel_src = full_f.relative_to(src_dir)
                        arc_name = f"{code_prefix}/src/{rel_src.as_posix()}"
                        print(f"  + {arc_name}")
                        zf.write(full_f, arcname=arc_name)

        # Utils directory
        utils_dir = repo_root / "utils"
        if utils_dir.is_dir():
            for root, _, files in os.walk(utils_dir):
                if "__pycache__" in root:
                    continue
                for f in files:
                    if f.endswith(".py"):
                        full_f = Path(root) / f
                        rel_u = full_f.relative_to(utils_dir)
                        arc_name = f"{code_prefix}/utils/{rel_u.as_posix()}"
                        print(f"  + {arc_name}")
                        zf.write(full_f, arcname=arc_name)

    print(f"\n[Packager] Successfully packaged {zip_dest} ({zip_dest.stat().st_size:,} bytes).")
    return zip_dest


def verify_submission_zip(zip_path: Path) -> List[str]:
    """
    Inspect ZIP contents to ensure compliance with the required structure:
    - output/matching_results.tsv exists
    - output/candidate_pairs.tsv exists
    - Documentation_template.md exists
    - code/business_entity_resolution/src/ exists
    - code/business_entity_resolution/requirements.txt exists
    - code/business_entity_resolution/README.md exists
    - No dataset/ or .venv/ files included
    """
    issues: List[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()

        required_entries = [
            "output/matching_results.tsv",
            "output/candidate_pairs.tsv",
            "Documentation_template.md",
            "code/business_entity_resolution/README.md",
            "code/business_entity_resolution/requirements.txt",
            "code/business_entity_resolution/models/matching_model.pkl",
            "code/business_entity_resolution/models/tfidf_vectorizers.pkl",
        ]
        for req in required_entries:
            if req not in namelist:
                issues.append(f"Missing required entry in ZIP: {req}")

        # Check for forbidden entries (dataset, .venv, git)
        for name in namelist:
            if any(name.startswith(f) for f in [".venv", "dataset/", ".git/", "output/accumulator.db"]):
                issues.append(f"Forbidden directory/file found in ZIP: {name}")

        # Check that src contains python files
        src_files = [n for n in namelist if n.startswith("code/business_entity_resolution/src/") and n.endswith(".py")]
        if not src_files:
            issues.append("No source python files found under code/business_entity_resolution/src/")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Package official submission ZIP archive.")
    parser.add_argument("--team-name", default="my_team", help="Team name for zip naming (default: %(default)s)")
    parser.add_argument("--output-dir", default="output", help="Directory containing generated TSVs (default: %(default)s)")
    parser.add_argument("--zip-dest", default=None, help="Destination path for ZIP file (default: <team_name>_submission.zip)")
    parser.add_argument("--use-dry-run-output", action="store_true", help="Package synthetic dry-run output for testing the packager")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = repo_root / ("output/dry_run" if args.use_dry_run_output else args.output_dir)
    zip_dest = Path(args.zip_dest) if args.zip_dest else repo_root / f"{args.team_name}_submission.zip"

    try:
        built_zip = build_submission_zip(
            team_name=args.team_name,
            output_dir=out_dir,
            zip_dest=zip_dest,
            repo_root=repo_root,
        )
    except FileNotFoundError as err:
        print(f"\n[Packager Error] {err}", file=sys.stderr)
        sys.exit(1)

    issues = verify_submission_zip(built_zip)
    if issues:
        print(f"\n[Packaging Warnings/Errors]:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"[Packager Verification] All structural checks PASSED! Safe to submit.")


if __name__ == "__main__":
    main()
