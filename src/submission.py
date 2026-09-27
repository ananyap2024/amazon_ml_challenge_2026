"""
ML Challenge 2026 — Submission Generator (Member 3: Pipeline, Integration & Submission)

Generates the official submission files for the Amazon ML Challenge 2026:
1. `output/matching_results.tsv`: Final predicted entity matches (leaderboard submission).
2. `output/candidate_pairs.tsv`: Final candidate set evaluated by the matching model.

Key Architectural & Formatting Guarantees:
- Tab-separated values (.tsv), UTF-8 encoded, without an index column.
- Strict 1-to-1 row alignment with test Source 1 entities (preserves file ordering).
- Empty matches and empty candidate sets are represented as empty fields (never None/NaN/[]).
- Deduplicates entity IDs inside lists while preserving stable ordering.
- Strict subset constraint: every matched entity ID is verified to be present in candidates.
- Memory-efficient streaming of IDs; avoids Cartesian products and avoids loading raw text fields.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import pandas as pd

# Constants matching official requirements
DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]
VALID_TARGET_PREFIXES = ("S2-", "S3-")


def load_source1_ids(
    source1_input: Union[str, Path, Iterable[str], pd.Series, pd.DataFrame]
) -> List[str]:
    """
    Load and validate Source 1 entity IDs from a file path or iterable.

    When given a file path (e.g., dataset/test/test_source1.tsv), reads ONLY the
    first column line-by-line. This avoids loading the heavy ~175MB TSV with full
    addresses and names into memory.

    Parameters
    ----------
    source1_input : Union[str, Path, Iterable[str], pd.Series, pd.DataFrame]
        Path to test_source1.tsv or a sequence/Series of entity IDs.

    Returns
    -------
    List[str]
        Ordered list of unique Source 1 entity IDs.

    Raises
    ------
    FileNotFoundError
        If the file path does not exist.
    ValueError
        If duplicate or invalid Source 1 IDs are encountered.
    """
    ids: List[str] = []
    seen: Set[str] = set()

    if isinstance(source1_input, (str, Path)):
        path = Path(source1_input)
        if not path.is_file():
            raise FileNotFoundError(f"Source 1 file not found: {path}")

        with open(path, mode="r", encoding="utf-8") as f:
            header = f.readline()
            if not header:
                raise ValueError(f"Source 1 file is empty: {path}")

            for line_num, line in enumerate(f, start=2):
                stripped = line.strip()
                if not stripped:
                    continue
                s1_id = line.split(DELIM, 1)[0].strip()
                if not s1_id:
                    continue
                if not s1_id.startswith("S1-"):
                    raise ValueError(
                        f"Invalid Source 1 entity ID '{s1_id}' at line {line_num} in {path}. "
                        f"Must start with 'S1-'."
                    )
                if "," in s1_id:
                    raise ValueError(
                        f"Source 1 entity ID '{s1_id}' at line {line_num} in {path} contains illegal comma character."
                    )
                if s1_id in seen:
                    raise ValueError(
                        f"Duplicate Source 1 entity ID '{s1_id}' found at line {line_num} in {path}."
                    )
                seen.add(s1_id)
                ids.append(s1_id)

    elif isinstance(source1_input, pd.DataFrame):
        col = (
            "source1_entity_id"
            if "source1_entity_id" in source1_input.columns
            else "entity_id"
        )
        if col not in source1_input.columns:
            raise ValueError(
                f"DataFrame must contain 'source1_entity_id' or 'entity_id' column. Found: {list(source1_input.columns)}"
            )
        raw_ids = source1_input[col].astype(str).tolist()
        return load_source1_ids(raw_ids)

    elif isinstance(source1_input, pd.Series):
        return load_source1_ids(source1_input.astype(str).tolist())

    elif isinstance(source1_input, Iterable):
        for idx, item in enumerate(source1_input):
            s1_id = str(item).strip()
            if not s1_id.startswith("S1-"):
                raise ValueError(
                    f"Invalid Source 1 entity ID '{s1_id}' at index {idx}. Must start with 'S1-'."
                )
            if "," in s1_id:
                raise ValueError(
                    f"Source 1 entity ID '{s1_id}' at index {idx} contains illegal comma character."
                )
            if s1_id in seen:
                raise ValueError(f"Duplicate Source 1 entity ID '{s1_id}' in input sequence.")
            seen.add(s1_id)
            ids.append(s1_id)
    else:
        raise TypeError(f"Unsupported type for source1_input: {type(source1_input)}")

    return ids


def validate_candidate_pairs_dataframe(df: pd.DataFrame) -> None:
    """
    Validate the internal candidate pairs DataFrame.

    Required columns:
    - `source1_entity_id`: must be string, must start with 'S1-'.
    - `candidate_entity_id`: must be string, must start with 'S2-' or 'S3-'.
    - `source`: optional metadata column (e.g. 'source2' or 'source3'), ignored in output.

    Raises
    ------
    ValueError
        If required columns are missing or if invalid ID formats / self-matches exist.
    """
    required_cols = {"source1_entity_id", "candidate_entity_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"candidate_pairs DataFrame missing required columns: {missing}")

    if df.empty:
        return

    # Check for S1 self-matches
    sample_targets = df["candidate_entity_id"].astype(str)
    s1_self = sample_targets[sample_targets.str.startswith("S1-")]
    if not s1_self.empty:
        example = s1_self.iloc[0]
        raise ValueError(
            f"candidate_pairs contains forbidden Source 1 self-matches (e.g., '{example}'). "
            f"Only 'S2-' or 'S3-' entities allowed."
        )

    # Check for invalid prefixes
    invalid_prefix = sample_targets[~sample_targets.str.startswith(VALID_TARGET_PREFIXES)]
    if not invalid_prefix.empty:
        example = invalid_prefix.iloc[0]
        raise ValueError(
            f"candidate_pairs contains IDs without valid S2-/S3- prefix (e.g., '{example}')."
        )

    # Check for corrupting delimiter characters in IDs (tabs, newlines, carriage returns, commas)
    s1_series = df["source1_entity_id"].astype(str)
    bad_delims = re.compile(r"[\t\n\r,]")
    s1_corrupt = s1_series[s1_series.str.contains(bad_delims, regex=True)]
    if not s1_corrupt.empty:
        example = repr(s1_corrupt.iloc[0])
        raise ValueError(
            f"candidate_pairs contains source1_entity_id with illegal delimiter characters (tab/newline/comma): {example}"
        )
    cand_corrupt = sample_targets[sample_targets.str.contains(bad_delims, regex=True)]
    if not cand_corrupt.empty:
        example = repr(cand_corrupt.iloc[0])
        raise ValueError(
            f"candidate_pairs contains candidate_entity_id with illegal delimiter characters (tab/newline/comma): {example}"
        )


def validate_predictions_dataframe(df: pd.DataFrame) -> None:
    """
    Validate model predictions DataFrame.

    Required columns:
    - `source1_entity_id`: string
    - `candidate_entity_id`: string
    - `match_probability`: float / numeric

    Raises
    ------
    ValueError
        If required columns are missing or data types are invalid.
    """
    required_cols = {"source1_entity_id", "candidate_entity_id", "match_probability"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"predictions DataFrame missing required columns: {missing}")

    if df.empty:
        return

    if not pd.api.types.is_numeric_dtype(df["match_probability"]):
        raise ValueError(
            f"'match_probability' column must be numeric. Found dtype: {df['match_probability'].dtype}"
        )
    probs = df["match_probability"]
    if probs.isna().any():
        raise ValueError("predictions DataFrame contains NaN in 'match_probability'.")
    if not probs.between(0.0, 1.0).all():
        raise ValueError(
            "predictions DataFrame contains values outside [0.0, 1.0] in 'match_probability'."
        )


def aggregate_candidates_map(
    candidate_pairs: pd.DataFrame,
) -> Tuple[Dict[str, List[str]], Dict[str, Set[str]]]:
    """
    Group candidate pairs by Source 1 entity ID, preserving appearance order while deduplicating.

    Returns
    -------
    Tuple[Dict[str, List[str]], Dict[str, Set[str]]]
        - candidates_ordered: mapping of s1_id -> list of candidate IDs in appearance order.
        - candidates_set: mapping of s1_id -> set of candidate IDs (for fast membership test).
    """
    validate_candidate_pairs_dataframe(candidate_pairs)

    candidates_ordered: Dict[str, List[str]] = {}
    candidates_set: Dict[str, Set[str]] = {}

    if candidate_pairs.empty:
        return candidates_ordered, candidates_set

    # Convert to string series for clean lookup
    s1_series = candidate_pairs["source1_entity_id"].astype(str)
    cand_series = candidate_pairs["candidate_entity_id"].astype(str)

    for s1_id, cand_id in zip(s1_series, cand_series):
        if s1_id not in candidates_set:
            candidates_ordered[s1_id] = [cand_id]
            candidates_set[s1_id] = {cand_id}
        else:
            if cand_id not in candidates_set[s1_id]:
                candidates_ordered[s1_id].append(cand_id)
                candidates_set[s1_id].add(cand_id)

    return candidates_ordered, candidates_set


def aggregate_matches_map(
    predictions: pd.DataFrame,
    candidates_set: Dict[str, Set[str]],
    threshold: float,
) -> Dict[str, List[str]]:
    """
    Filter predictions by probability threshold and group by Source 1 entity ID.
    Enforces that matched entity IDs must be present in the candidate set for that S1 entity.

    Parameters
    ----------
    predictions : pd.DataFrame
        Predictions with columns [source1_entity_id, candidate_entity_id, match_probability].
    candidates_set : Dict[str, Set[str]]
        Valid candidates per S1 entity ID.
    threshold : float
        Probability threshold (match_probability >= threshold).

    Returns
    -------
    Dict[str, List[str]]
        Mapping of s1_id -> list of matched entity IDs.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be between 0.0 and 1.0. Given: {threshold}")

    validate_predictions_dataframe(predictions)

    matches_map: Dict[str, List[str]] = {}
    matches_seen: Dict[str, Set[str]] = {}

    if predictions.empty:
        return matches_map

    # Filter by threshold first (vectorized in pandas)
    qualified = predictions[predictions["match_probability"] >= threshold]
    if qualified.empty:
        return matches_map

    s1_series = qualified["source1_entity_id"].astype(str)
    cand_series = qualified["candidate_entity_id"].astype(str)

    for s1_id, cand_id in zip(s1_series, cand_series):
        # Subset constraint: candidate must exist in candidates_set for this S1 entity
        valid_candidates = candidates_set.get(s1_id)
        if not valid_candidates or cand_id not in valid_candidates:
            # Skip predictions for entities that were never candidate pairs
            continue

        if s1_id not in matches_seen:
            matches_map[s1_id] = [cand_id]
            matches_seen[s1_id] = {cand_id}
        else:
            if cand_id not in matches_seen[s1_id]:
                matches_map[s1_id].append(cand_id)
                matches_seen[s1_id].add(cand_id)

    return matches_map


def build_submission_dataframes(
    all_source1_ids: Sequence[str],
    candidate_pairs: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the final DataFrames for matching_results and candidate_pairs.

    Guarantees:
    - Every ID in all_source1_ids appears exactly once.
    - Preserves all_source1_ids ordering.
    - Empty match or empty candidate representations are empty strings ("").
    - ID lists are comma-separated and deduplicated.
    - All matched IDs are guaranteed to be a subset of candidate IDs.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (matching_results_df, candidate_pairs_df)
    """
    cand_ordered, cand_set = aggregate_candidates_map(candidate_pairs)
    matches_map = aggregate_matches_map(predictions, cand_set, threshold=threshold)

    matching_rows = []
    candidate_rows = []

    for s1_id in all_source1_ids:
        # Candidates
        cands = cand_ordered.get(s1_id, [])
        cands_str = ",".join(cands) if cands else ""
        candidate_rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": cands_str,
        })

        # Matches
        matches = matches_map.get(s1_id, [])
        matches_str = ",".join(matches) if matches else ""
        matching_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": matches_str,
        })

    matching_df = pd.DataFrame(matching_rows, columns=MATCHING_HEADER)
    candidate_df = pd.DataFrame(candidate_rows, columns=CANDIDATE_HEADER)

    return matching_df, candidate_df


class DiskBackedSubmissionAccumulator:
    """
    Disk-backed accumulator using standard library sqlite3 for incremental processing.
    Ensures memory usage is strictly bounded to the size of a single chunk,
    even when processing hundreds of millions of candidate pairs.

    Guarantees:
    - Peak memory is bounded to a single chunk (~a few MB).
    - Preserves appearance order of candidate IDs per Source 1 entity.
    - Strictly enforces matched IDs as a subset of candidate IDs.
    - Preserves all authoritative S1 IDs (singletons get empty strings).
    - Temporary database is cleanly closed and removed upon completion.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        temp_dir: Optional[Union[str, Path]] = None,
    ):
        self._temp_dir = None
        if db_path is None:
            self._temp_dir = tempfile.TemporaryDirectory(
                dir=str(temp_dir) if temp_dir else None
            )
            self.db_path = Path(self._temp_dir.name) / "accumulator.db"
        else:
            self.db_path = Path(db_path)
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA journal_mode = OFF")
        # Store insertion order (ord) to guarantee appearance-order preservation
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS candidates ("
            "s1_id TEXT, cand_id TEXT, ord INTEGER, "
            "PRIMARY KEY (s1_id, cand_id)"
            ") WITHOUT ROWID"
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS matches ("
            "s1_id TEXT, cand_id TEXT, ord INTEGER, "
            "PRIMARY KEY (s1_id, cand_id)"
            ") WITHOUT ROWID"
        )
        self._ord = 0

    def add_chunk(
        self,
        candidate_chunk: pd.DataFrame,
        probs: Sequence[float],
        threshold: float = 0.5,
    ) -> None:
        """
        Incorporate a scored candidate batch into the accumulator.
        """
        validate_candidate_pairs_dataframe(candidate_chunk)
        if len(candidate_chunk) != len(probs):
            raise ValueError(
                f"Mismatch: {len(candidate_chunk)} candidate rows vs {len(probs)} probabilities."
            )

        cands_to_insert = []
        matches_to_insert = []

        s1_series = candidate_chunk["source1_entity_id"].astype(str)
        cand_series = candidate_chunk["candidate_entity_id"].astype(str)

        for s1, cand, p in zip(s1_series, cand_series, probs):
            p_float = float(p)
            if math.isnan(p_float) or math.isinf(p_float) or not (0.0 <= p_float <= 1.0):
                raise ValueError(
                    f"Invalid match probability {p_float} for candidate pair ({s1}, {cand}). "
                    f"Must be a finite float in [0.0, 1.0]."
                )
            self._ord += 1
            cands_to_insert.append((s1, cand, self._ord))
            if p_float >= threshold:
                matches_to_insert.append((s1, cand, self._ord))

        if cands_to_insert:
            self.conn.executemany(
                "INSERT OR IGNORE INTO candidates VALUES (?, ?, ?)", cands_to_insert
            )
        if matches_to_insert:
            self.conn.executemany(
                "INSERT OR IGNORE INTO matches VALUES (?, ?, ?)", matches_to_insert
            )
        self.conn.commit()

    def write_outputs(
        self,
        all_source1_ids: Sequence[str],
        matching_path: Union[str, Path],
        candidate_path: Union[str, Path],
    ) -> Tuple[Path, Path]:
        """
        Stream out matching_results.tsv and candidate_pairs.tsv in the exact order of all_source1_ids.
        Zero-candidate entities write empty fields rather than being dropped.
        """
        m_path = Path(matching_path).resolve()
        c_path = Path(candidate_path).resolve()
        m_path.parent.mkdir(parents=True, exist_ok=True)
        c_path.parent.mkdir(parents=True, exist_ok=True)

        cur = self.conn.cursor()

        with open(c_path, "w", encoding="utf-8") as f_cand, open(m_path, "w", encoding="utf-8") as f_match:
            f_cand.write(f"{CANDIDATE_HEADER[0]}{DELIM}{CANDIDATE_HEADER[1]}\n")
            f_match.write(f"{MATCHING_HEADER[0]}{DELIM}{MATCHING_HEADER[1]}\n")

            for s1 in all_source1_ids:
                s1_clean = str(s1).strip()

                # Candidates (stream cursor to keep per-entity memory bounded)
                cur.execute(
                    "SELECT cand_id FROM candidates WHERE s1_id = ? ORDER BY ord",
                    (s1_clean,),
                )
                cands_str = ",".join(r[0] for r in cur)
                f_cand.write(f"{s1_clean}{DELIM}{cands_str}\n")

                # Matches (strictly subset of candidates because they originate from scored candidates)
                cur.execute(
                    "SELECT cand_id FROM matches WHERE s1_id = ? ORDER BY ord",
                    (s1_clean,),
                )
                matches_str = ",".join(r[0] for r in cur)
                f_match.write(f"{s1_clean}{DELIM}{matches_str}\n")

        return m_path, c_path

    def close(self) -> None:
        """Close SQLite connection and clean up temporary directory."""
        try:
            self.conn.close()
        except Exception:
            pass
        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass


def write_submission_tsv(df: pd.DataFrame, output_path: Union[str, Path]) -> Path:
    """
    Save a DataFrame as a tab-separated UTF-8 TSV without index, ensuring empty fields are blank.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export.
    output_path : Union[str, Path]
        Target file path.

    Returns
    -------
    Path
        Absolute Path of the written file.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # na_rep="" ensures any missing / NA value is written as an empty field
    df.to_csv(out_file, sep=DELIM, index=False, encoding="utf-8", na_rep="")
    return out_file


def generate_submission_files(
    all_source1_ids: Union[str, Path, Sequence[str]],
    candidate_pairs: pd.DataFrame,
    predictions: pd.DataFrame,
    threshold: float = 0.5,
    output_dir: Union[str, Path] = "output",
    validate_after: bool = True,
    test_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """
    Full pipeline entrypoint: loads S1 IDs, aggregates candidates and predictions,
    writes both TSV submission files, and optionally runs the official validator.

    Parameters
    ----------
    all_source1_ids : Union[str, Path, Sequence[str]]
        Source 1 entity IDs or path to test_source1.tsv.
    candidate_pairs : pd.DataFrame
        Candidate pairs [source1_entity_id, candidate_entity_id, (source)].
    predictions : pd.DataFrame
        Model predictions [source1_entity_id, candidate_entity_id, match_probability].
    threshold : float
        Decision threshold for matching (default 0.5).
    output_dir : Union[str, Path]
        Output directory (default 'output').
    validate_after : bool
        Whether to execute utils/validate_submission.py on the outputs.
    test_dir : Optional[Union[str, Path]]
        Path to test directory for validator (default None -> inferred if all_source1_ids is path).

    Returns
    -------
    Tuple[Path, Path]
        (matching_path, candidate_path)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matching_path = out_dir / "matching_results.tsv"
    candidate_path = out_dir / "candidate_pairs.tsv"

    print(f"[Submission] Loading Source 1 IDs...")
    s1_ids = load_source1_ids(all_source1_ids)
    print(f"[Submission] Total Source 1 test entities: {len(s1_ids):,}")

    print(f"[Submission] Building submission tables with threshold={threshold}...")
    matching_df, candidate_df = build_submission_dataframes(
        all_source1_ids=s1_ids,
        candidate_pairs=candidate_pairs,
        predictions=predictions,
        threshold=threshold,
    )

    print(f"[Submission] Writing {matching_path}...")
    write_submission_tsv(matching_df, matching_path)

    print(f"[Submission] Writing {candidate_path}...")
    write_submission_tsv(candidate_df, candidate_path)

    # Report quick stats
    non_empty_cands = (candidate_df["candidate_entity_ids"] != "").sum()
    non_empty_matches = (matching_df["matched_entity_ids"] != "").sum()
    print(
        f"[Submission] Candidate coverage: {non_empty_cands:,} / {len(s1_ids):,} entities with >=1 candidate."
    )
    print(
        f"[Submission] Match coverage: {non_empty_matches:,} / {len(s1_ids):,} entities with >=1 match."
    )

    if validate_after:
        # Determine test_dir
        if test_dir is None:
            if isinstance(all_source1_ids, (str, Path)) and Path(all_source1_ids).is_file():
                test_dir = Path(all_source1_ids).parent
            else:
                test_dir = Path("dataset/test")

        run_validator(
            matching_path=matching_path,
            candidate_path=candidate_path,
            test_dir=test_dir,
        )

    return matching_path, candidate_path


def run_validator(
    matching_path: Union[str, Path],
    candidate_path: Optional[Union[str, Path]] = None,
    test_dir: Union[str, Path] = "dataset/test",
    check_ids: bool = False,
) -> int:
    """
    Run utils/validate_submission.py against the generated outputs.

    Returns
    -------
    int
        Exit code (0 = PASS, 1 = FAIL).
    """
    validator_script = Path(__file__).resolve().parent.parent / "utils" / "validate_submission.py"
    if not validator_script.is_file():
        print(f"[Validator Warning] Validator script not found at {validator_script}. Skipping.")
        return -1

    cmd = [
        sys.executable,
        str(validator_script),
        "--matching",
        str(matching_path),
        "--test-dir",
        str(test_dir),
    ]
    if candidate_path:
        cmd.extend(["--candidate", str(candidate_path)])
    if check_ids:
        cmd.append("--check-ids")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"[Submission] Running official validator:\n  {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ML Challenge 2026 submission files."
    )
    parser.add_argument(
        "--test-source1",
        default="dataset/test/test_source1.tsv",
        help="Path to test_source1.tsv",
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to TSV/Parquet containing candidate pairs",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to TSV/Parquet containing model predictions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Destination directory for output TSVs (default: output)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip executing utils/validate_submission.py",
    )
    return parser


def main() -> None:
    parser = _build_cli()
    args = parser.parse_args()

    # Load candidate pairs
    cand_path = Path(args.candidates)
    if cand_path.suffix == ".parquet":
        candidates_df = pd.read_parquet(cand_path)
    else:
        candidates_df = pd.read_csv(cand_path, sep=DELIM)

    # Load predictions
    pred_path = Path(args.predictions)
    if pred_path.suffix == ".parquet":
        predictions_df = pd.read_parquet(pred_path)
    else:
        predictions_df = pd.read_csv(pred_path, sep=DELIM)

    generate_submission_files(
        all_source1_ids=args.test_source1,
        candidate_pairs=candidates_df,
        predictions=predictions_df,
        threshold=args.threshold,
        output_dir=args.output_dir,
        validate_after=not args.skip_validation,
    )


if __name__ == "__main__":
    main()
