# data/persistence.py
"""Data persistence functions for saving results."""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime
from ..api import OUTPUT_FORMAT

# File constants
RESULTS_DIR = "results"
RESPONSES_DIR = "responses"
RESPONSES_FILE_CSV = "responses.csv"
RESPONSES_FILE_JSON = "responses.json"
RUNS_DIR = "runs"
SSS_FILE_CSV = "sss_scores.csv"
SSS_FILE_JSON = "sss_scores.json"
VECTORS_FILE = "sss_vectors.json"
SIMILARITY_FILE = "similarity_matrix.npy"
DISTANCE_FILE = "distance_matrix.npy"
NAMES_FILE = "model_names.json"
METADATA_FILE = "metadata.json"
SCORED_ROWS_FILE_CSV = "scored_rows.csv"
SCORED_ROWS_FILE_JSON = "scored_rows.json"


def _atomic_write_json(obj, path: str) -> None:
    """Atomically write JSON to path by writing to a temp file and replacing.

    This prevents truncated files (e.g., if a process is interrupted mid-write).
    """
    dirpath = os.path.dirname(path) or "."
    tmp_path = os.path.join(dirpath, f".{os.path.basename(path)}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                # Best effort: not all environments require/allow fsync
                pass
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def ensure_results_dir(prefix: str) -> str:
    """Ensure the results directory exists."""
    os.makedirs(prefix, exist_ok=True)
    return prefix


def ensure_responses_dir(prefix: str) -> str:
    """Ensure the responses directory exists."""
    responses_path = os.path.join(prefix, RESPONSES_DIR)
    os.makedirs(responses_path, exist_ok=True)
    return responses_path


def save_responses(prefix: str, responses_df: pd.DataFrame):
    """Save collected responses for this run and maintain a global aggregated JSON.

    - Writes to a timestamped run directory under `<prefix>/responses/run_YYYYMMDD_HHMMSS/`.
    - Respects OUTPUT_FORMAT["responses"] for primary format, but always also writes a JSON copy.
    - Adds a `run_id` column to saved rows, and appends valid rows to `results/final_responses.json`,
      deduplicating by (model, prompt_id) keeping the latest run_id.
    """
    responses_path = ensure_responses_dir(prefix)
    # Generate a timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_dir = os.path.join(responses_path, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Ensure run_id column exists for saved/aggregated files
    df_save = responses_df.copy()
    if "run_id" not in df_save.columns:
        df_save["run_id"] = run_id

    # Primary save based on OUTPUT_FORMAT
    if OUTPUT_FORMAT.get("responses") == "json":
        primary_path = os.path.join(run_dir, RESPONSES_FILE_JSON)
        # Write atomically to avoid partial files
        try:
            _atomic_write_json(df_save.replace({np.nan: None}).to_dict(orient="records"), primary_path)
        except Exception:
            # Fallback to pandas writer if anything unexpected happens
            df_save.to_json(primary_path, orient="records", indent=2)
    else:
        primary_path = os.path.join(run_dir, RESPONSES_FILE_CSV)
        df_save.to_csv(primary_path, index=False)

    # Always write a JSON copy for convenience
    json_copy_path = os.path.join(run_dir, RESPONSES_FILE_JSON)
    try:
        # Use atomic write for the JSON copy as well
        _atomic_write_json(df_save.replace({np.nan: None}).to_dict(orient="records"), json_copy_path)
    except Exception:
        # Best-effort: convert to records via pandas to handle non-serializable types
        try:
            import numpy as _np
            df_json = df_save.replace({_np.nan: None})
            with open(json_copy_path, "w", encoding="utf-8") as f:
                json.dump(df_json.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # Append to global aggregated final_responses.json, filtering out error rows
    try:
        agg_dir = os.path.join("results")
        os.makedirs(agg_dir, exist_ok=True)
        final_path = os.path.join(agg_dir, "final_responses.json")

        def _filter_valid(dd: pd.DataFrame) -> pd.DataFrame:
            # Drop obvious error rows: empty response, stop_reason == 'error', non-200 http_status
            if dd.empty:
                return dd
            out = dd.copy()
            # Empty responses
            out["__resp_empty__"] = out.get("response").isna() | (out.get("response").astype(str).str.strip() == "") if "response" in out.columns else True
            # stop_reason
            is_error = (out.get("stop_reason").astype(str).str.lower() == "error") if "stop_reason" in out.columns else False
            # http_status
            if "http_status" in out.columns:
                try:
                    hs = pd.to_numeric(out["http_status"], errors="coerce")
                    bad_status = hs.notna() & (hs.astype(float) != 200.0)
                except Exception:
                    bad_status = False
            else:
                bad_status = False
            keep = (~out["__resp_empty__"]) & (~is_error) & (~bad_status)
            return out[keep].drop(columns=["__resp_empty__"], errors="ignore")

        new_rows = _filter_valid(df_save)
        # Load existing
        if os.path.exists(final_path):
            try:
                with open(final_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                df_existing = pd.DataFrame(existing)
            except Exception:
                df_existing = pd.DataFrame([])
        else:
            df_existing = pd.DataFrame([])

        # Merge and dedup keeping latest by run_id timestamp
        merged = pd.concat([df_existing, new_rows], ignore_index=True)
        # Create run_datetime for ordering if available
        if "run_id" in merged.columns:
            ts = merged["run_id"].astype(str).str.replace("run_", "", regex=False)
            merged["run_datetime"] = pd.to_datetime(ts, errors="coerce", format="%Y%m%d_%H%M%S")
        else:
            merged["run_datetime"] = pd.NaT
        # Sort by model, prompt_id, then run_datetime; keep last
        for col in ("model", "prompt_id"):
            if col not in merged.columns:
                merged[col] = None
        merged = merged.sort_values(["model", "prompt_id", "run_datetime"], ascending=[True, True, True], kind="mergesort")
        merged = merged.drop_duplicates(subset=["model", "prompt_id"], keep="last")
        # Write back
        # Atomically write the aggregated file to prevent truncation
        _atomic_write_json(merged.to_dict(orient="records"), final_path)
    except Exception:
        # Aggregation is best-effort and should not fail the run save
        pass


def save_sss(prefix: str, sss_df: pd.DataFrame):
    """Save SSS scores to a file (CSV or JSON based on config)."""
    results_path = ensure_results_dir(prefix)
    
    if OUTPUT_FORMAT["sss"] == "json":
        file_path = os.path.join(results_path, SSS_FILE_JSON)
        sss_df.to_json(file_path, orient="records", indent=2)
    else:
        file_path = os.path.join(results_path, SSS_FILE_CSV)
        sss_df.to_csv(file_path, index=False)


def save_vectors(prefix: str, per_vec: Dict[str, List[float]]):
    """Save SSS vectors to a JSON file."""
    results_path = ensure_results_dir(prefix)
    file_path = os.path.join(results_path, VECTORS_FILE)
    with open(file_path, 'w') as f:
        json.dump(per_vec, f)


def save_matrices(prefix: str, names: List[str], S: np.ndarray, D: np.ndarray):
    """Save similarity and distance matrices."""
    results_path = ensure_results_dir(prefix)
    # Save names
    names_path = os.path.join(results_path, NAMES_FILE)
    with open(names_path, 'w') as f:
        json.dump(names, f)
    
    # Save matrices
    S_path = os.path.join(results_path, SIMILARITY_FILE)
    D_path = os.path.join(results_path, DISTANCE_FILE)
    np.save(S_path, S)
    np.save(D_path, D)


def save_scored_rows(prefix: str, scored_df: pd.DataFrame, as_json: bool = False):
    """Save per-response judged rows to results directory (CSV by default)."""
    results_path = ensure_results_dir(prefix)
    if as_json:
        file_path = os.path.join(results_path, SCORED_ROWS_FILE_JSON)
        scored_df.to_json(file_path, orient="records", indent=2)
    else:
        file_path = os.path.join(results_path, SCORED_ROWS_FILE_CSV)
        scored_df.to_csv(file_path, index=False)


def save_metadata(prefix: str, metadata: Dict):
    """Save metadata about the run."""
    results_path = ensure_results_dir(prefix)
    file_path = os.path.join(results_path, METADATA_FILE)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
