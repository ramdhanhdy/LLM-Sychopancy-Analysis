# data/loading.py
"""Data loading functions for retrieving saved results."""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from .persistence import (
    ensure_results_dir, ensure_responses_dir, RESPONSES_FILE_CSV, RESPONSES_FILE_JSON,
    SSS_FILE_CSV, SSS_FILE_JSON, VECTORS_FILE, SIMILARITY_FILE, DISTANCE_FILE,
    NAMES_FILE, METADATA_FILE
)


def _first_existing(*paths: str) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _resolve_file(prefix: str, filename: str) -> Optional[str]:
    """Return existing file path from flat (prefix/filename) or legacy (prefix/results/filename)."""
    flat = os.path.join(prefix, filename)
    legacy = os.path.join(prefix, "results", filename)
    return _first_existing(flat, legacy)


def _resolve_dir(prefix: str, dirname: str) -> List[str]:
    """Return list of existing directories among flat (prefix/dirname) and legacy (prefix/results/dirname)."""
    flat = os.path.join(prefix, dirname)
    legacy = os.path.join(prefix, "results", dirname)
    out = []
    if os.path.isdir(flat):
        out.append(flat)
    if os.path.isdir(legacy) and legacy != flat:
        out.append(legacy)
    return out


def load_all_responses(prefix: str) -> pd.DataFrame:
    """Load all responses from all runs from flat or legacy structure (CSV/JSON)."""
    all_responses: List[pd.DataFrame] = []
    for responses_path in _resolve_dir(prefix, "responses"):
        for run_dir in os.listdir(responses_path):
            run_path = os.path.join(responses_path, run_dir)
            if os.path.isdir(run_path):
                json_path = os.path.join(run_path, RESPONSES_FILE_JSON)
                csv_path = os.path.join(run_path, RESPONSES_FILE_CSV)
                df = None
                if os.path.exists(json_path):
                    df = pd.read_json(json_path, orient="records")
                elif os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                if df is not None:
                    df['run_id'] = run_dir
                    all_responses.append(df)
    if all_responses:
        return pd.concat(all_responses, ignore_index=True)
    return pd.DataFrame(columns=["model", "prompt_id", "response", "run_id"])


def get_latest_responses(prefix: str) -> pd.DataFrame:
    """Get the latest response for each model/prompt combination."""
    all_responses = load_all_responses(prefix)
    if all_responses.empty:
        return all_responses
    
    # Sort by run_id (which contains timestamp) and keep the last occurrence of each model/prompt
    # Assuming run_id format is "run_YYYYMMDD_HHMMSS"
    all_responses['run_timestamp'] = all_responses['run_id'].str.replace('run_', '')
    all_responses['run_datetime'] = pd.to_datetime(all_responses['run_timestamp'], format='%Y%m%d_%H%M%S')
    latest_responses = all_responses.sort_values('run_datetime').groupby(['model', 'prompt_id']).tail(1)
    
    # Drop the extra columns we added for sorting
    latest_responses = latest_responses.drop(columns=['run_timestamp', 'run_datetime'])
    
    return latest_responses.reset_index(drop=True)


def load_sss(prefix: str) -> pd.DataFrame:
    """Load previously computed SSS scores (flat or legacy; JSON preferred)."""
    json_path = _resolve_file(prefix, SSS_FILE_JSON)
    csv_path = _resolve_file(prefix, SSS_FILE_CSV)
    if json_path:
        return pd.read_json(json_path, orient="records")
    if csv_path:
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def load_vectors(prefix: str) -> Dict[str, List[float]]:
    """Load previously computed SSS vectors from JSON (flat or legacy)."""
    file_path = _resolve_file(prefix, VECTORS_FILE)
    if file_path:
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}


def load_matrices(prefix: str) -> Tuple[Optional[List[str]], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load previously computed similarity and distance matrices (flat or legacy)."""
    names_path = _resolve_file(prefix, NAMES_FILE)
    S_path = _resolve_file(prefix, SIMILARITY_FILE)
    D_path = _resolve_file(prefix, DISTANCE_FILE)
    if names_path and S_path and D_path:
        with open(names_path, 'r') as f:
            names = json.load(f)
        S = np.load(S_path)
        D = np.load(D_path)
        return names, S, D
    return None, None, None


def load_metadata(prefix: str) -> Dict:
    """Load metadata about the run (flat or legacy)."""
    file_path = _resolve_file(prefix, METADATA_FILE)
    if file_path:
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}
