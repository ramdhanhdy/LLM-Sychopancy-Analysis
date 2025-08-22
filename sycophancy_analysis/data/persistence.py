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
    """Save collected responses to a timestamped file (CSV or JSON based on config)."""
    responses_path = ensure_responses_dir(prefix)
    # Generate a timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(responses_path, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    if OUTPUT_FORMAT["responses"] == "json":
        file_path = os.path.join(run_dir, RESPONSES_FILE_JSON)
        responses_df.to_json(file_path, orient="records", indent=2)
    else:
        file_path = os.path.join(run_dir, RESPONSES_FILE_CSV)
        responses_df.to_csv(file_path, index=False)


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
