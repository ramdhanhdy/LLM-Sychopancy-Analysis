# data_manager.py
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

RESULTS_DIR = "results"
RESPONSES_DIR = "responses"
RESPONSES_FILE = "responses.csv"
RUNS_DIR = "runs"
SSS_FILE = "sss_scores.csv"
VECTORS_FILE = "sss_vectors.json"
SIMILARITY_FILE = "similarity_matrix.npy"
DISTANCE_FILE = "distance_matrix.npy"
NAMES_FILE = "model_names.json"
METADATA_FILE = "metadata.json"

def ensure_results_dir(prefix: str) -> str:
    """Ensure the results directory exists."""
    results_path = os.path.join(prefix, RESULTS_DIR)
    os.makedirs(results_path, exist_ok=True)
    return results_path

def ensure_responses_dir(prefix: str) -> str:
    """Ensure the responses directory exists."""
    responses_path = os.path.join(prefix, RESULTS_DIR, RESPONSES_DIR)
    os.makedirs(responses_path, exist_ok=True)
    return responses_path

def save_responses(prefix: str, responses_df: pd.DataFrame):
    """Save collected responses to a timestamped CSV file."""
    responses_path = ensure_responses_dir(prefix)
    # Generate a timestamp for the run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(responses_path, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    file_path = os.path.join(run_dir, RESPONSES_FILE)
    responses_df.to_csv(file_path, index=False)

def load_all_responses(prefix: str) -> pd.DataFrame:
    """Load all responses from all runs."""
    responses_path = ensure_responses_dir(prefix)
    all_responses = []
    
    # Iterate through all run directories
    if os.path.exists(responses_path):
        for run_dir in os.listdir(responses_path):
            run_path = os.path.join(responses_path, run_dir)
            if os.path.isdir(run_path):
                file_path = os.path.join(run_path, RESPONSES_FILE)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    # Add a column to track which run this response is from
                    df['run_id'] = run_dir
                    all_responses.append(df)
    
    if all_responses:
        return pd.concat(all_responses, ignore_index=True)
    else:
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

def save_sss(prefix: str, sss_df: pd.DataFrame):
    """Save SSS scores to a CSV file."""
    results_path = ensure_results_dir(prefix)
    file_path = os.path.join(results_path, SSS_FILE)
    sss_df.to_csv(file_path, index=False)

def load_sss(prefix: str) -> pd.DataFrame:
    """Load previously computed SSS scores from a CSV file."""
    results_path = ensure_results_dir(prefix)
    file_path = os.path.join(results_path, SSS_FILE)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame()

def save_vectors(prefix: str, per_vec: Dict[str, List[float]]):
    """Save SSS vectors to a JSON file."""
    results_path = ensure_results_dir(prefix)
    file_path = os.path.join(results_path, VECTORS_FILE)
    with open(file_path, 'w') as f:
        json.dump(per_vec, f)

def load_vectors(prefix: str) -> Dict[str, List[float]]:
    """Load previously computed SSS vectors from a JSON file."""
    results_path = ensure_results_dir(prefix)
    file_path = os.path.join(results_path, VECTORS_FILE)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        return {}

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

def load_matrices(prefix: str) -> Tuple[Optional[List[str]], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load previously computed similarity and distance matrices."""
    results_path = ensure_results_dir(prefix)
    names_path = os.path.join(results_path, NAMES_FILE)
    S_path = os.path.join(results_path, SIMILARITY_FILE)
    D_path = os.path.join(results_path, DISTANCE_FILE)
    
    if os.path.exists(names_path) and os.path.exists(S_path) and os.path.exists(D_path):
        with open(names_path, 'r') as f:
            names = json.load(f)
        S = np.load(S_path)
        D = np.load(D_path)
        return names, S, D
    else:
        return None, None, None

def save_metadata(prefix: str, metadata: Dict):
    """Save metadata about the run."""
    results_path = ensure_results_dir(prefix)
    file_path = os.path.join(results_path, METADATA_FILE)
    with open(file_path, 'w') as f:
        json.dump(metadata, f)

def load_metadata(prefix: str) -> Dict:
    """Load metadata about the run."""
    results_path = ensure_results_dir(prefix)
    file_path = os.path.join(results_path, METADATA_FILE)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        return {}