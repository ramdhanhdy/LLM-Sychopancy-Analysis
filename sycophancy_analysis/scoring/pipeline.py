# scoring/pipeline.py
"""Pipeline function for scoring stage."""

import pandas as pd
from typing import List, Dict, Optional
from .sss import build_sss
from ..api import MODEL_CONFIGS
from ..data import build_sycophancy_battery
from ..visualization.network import similarity_from_vectors, _symmetrize_clip, _dist_from_sim
from ..data import (
    get_latest_responses, save_sss, save_scored_rows, save_vectors, 
    load_vectors, save_matrices
)


def run_scoring(
    *,
    api_key: str,
    save_prefix: str,
    model_configs: List[Dict] = MODEL_CONFIGS,
    prompts_df: Optional[pd.DataFrame] = None,
    responses_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Run only the scoring stage on existing responses and persist results under `save_prefix`.
    - Loads latest responses via `get_latest_responses(save_prefix)` when `responses_df` not provided.
    - Uses provided `prompts_df` or builds the default battery.
    - Computes SSS, per-response judged rows, and vectors; merges vectors with any prior saved vectors.
    Returns a dict with keys: sss_df, vectors, scored_rows.
    """
    if not save_prefix:
        raise ValueError("run_scoring requires save_prefix to load responses and save outputs.")

    if prompts_df is None:
        prompts_df = build_sycophancy_battery()

    if responses_df is None:
        responses_df = get_latest_responses(save_prefix)
        if responses_df is None or responses_df.empty:
            raise FileNotFoundError(
                f"No responses found under '{save_prefix}'. Run collection first or pass responses_df."
            )

    # Compute SSS and vectors
    sss_df, per_vec, scored_rows = build_sss(prompts_df, responses_df, api_key=api_key)

    # Merge with existing vectors to keep cumulative similarity across runs
    existing_per_vec = load_vectors(save_prefix)
    combined_per_vec = {**existing_per_vec, **per_vec}

    # Persist outputs
    save_sss(save_prefix, sss_df)
    if scored_rows is not None and not scored_rows.empty:
        save_scored_rows(save_prefix, scored_rows)
    save_vectors(save_prefix, combined_per_vec)

    # Also compute and persist similarity/distance matrices for downstream viz
    names, S = similarity_from_vectors(combined_per_vec)
    S = _symmetrize_clip(S)
    D = _dist_from_sim(S)
    save_matrices(save_prefix, names, S, D)

    return {
        "sss_df": sss_df,
        "vectors": combined_per_vec,
        "scored_rows": scored_rows,
        "names": names,
        "S": S,
        "D": D,
    }
