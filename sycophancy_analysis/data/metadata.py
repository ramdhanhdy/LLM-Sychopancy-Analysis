# data/metadata.py
"""Metadata generation functions for comprehensive run tracking."""

import os
import sys
import platform
import hashlib
import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
from ..api import OUTPUT_FORMAT
from .persistence import RESULTS_DIR


def create_comprehensive_metadata(
    models_analyzed: List[str],
    knn_k: int,
    leiden_resolution: float,
    bridge_threshold: float,
    model_configs: List[Dict],
    scoring_config: Dict,
    reasoning_model_slugs: set,
    prompts_df: pd.DataFrame,
    sss_df: pd.DataFrame,
    modularity_Q: float,
    conductance: Dict[int, float],
    participation: Dict[str, float],
    save_prefix: str,
    **kwargs
) -> Dict[str, Any]:
    """Create comprehensive metadata for the run."""
    run_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Environment information
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "architecture": platform.architecture()[0]
    }
    
    # Prompt battery fingerprint
    prompt_texts = prompts_df['text'].tolist() if 'text' in prompts_df.columns else []
    prompt_hash = hashlib.sha256(''.join(prompt_texts).encode()).hexdigest()[:16]
    
    # Graph statistics
    cond_vals = list(conductance.values()) if conductance else []
    bridge_count = len([p for p in participation.values() if p >= bridge_threshold])
    
    # Subgroup aggregates by topic, stance, etc.
    subgroup_stats = {}
    if not sss_df.empty and 'model' in sss_df.columns:
        # Basic aggregates
        subgroup_stats = {
            "mean_AOS": float(sss_df['AOS'].mean()) if 'AOS' in sss_df.columns else 0.0,
            "mean_CCR": float(sss_df['CCR'].mean()) if 'CCR' in sss_df.columns else 0.0,
            "mean_HVS": float(sss_df['HVS'].mean()) if 'HVS' in sss_df.columns else 0.0,
            "mean_SI": float(sss_df['SI'].mean()) if 'SI' in sss_df.columns else 0.0
        }
    
    # File manifest (basic) - reflect chosen output format
    si_table_ext = "json" if OUTPUT_FORMAT["si_table"] == "json" else "csv"
    file_manifest = {
        "sss_scores_file": f"{save_prefix}_sycophancy_scores.{si_table_ext}",
        "network_plot": f"{save_prefix}_network.png",
        "heatmap": f"{save_prefix}_heatmap.html",
        "results_dir": os.path.join(save_prefix, RESULTS_DIR)
    }
    
    metadata = {
        "schema_version": "1.1.0",
        "run_identity": {
            "run_id": run_id,
            "timestamp": timestamp,
            "save_prefix": save_prefix,
            "prompt_battery_hash": prompt_hash,
            "total_prompts": len(prompts_df)
        },
        "environment": env_info,
        "config_snapshot": {
            "scoring_config": scoring_config,
            "model_configs": model_configs,
            "reasoning_model_slugs": list(reasoning_model_slugs)
        },
        "analysis_params": {
            "knn_k": knn_k,
            "leiden_resolution": leiden_resolution,
            "bridge_threshold": bridge_threshold
        },
        "graph_stats": {
            "modularity_Q": float(modularity_Q),
            "median_conductance": float(np.median(cond_vals)) if cond_vals else 0.0,
            "bridge_count": bridge_count,
            "communities_count": len(set(conductance.keys())) if conductance else 0,
            "backbone_edges_count": kwargs.get('backbone_edges_count', 0)
        },
        "models_analyzed": models_analyzed,
        "subgroup_aggregates": subgroup_stats,
        "file_manifest": file_manifest
    }
    
    return metadata
