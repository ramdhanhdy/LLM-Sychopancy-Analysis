# visualization/heatmap.py
"""Heatmap visualization functions."""

from typing import List, Optional
import numpy as np


def altair_heatmap(
    names: List[str],
    S: np.ndarray,
    save_path: Optional[str] = None,
    order: Optional[str] = None,
):
    """Create Altair-based heatmap visualization.

    - When save_path is provided, saves the chart to that path and returns the path on success (or None on failure).
    - When save_path is None, returns the Altair Chart object so callers can .save() themselves.
    - The 'order' parameter is accepted for backward compatibility; it is currently unused.
    """
    try:
        import altair as alt
        import pandas as pd
    except ImportError:
        print("Warning: altair not available, skipping heatmap")
        return None if save_path else None

    # Create long-form data for Altair
    data = []
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            data.append({
                'x': name_i,
                'y': name_j,
                'similarity': float(S[i, j])
            })
    
    df = pd.DataFrame(data)
    
    # Create heatmap
    chart = alt.Chart(df).mark_rect().add_selection(
        alt.selection_single()
    ).encode(
        x=alt.X('x:O', title='Model', sort=names),
        y=alt.Y('y:O', title='Model', sort=names),
        color=alt.Color(
            'similarity:Q',
            title='Similarity',
            scale=alt.Scale(scheme='viridis', domain=[0, 1])
        ),
        tooltip=['x:O', 'y:O', 'similarity:Q']
    ).properties(
        width=400,
        height=400,
        title='Model Similarity Heatmap'
    ).resolve_scale(
        color='independent'
    )
    
    if save_path:
        try:
            chart.save(save_path)
            print(f"Heatmap saved to {save_path}")
            return save_path
        except Exception as e:
            print(f"Warning: Could not save heatmap to {save_path}: {e}")
            return None
    # Return the Chart for callers that want to save or embed it themselves
    return chart
