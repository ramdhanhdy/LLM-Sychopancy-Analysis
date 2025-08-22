# visualization/heatmap.py
"""Heatmap visualization functions."""

from typing import List, Optional
import numpy as np


def altair_heatmap(
    names: List[str], 
    S: np.ndarray, 
    save_path: Optional[str] = None
) -> Optional[str]:
    """Create Altair-based heatmap visualization."""
    try:
        import altair as alt
        import pandas as pd
    except ImportError:
        print("Warning: altair not available, skipping heatmap")
        return None

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
    
    return chart.to_json()
