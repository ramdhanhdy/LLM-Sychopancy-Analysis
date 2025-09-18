#!/usr/bin/env python3
"""
Generate delta_by_topic.json and strength_curves.json from existing scored_rows.csv
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sycophancy_analysis.scoring.sss import fit_elasticity

def ols_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Simple OLS fit returning slope and intercept"""
    if len(x) < 2:
        return 0.0, float(y.mean()) if len(y) else 0.0
    b, a = np.polyfit(x, y, deg=1)  # y = b*x + a
    return float(b), float(a)

def compute_delta_by_topic(scored: pd.DataFrame) -> List[Dict]:
    """Compute devil's advocate delta by topic"""
    rows = []
    if scored.empty:
        return rows
    
    scored["ask_devil"] = scored["ask_devil"].astype(bool)
    for (model, topic), grp in scored.groupby(["model", "topic"], dropna=False):
        reg = grp[~grp["ask_devil"]]["endorse_stance"].mean()
        dev = grp[grp["ask_devil"]]["endorse_stance"].mean()
        cnt_reg = int((~grp["ask_devil"]).sum())
        cnt_dev = int((grp["ask_devil"]).sum())
        rows.append({
            "model": model,
            "topic": topic,
            "AOS_regular": float(reg) if pd.notna(reg) else None,
            "AOS_devil": float(dev) if pd.notna(dev) else None,
            "delta": (float(reg) - float(dev)) if (pd.notna(reg) and pd.notna(dev)) else None,
            "n_regular": cnt_reg,
            "n_devil": cnt_dev,
        })
    return rows

def compute_strength_curves(scored: pd.DataFrame) -> Dict[str, Dict]:
    """Compute stance elasticity curves by model"""
    out = {}
    if scored.empty:
        return out
    
    scored = scored.copy()
    scored["strength"] = scored["strength"].astype(float)
    scored["ask_devil"] = scored["ask_devil"].astype(bool)
    
    for model, g in scored.groupby("model"):
        mod = {}
        for label, sub in [("regular", g[~g["ask_devil"]]), ("devil", g[g["ask_devil"]])]:
            pts = (
                sub.groupby("strength", as_index=False)["endorse_stance"]
                .mean()
                .sort_values("strength")
            )
            x = pts["strength"].to_numpy(dtype=float)
            y = pts["endorse_stance"].to_numpy(dtype=float)
            slope, intercept = ols_fit(x, y)
            AE = fit_elasticity(x.tolist(), y.tolist()) if len(x) >= 2 else 0.0
            
            mod[label] = {
                "points": [{"strength": float(xx), "endorse": float(yy)} for xx, yy in zip(x, y)],
                "ols_slope": slope,
                "ols_intercept": intercept,
                "AE_standardized": float(AE),
            }
        out[model] = mod
    return out

def main():
    # Load the existing scored data
    input_path = Path("results/combined_run_0c_1_1b/scored_rows.csv")
    output_dir = Path("results/combined_run_0c_1_1b")
    
    print(f"Loading scored data from: {input_path}")
    scored_df = pd.read_csv(input_path)
    print(f"Loaded {len(scored_df)} scored responses")
    
    # Generate delta by topic
    print("Computing delta by topic...")
    delta_data = compute_delta_by_topic(scored_df)
    delta_path = output_dir / "delta_by_topic.json"
    with open(delta_path, 'w') as f:
        json.dump(delta_data, f, indent=2)
    print(f"Wrote {len(delta_data)} model-topic pairs to {delta_path}")
    
    # Generate strength curves
    print("Computing strength curves...")
    curves_data = compute_strength_curves(scored_df)
    curves_path = output_dir / "strength_curves.json"
    with open(curves_path, 'w') as f:
        json.dump(curves_data, f, indent=2)
    print(f"Wrote strength curves for {len(curves_data)} models to {curves_path}")
    
    print("\nFiles generated successfully! Your notebook should now work.")

if __name__ == "__main__":
    main()
