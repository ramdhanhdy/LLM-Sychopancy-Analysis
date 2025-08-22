#!/usr/bin/env python3
"""
Score combined responses and compute analysis artifacts:
- Per-response scored rows (LLM judge if OPENROUTER_API_KEY is set; else heuristic)
- Devil's-Advocate Delta (AOS_regular - AOS_devil) per model/topic
- Stance-elasticity curve points and linear fits per model (regular vs devil)

Inputs
- Combined responses produced by utils/combine_responses.py (JSON/JSONL/CSV)
  Must contain: model, prompt_id, response, and prompt metadata columns attached by the combiner:
    text, topic, persona, stance, strength, is_harmful, ask_devil

Outputs (default paths under results/combined_run_0c_1_1b/)
- scored_rows.json
- delta_by_topic.json
- strength_curves.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sycophancy_analysis.scoring.sss import build_sss, fit_elasticity

DEFAULT_INPUT = r"results/combined_run_0c_1_1b/responses_combined.json"
DEFAULT_PREFIX = r"results/combined_run_0c_1_1b"


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        recs = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                recs.append(json.loads(line))
        return pd.DataFrame(recs)
    elif suf == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "records" in data:
            data = data["records"]
        return pd.DataFrame(data)
    elif suf == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input format: {path}")


def _build_prompts_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "prompt_id",
        "text",
        "topic",
        "persona",
        "stance",
        "strength",
        "is_harmful",
        "ask_devil",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Combined file missing required columns for prompts_df: {missing}")
    p = (
        df[cols]
        .drop_duplicates(subset=["prompt_id"], keep="first")
        .reset_index(drop=True)
        .copy()
    )
    # Ensure types
    p["prompt_id"] = p["prompt_id"].astype(str)
    p["stance"] = p["stance"].astype(int)
    p["strength"] = p["strength"].astype(float)
    p["is_harmful"] = p["is_harmful"].astype(bool)
    p["ask_devil"] = p["ask_devil"].astype(bool)
    return p


def _build_responses_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["model", "prompt_id", "response"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Combined file missing required columns for responses_df: {missing}")
    rcols = cols + (["model_slug"] if "model_slug" in df.columns else [])
    r = df[rcols].copy()
    r["prompt_id"] = r["prompt_id"].astype(str)
    return r


def _write_json(path: Path, obj, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        else:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def _compute_delta_by_topic(scored: pd.DataFrame) -> List[Dict]:
    rows: List[Dict] = []
    if scored.empty:
        return rows
    scored["ask_devil"] = scored["ask_devil"].astype(bool)
    for (model, topic), grp in scored.groupby(["model", "topic"], dropna=False):
        reg = grp[~grp["ask_devil"]]["endorse_stance"].mean()
        dev = grp[grp["ask_devil"]]["endorse_stance"].mean()
        cnt_reg = int((~grp["ask_devil"]).sum())
        cnt_dev = int((grp["ask_devil"]).sum())
        rows.append(
            {
                "model": model,
                "topic": topic,
                "AOS_regular": float(reg) if pd.notna(reg) else None,
                "AOS_devil": float(dev) if pd.notna(dev) else None,
                "delta": (float(reg) - float(dev)) if (pd.notna(reg) and pd.notna(dev)) else None,
                "n_regular": cnt_reg,
                "n_devil": cnt_dev,
            }
        )
    return rows


def _ols_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2:
        return 0.0, float(y.mean()) if len(y) else 0.0
    b, a = np.polyfit(x, y, deg=1)  # y = b*x + a
    return float(b), float(a)


def _compute_strength_curves(scored: pd.DataFrame) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    if scored.empty:
        return out
    # Ensure numeric strength
    scored = scored.copy()
    scored["strength"] = scored["strength"].astype(float)
    scored["ask_devil"] = scored["ask_devil"].astype(bool)
    for model, g in scored.groupby("model"):
        mod: Dict[str, Dict] = {}
        for label, sub in [("regular", g[~g["ask_devil"]]), ("devil", g[g["ask_devil"]])]:
            pts = (
                sub.groupby("strength", as_index=False)["endorse_stance"].mean().sort_values("strength")
            )
            x = pts["strength"].to_numpy(dtype=float)
            y = pts["endorse_stance"].to_numpy(dtype=float)
            slope, intercept = _ols_fit(x, y)
            # AE using standardized slope function for reference
            AE = fit_elasticity(x.tolist(), y.tolist()) if len(x) >= 2 else 0.0
            mod[label] = {
                "points": [{"strength": float(xx), "endorse": float(yy)} for xx, yy in zip(x, y)],
                "ols_slope": slope,
                "ols_intercept": intercept,
                "AE_standardized": float(AE),
            }
        out[model] = mod
    return out


def _extract_existing_scores(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Extract existing SSS scores from source run files to avoid re-running LLM judges.
    Returns scored DataFrame if scores are found, None otherwise.
    """
    if df.empty:
        return None
    
    # Check if scores already exist in the combined data
    score_cols = ['endorse_stance', 'harm_validation', 'flattery_density', 'challenge']
    if all(col in df.columns for col in score_cols):
        print("[_extract_existing_scores] Scores already present in combined data")
        return df
    
    # Get the project root directory (parent of utils/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / "results"
    
    # Get unique source runs from the data
    source_runs = []
    if 'source_run' in df.columns:
        source_runs.extend(df['source_run'].dropna().unique())
    if 'run_id' in df.columns:
        source_runs.extend(df['run_id'].dropna().unique())
    
    source_runs = list(set(source_runs))  # Remove duplicates
    print(f"[_extract_existing_scores] Found source runs: {source_runs}")
    print(f"[_extract_existing_scores] Looking in results directory: {results_dir}")
    
    # Scan for existing run directories to avoid searching non-existent paths
    existing_run_dirs = set()
    if results_dir.exists():
        for item in results_dir.iterdir():
            if item.is_dir() and item.name.startswith('run_'):
                existing_run_dirs.add(item.name)
    
    print(f"[_extract_existing_scores] Found existing run directories: {sorted(existing_run_dirs)}")
    
    # Try to load existing scored_rows from source runs
    scored_dfs = []
    
    for source_run in source_runs:
        # Extract base run name (e.g., "run_1b" from "run_20250819_102725")
        base_run = None
        if source_run.startswith("run_"):
            # Try to match pattern like run_20250819_102725 -> run_1b
            if "20250819" in source_run:
                base_run = "run_1b"
            elif "20250815" in source_run:
                base_run = "run_0c"
            elif "20250816" in source_run:
                base_run = "run_1"
            # Add more mappings as needed
        else:
            base_run = source_run
        
        # Only check paths for directories that actually exist
        possible_paths = []
        if base_run and base_run in existing_run_dirs:
            possible_paths.append(results_dir / base_run / "scored_rows.csv")
        if source_run in existing_run_dirs:
            possible_paths.append(results_dir / source_run / "scored_rows.csv")
            possible_paths.append(results_dir / source_run / "results" / "scored_rows.csv")
        
        found_scores = False
        for path in possible_paths:
            if path.exists():
                try:
                    scored_run = pd.read_csv(path)
                    scored_run['source_run'] = source_run
                    scored_dfs.append(scored_run)
                    print(f"[_extract_existing_scores] Loaded scores from {path} ({len(scored_run)} rows)")
                    found_scores = True
                    break
                except Exception as e:
                    print(f"[_extract_existing_scores] Failed to load {path}: {e}")
            else:
                print(f"[_extract_existing_scores] Path does not exist: {path}")
        
        if not found_scores:
            print(f"[_extract_existing_scores] No scored_rows found for {source_run} (base: {base_run})")
            print(f"[_extract_existing_scores] Checked paths: {[str(p) for p in possible_paths]}")
    
    if not scored_dfs:
        print("[_extract_existing_scores] No existing scores found in any source runs")
        return None
    
    # Combine all scored data
    all_scored = pd.concat(scored_dfs, ignore_index=True)
    print(f"[_extract_existing_scores] Combined {len(all_scored)} scored rows from {len(scored_dfs)} runs")
    
    # Merge with original combined data to get all metadata
    merge_cols = ['model', 'prompt_id']
    if 'source_run' in df.columns and 'source_run' in all_scored.columns:
        merge_cols.append('source_run')
    
    print(f"[_extract_existing_scores] Merging on columns: {merge_cols}")
    merged = df.merge(all_scored, on=merge_cols, how='left', suffixes=('', '_scored'))
    
    # Check merge success
    score_coverage = merged[score_cols[0]].notna().mean() if score_cols[0] in merged.columns else 0
    print(f"[_extract_existing_scores] Score coverage: {score_coverage:.1%} ({merged[score_cols[0]].notna().sum()}/{len(merged)} responses)")
    
    if score_coverage > 0.8:  # If we have scores for >80% of responses
        return merged
    else:
        print(f"[_extract_existing_scores] Low score coverage ({score_coverage:.1%}), will run scoring")
        return None


def score_and_metrics(
    input_path: os.PathLike | str = DEFAULT_INPUT,
    *,
    output_prefix: os.PathLike | str = DEFAULT_PREFIX,
    pretty: bool = False,
    api_key: Optional[str] = None,
    force_rescore: bool = False,
) -> Dict[str, Path]:
    in_path = Path(input_path)
    df = _read_any(in_path)
    prompts_df = _build_prompts_df(df)
    responses_df = _build_responses_df(df)
    
    # Try to extract existing scores first (unless forced to rescore)
    scored = None
    if not force_rescore:
        scored = _extract_existing_scores(df)
    
    if scored is None:
        # Fall back to running build_sss
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        total_responses = len(responses_df)
        print(f"[score_and_metrics] Running build_sss() on {total_responses} responses - this may use API credits")
        sss_df, _, scored = build_sss(prompts_df, responses_df, api_key=key)
    else:
        # Count how many responses already have scores vs need scoring
        score_cols = ['endorse_stance', 'harm_validation', 'flattery_density', 'challenge']
        has_scores = scored[score_cols[0]].notna().sum()
        total_responses = len(scored)
        needs_scoring = total_responses - has_scores
        
        if needs_scoring > 0:
            print(f"[score_and_metrics] Using existing scores for {has_scores} responses, need to score {needs_scoring} more")
        else:
            print(f"[score_and_metrics] Using existing scores for all {total_responses} responses")

    # Outputs
    prefix = Path(output_prefix)
    out_scored = prefix / "scored_rows.json"
    out_delta = prefix / "delta_by_topic.json"
    out_curves = prefix / "strength_curves.json"

    _write_json(out_scored, scored.replace({np.nan: None}).to_dict(orient="records"), pretty=pretty)
    _write_json(out_delta, _compute_delta_by_topic(scored), pretty=pretty)
    _write_json(out_curves, _compute_strength_curves(scored), pretty=pretty)

    print("[score_and_metrics] Wrote:")
    print(" -", out_scored)
    print(" -", out_delta)
    print(" -", out_curves)

    return {"scored_rows": out_scored, "delta_by_topic": out_delta, "strength_curves": out_curves}


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Score combined responses and compute Δ & curves")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help=f"Combined responses path (default: {DEFAULT_INPUT})")
    parser.add_argument(
        "--output-prefix",
        "-o",
        default=DEFAULT_PREFIX,
        help=f"Output prefix directory (default: {DEFAULT_PREFIX})",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty JSON output")
    parser.add_argument("--force-rescore", action="store_true", help="Force re-running LLM judges even if existing scores found")
    args = parser.parse_args(argv)

    score_and_metrics(
        input_path=args.input, 
        output_prefix=args.output_prefix, 
        pretty=args.pretty,
        force_rescore=args.force_rescore
    )


if __name__ == "__main__":
    main()
