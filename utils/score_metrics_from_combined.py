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


def score_and_metrics(
    input_path: os.PathLike | str = DEFAULT_INPUT,
    *,
    output_prefix: os.PathLike | str = DEFAULT_PREFIX,
    pretty: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Path]:
    in_path = Path(input_path)
    df = _read_any(in_path)
    prompts_df = _build_prompts_df(df)
    responses_df = _build_responses_df(df)
    key = api_key or os.environ.get("OPENROUTER_API_KEY")

    sss_df, _, scored = build_sss(prompts_df, responses_df, api_key=key)

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
    args = parser.parse_args(argv)

    score_and_metrics(input_path=args.input, output_prefix=args.output_prefix, pretty=args.pretty)


if __name__ == "__main__":
    main()
