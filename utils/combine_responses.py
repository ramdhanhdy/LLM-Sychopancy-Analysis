#!/usr/bin/env python3
"""
Combine scattered response CSVs (collected across different runs) into a single normalized file.

Features
- Reads multiple input CSVs (with possibly different schemas)
- Unions columns and fills missing values
- Adds source metadata: source_run, run_id, source_file, source_priority
- Enriches rows with prompt metadata from dataset/prompt_battery.json
  (stance, strength, ask_devil, topic, persona, is_harmful, prompt_text)
- Optional de-duplication by (model, prompt_id), preferring newer/richer sources
- Saves a combined CSV and prints a summary

Defaults are tailored to this repository's structure.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np

# Default inputs discovered from the project structure and user prompt
DEFAULT_INPUTS = [
    r"results/run_0c/results/responses/run_20250815_202727/responses.csv",
    r"results/run_1/results/responses/run_20250816_134458/responses.csv",
    r"results/run_1b/responses/run_20250819_102725/responses.csv",
]

DEFAULT_OUTPUT = r"results/combined_run_0c_1_1b/responses_combined.json"
PROMPT_BATTERY_PATH = r"dataset/prompt_battery.json"

# Priority ordering when deduplicating: higher number wins
SOURCE_PRIORITY = {
    "run_0c": 1,
    "run_1": 2,
    "run_1b": 3,
}


def _parse_source(path: Path) -> Tuple[str, str]:
    """Infer (source_run, run_id) from a path like
    results/run_1/results/responses/run_20250816_134458/responses.csv
    results/run_1b/responses/run_20250819_102725/responses.csv
    """
    parts = list(path.parts)
    # Guess source_run as the first segment that startswith("run_") and is not a leaf file
    source_run = next((p for p in parts if p.startswith("run_") and p.endswith(("0c", "_1", "_1b")) is False), None)
    # Fallback: choose the first part that starts with run_
    if source_run is None:
        source_run = next((p for p in parts if p.startswith("run_")), "unknown")
    # Guess run_id as the folder under .../responses/<run_id>/responses.csv
    run_id = "unknown"
    try:
        if "responses" in parts:
            idx = parts.index("responses")
            if idx + 1 < len(parts):
                candidate = parts[idx + 1]
                if candidate.startswith("run_"):
                    run_id = candidate
    except Exception:
        pass
    return source_run, run_id


def _load_prompt_battery(battery_path: Path) -> Dict[str, dict]:
    """Load prompt battery and return a mapping prompt_id -> meta dict.
    Expected keys per prompt: prompt_id, text, topic, persona, stance, strength, is_harmful, ask_devil
    """
    if not battery_path.exists():
        return {}
    with battery_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Handle either list or wrapped dict with key like "prompts"
    if isinstance(data, dict):
        items = data.get("prompts") or data.get("data") or data.get("items") or []
    else:
        items = data
    mapping: Dict[str, dict] = {}
    for obj in items:
        # Be tolerant to key variants
        pid = obj.get("prompt_id") or obj.get("id") or obj.get("pid")
        if not pid:
            continue
        mapping[str(pid)] = {
            "prompt_text": obj.get("text") or obj.get("prompt_text") or obj.get("prompt") or "",
            "topic": obj.get("topic"),
            "persona": obj.get("persona"),
            "stance": obj.get("stance"),
            "strength": obj.get("strength"),
            "is_harmful": obj.get("is_harmful"),
            "ask_devil": obj.get("ask_devil"),
        }
    return mapping


def _normalize_dataframe(df: pd.DataFrame, src_path: Path) -> pd.DataFrame:
    """Ensure required base columns exist and attach source metadata."""
    # Ensure base columns
    for col in ("model", "prompt_id", "response"):
        if col not in df.columns:
            df[col] = pd.NA
    # Add source metadata
    source_run, run_id = _parse_source(src_path)
    df["source_run"] = source_run
    df["run_id"] = run_id
    df["source_file"] = str(src_path)
    df["source_priority"] = SOURCE_PRIORITY.get(source_run, 0)
    # Enforce types
    df["prompt_id"] = df["prompt_id"].astype(str)
    df["model"] = df["model"].astype(str)
    df["response"] = df["response"].astype(str)
    return df


def _unionize_columns(frames: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Add any missing columns across frames as NA to achieve a consistent schema."""
    all_cols: List[str] = []
    seen = set()
    for df in frames:
        for c in df.columns:
            if c not in seen:
                seen.add(c)
                all_cols.append(c)
    # Add missing columns per frame
    out: List[pd.DataFrame] = []
    for df in frames:
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = pd.NA
        # Reorder to common ordering
        out.append(df[all_cols])
    return out


def _attach_prompt_meta(df: pd.DataFrame, meta_map: Dict[str, dict]) -> pd.DataFrame:
    if not meta_map:
        return df
    # Map each field
    for key in ("prompt_text", "topic", "persona", "stance", "strength", "is_harmful", "ask_devil"):
        df[key] = df["prompt_id"].map(lambda pid: (meta_map.get(str(pid)) or {}).get(key))
    # Back-compat for scoring.build_sss(): ensure a 'text' column
    if "text" not in df.columns:
        df["text"] = df["prompt_text"]
    else:
        df["text"] = df["text"].fillna(df["prompt_text"])  # fill any missing from prompt_text
    return df


def combine_responses_files(
    inputs: Iterable[os.PathLike | str],
    *,
    output: os.PathLike | str,
    attach_prompt_meta: bool = True,
    dedup: bool = True,
    dedup_keys: Tuple[str, str] = ("model", "prompt_id"),
    output_format: Optional[str] = None,  # 'json', 'jsonl', 'csv' or None to infer by extension
    pretty: bool = False,
) -> pd.DataFrame:
    """Combine CSVs and write to output. Returns the combined DataFrame.

    - inputs: iterable of file paths
    - output: output path (JSON/JSONL/CSV inferred by extension)
    - attach_prompt_meta: whether to enrich with prompt battery fields
    - dedup: if True, drop duplicates by dedup_keys, keeping highest source_priority
    - output_format: override format detection ('json', 'jsonl', 'csv')
    - pretty: pretty-print JSON output
    """
    in_paths = [Path(p) for p in inputs]
    frames: List[pd.DataFrame] = []
    for p in in_paths:
        if not p.exists():
            print(f"[combine_responses] WARN: missing input {p}")
            continue
        try:
            df = pd.read_csv(p, dtype={"prompt_id": str}, keep_default_na=False)
        except Exception:
            # Try with default parser
            df = pd.read_csv(p)
            if "prompt_id" in df.columns:
                df["prompt_id"] = df["prompt_id"].astype(str)
        df = _normalize_dataframe(df, p)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No valid input CSVs were found.")

    frames = _unionize_columns(frames)
    df_all = pd.concat(frames, ignore_index=True)

    # Attach prompt metadata
    if attach_prompt_meta:
        meta_map = _load_prompt_battery(Path(PROMPT_BATTERY_PATH))
        if not meta_map:
            print(f"[combine_responses] WARN: prompt battery not found or empty at {PROMPT_BATTERY_PATH}")
        df_all = _attach_prompt_meta(df_all, meta_map)

    # Deduplicate if requested
    removed = 0
    if dedup and len(df_all) > 0:
        # Keep highest priority per key
        sort_cols = list(dedup_keys) + ["source_priority"]
        df_all = df_all.sort_values(sort_cols, ascending=[True, True, False], kind="mergesort")
        before = len(df_all)
        df_all = df_all.drop_duplicates(subset=list(dedup_keys), keep="first")
        removed = before - len(df_all)

    # Ensure output directory exists
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine output format
    fmt = (output_format or "").lower()
    if not fmt:
        suffix = out_path.suffix.lower()
        if suffix == ".jsonl":
            fmt = "jsonl"
        elif suffix == ".json":
            fmt = "json"
        elif suffix == ".csv":
            fmt = "csv"
        else:
            fmt = "json"  # default

    # Normalize NaNs to None for JSON
    if fmt in ("json", "jsonl"):
        df_json = df_all.replace({np.nan: None})

    # Write
    if fmt == "jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for rec in df_json.to_dict(orient="records"):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    elif fmt == "json":
        records = df_json.to_dict(orient="records")
        with out_path.open("w", encoding="utf-8") as f:
            if pretty:
                json.dump(records, f, ensure_ascii=False, indent=2)
            else:
                json.dump(records, f, ensure_ascii=False, separators=(",", ":"))
    elif fmt == "csv":
        df_all.to_csv(out_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")

    # Summary
    by_source = df_all.groupby("source_run", dropna=False)["prompt_id"].count().sort_index()
    print("[combine_responses] Wrote:", out_path)
    print("[combine_responses] Rows:", len(df_all), f"(removed {removed} duplicates)")
    try:
        print("[combine_responses] By source:")
        for src, cnt in by_source.items():
            print(f"  - {src or 'unknown'}: {cnt}")
    except Exception:
        pass

    return df_all


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Combine scattered response CSVs into one file")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input CSVs (default: known three scattered files)",
        default=DEFAULT_INPUTS,
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output path (infer by extension .json/.jsonl/.csv; default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl", "csv"],
        default=None,
        help="Override output format (otherwise inferred from --output extension)",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON with indent=2",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Keep all rows (no de-dup by model,prompt_id)",
    )
    parser.add_argument(
        "--no-prompt-meta",
        action="store_true",
        help="Do not attach prompt metadata from dataset/prompt_battery.json",
    )
    args = parser.parse_args(argv)

    combine_responses_files(
        inputs=args.inputs,
        output=args.output,
        attach_prompt_meta=not args.no_prompt_meta,
        dedup=not args.keep_all,
        output_format=args.format,
        pretty=args.pretty,
    )


if __name__ == "__main__":
    main()
