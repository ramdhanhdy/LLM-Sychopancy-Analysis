#!/usr/bin/env python3
"""
Convert legacy responses.csv files to responses.json for all runs.

Supports both the legacy nested layout:
  <prefix>/results/responses/run_YYYYMMDD_HHMMSS/responses.csv
and the flat layout:
  <prefix>/responses/run_YYYYMMDD_HHMMSS/responses.csv

Usage examples:
  # Convert for a specific save_prefix directory
  python utils/convert_responses_csv_to_json.py --prefix results/run_1b

  # Recursively scan a root directory (e.g., "results/") and convert all found runs
  python utils/convert_responses_csv_to_json.py --scan-root results

Options:
  --overwrite     Overwrite responses.json if it already exists (default: skip existing)
  --delete-csv    Delete responses.csv after successful conversion (default: keep)
  --dry-run       Show what would be converted without writing files
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List, Set

import pandas as pd

# Reuse file-name constants from the package to avoid drift
try:
    from sycophancy_analysis.data.persistence import (
        RESPONSES_FILE_CSV,
        RESPONSES_FILE_JSON,
    )
except Exception:
    # Fallbacks if the package isn't importable for any reason
    RESPONSES_FILE_CSV = "responses.csv"
    RESPONSES_FILE_JSON = "responses.json"


def find_run_dirs_for_prefix(prefix: str) -> List[str]:
    """Return run directories under the prefix for both flat and legacy layouts."""
    run_dirs: List[str] = []
    # Flat: <prefix>/responses/run_*/
    flat_responses = os.path.join(prefix, "responses")
    if os.path.isdir(flat_responses):
        for d in sorted(os.listdir(flat_responses)):
            p = os.path.join(flat_responses, d)
            if os.path.isdir(p):
                run_dirs.append(p)
    # Legacy: <prefix>/results/responses/run_*/
    legacy_responses = os.path.join(prefix, "results", "responses")
    if os.path.isdir(legacy_responses):
        for d in sorted(os.listdir(legacy_responses)):
            p = os.path.join(legacy_responses, d)
            if os.path.isdir(p):
                run_dirs.append(p)
    return run_dirs


def discover_prefixes(scan_root: str) -> List[str]:
    """Walk scan_root and return directories that look like save_prefix roots.

    A directory qualifies as a prefix if it contains either:
      - a child directory named 'responses', or
      - a child directory path 'results/responses'.
    """
    prefixes: Set[str] = set()
    scan_root = os.path.abspath(scan_root)
    for dirpath, dirnames, _ in os.walk(scan_root):
        # Flat layout indicator
        if "responses" in dirnames:
            prefixes.add(dirpath)
        # Legacy layout indicator
        if "results" in dirnames:
            if os.path.isdir(os.path.join(dirpath, "results", "responses")):
                prefixes.add(dirpath)
    return sorted(prefixes)


def convert_csv_to_json_in_dir(run_dir: str, *, overwrite: bool, delete_csv: bool, dry_run: bool) -> bool:
    """Convert a single run directory's responses.csv to responses.json.

    Returns True if a conversion was performed or file existed; False if skipped.
    """
    csv_path = os.path.join(run_dir, RESPONSES_FILE_CSV)
    json_path = os.path.join(run_dir, RESPONSES_FILE_JSON)

    if not os.path.exists(csv_path):
        # Nothing to convert
        return False

    if os.path.exists(json_path) and not overwrite:
        print(f"[skip] JSON already exists: {json_path}")
        return True

    if dry_run:
        print(f"[dry-run] Would convert: {csv_path} -> {json_path}")
        return True

    # Load CSV and write JSON
    df = pd.read_csv(csv_path)
    df.to_json(json_path, orient="records", indent=2)
    print(f"[ok] Wrote JSON: {json_path}")

    if delete_csv:
        try:
            os.remove(csv_path)
            print(f"[ok] Deleted CSV: {csv_path}")
        except Exception as e:
            print(f"[warn] Could not delete CSV {csv_path}: {e}")

    return True


def process_prefix(prefix: str, *, overwrite: bool, delete_csv: bool, dry_run: bool) -> int:
    count = 0
    for run_dir in find_run_dirs_for_prefix(prefix):
        if convert_csv_to_json_in_dir(run_dir, overwrite=overwrite, delete_csv=delete_csv, dry_run=dry_run):
            count += 1
    return count


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert legacy responses.csv to responses.json across runs.")
    ap.add_argument("--prefix", action="append", default=[], help="Save prefix directory to process (can be used multiple times)")
    ap.add_argument("--scan-root", default=None, help="Root directory to recursively scan for prefixes (e.g., 'results')")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite responses.json if it already exists")
    ap.add_argument("--delete-csv", action="store_true", help="Delete responses.csv after successful conversion")
    ap.add_argument("--dry-run", action="store_true", help="Show planned conversions without writing")

    args = ap.parse_args(list(argv) if argv is not None else None)

    prefixes: List[str] = []
    prefixes.extend(args.prefix)
    if args.scan_root and not prefixes:
        prefixes = discover_prefixes(args.scan_root)
        if not prefixes:
            print(f"No prefixes found under '{args.scan_root}'. Nothing to do.")
            return 0

    if not prefixes:
        ap.error("Provide at least one --prefix or a --scan-root to discover prefixes.")

    total_converted = 0
    for p in prefixes:
        p_abs = os.path.abspath(p)
        if not os.path.isdir(p_abs):
            print(f"[warn] Not a directory, skipping: {p_abs}")
            continue
        n = process_prefix(p_abs, overwrite=args.overwrite, delete_csv=args.delete_csv, dry_run=args.dry_run)
        print(f"[info] {p_abs}: processed {n} run(s)")
        total_converted += n

    print(f"[done] Total runs processed: {total_converted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
