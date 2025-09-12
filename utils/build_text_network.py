#!/usr/bin/env python3
"""
Build a model-level network from RAW RESPONSES (text), without using Sycophancy Stylometry vectors.

- Input: one or more response tables (CSV/JSON/JSONL). Typically use the combined JSON you created.
- For each (model, prompt_id), we keep the latest run (if run_id present), else last by index.
- For each model, we concatenate all its responses into a single document.
- We compute TF-IDF features and cosine similarity between model documents.
- We then create a network using the existing visualization utilities (UMAP layout, MST backbone, kNN graph, Leiden).
- Outputs are saved under --save_prefix: network image (PNG), heatmap (HTML), matrices, and metadata.

This does NOT modify or remove the existing stylometry-based chart.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Allow running this script directly from utils/ by adding project root to sys.path if needed
try:
    from sycophancy_analysis.data.persistence import ensure_results_dir, save_matrices, save_metadata
    from sycophancy_analysis.visualization import plot_network, altair_heatmap
    from sycophancy_analysis.visualization.network import (
        umap_layout,
        mst_backbone,
        knn_graph_from_similarity,
        detect_communities,
        weighted_modularity,
        conductance_per_community,
        participation_coefficient,
    )
except ModuleNotFoundError:
    import sys as _sys
    _sys.path.append(str(Path(__file__).resolve().parents[1]))
    from sycophancy_analysis.data.persistence import ensure_results_dir, save_matrices, save_metadata
    from sycophancy_analysis.visualization import plot_network, altair_heatmap
    from sycophancy_analysis.visualization.network import (
        umap_layout,
        mst_backbone,
        knn_graph_from_similarity,
        detect_communities,
        weighted_modularity,
        conductance_per_community,
        participation_coefficient,
    )


# ------------------------------
# IO helpers
# ------------------------------

def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    if suffix == ".jsonl":
        recs: List[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except Exception:
                    pass
        return pd.DataFrame(recs)
    # Fallback
    return pd.read_csv(path)


def _latest_per_model_prompt(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    dd = df.copy()
    # normalize essential columns
    for c in ("model", "prompt_id", "response"):
        if c not in dd.columns:
            dd[c] = pd.NA
    dd["model"] = dd["model"].astype(str)
    dd["prompt_id"] = dd["prompt_id"].astype(str)
    dd["response"] = dd["response"].astype(str)

    if "run_id" not in dd.columns:
        return dd.sort_index().groupby(["model", "prompt_id"], as_index=False).tail(1).reset_index(drop=True)

    # Expect run_id like run_YYYYMMDD_HHMMSS
    ts = dd["run_id"].astype(str).str.replace("run_", "", regex=False)
    dd["run_datetime"] = pd.to_datetime(ts, errors="coerce", format="%Y%m%d_%H%M%S")
    # Sort with NaT first so valid timestamps win when taking tail(1)
    dd = dd.sort_values(["model", "prompt_id", "run_datetime", "run_id"], na_position="first")
    latest = dd.groupby(["model", "prompt_id"], as_index=False).tail(1)
    return latest.reset_index(drop=True)


# ------------------------------
# Core computation
# ------------------------------

def _build_docs_per_model(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (names, docs) where each doc is the concatenation of all responses for that model."""
    if df.empty:
        return [], []
    grouped = df.groupby("model")["response"].apply(lambda s: "\n\n".join([str(x) for x in s if isinstance(x, str) and x.strip() != ""]))
    names = list(grouped.index)
    docs = list(grouped.values)
    return names, docs


def _tfidf_cosine(docs: List[str], *, stop_words: str | None = "english", ngram_range=(1, 2), min_df=1) -> np.ndarray:
    vec = TfidfVectorizer(stop_words=stop_words, ngram_range=ngram_range, min_df=min_df)
    X = vec.fit_transform(docs)
    S = cosine_similarity(X)  # already symmetric, in [0,1]
    # Numerical guard
    S = np.clip(S, 0.0, 1.0)
    return S


def build_text_network(
    inputs: Iterable[str],
    *,
    output_prefix: str,
    knn_k: int = 8,
    leiden_resolution: float = 1.0,
    bridge_threshold: float = 0.5,
    stop_words: str | None = "english",
    ngram_range=(1, 2),
    min_df: int = 1,
    fig_width: float = 20.0,
    fig_height: float = 15.0,
) -> dict:
    # 1) Load inputs
    frames: List[pd.DataFrame] = []
    for p in inputs:
        path = Path(p)
        if not path.exists():
            print(f"[textnet] WARN missing input: {p}")
            continue
        df = _read_table(path)
        frames.append(df)
    if not frames:
        raise SystemExit("No valid input files provided.")

    all_df = pd.concat(frames, ignore_index=True)

    # 2) Reduce to latest per (model, prompt_id)
    latest = _latest_per_model_prompt(all_df)
    print(f"[textnet] latest per (model,prompt): {len(latest)} rows, models={latest['model'].nunique()} prompts={latest['prompt_id'].nunique()}")

    # 3) Build docs per model
    names, docs = _build_docs_per_model(latest)
    print(f"[textnet] corpus: {len(names)} model documents")

    # 4) TF-IDF cosine similarity
    S = _tfidf_cosine(docs, stop_words=stop_words, ngram_range=ngram_range, min_df=min_df)
    D = 1.0 - S

    # 5) Layout and graphs
    pos = umap_layout(names, D)
    G_backbone = mst_backbone(names, D)
    G_comm = knn_graph_from_similarity(names, S, k=knn_k, sym_mode="max")
    node_to_comm, _ = detect_communities(G_comm, method="leiden", resolution=leiden_resolution, seed=42)
    Q = weighted_modularity(G_comm, node_to_comm)
    cond = conductance_per_community(G_comm, node_to_comm)
    part = participation_coefficient(G_comm, node_to_comm)

    # 6) Plot
    results_path = ensure_results_dir(output_prefix)
    fig = plot_network(
        names,
        S,
        pos,
        G_backbone,
        node_to_comm,
        Q=Q,
        conductance=cond,
        participation=part,
        title="Response Text Similarity • Model Network",
        bridge_threshold=bridge_threshold,
        figsize=(float(fig_width), float(fig_height)),
    )
    out_img = os.path.join(results_path, "text_network.png")
    fig.savefig(out_img, dpi=180, bbox_inches="tight")
    print(f"[textnet] saved network image: {out_img}")

    # 7) Heatmap
    try:
        chart = altair_heatmap(names, S, order="spectral")
        out_heat = os.path.join(results_path, "text_heatmap.html")
        chart.save(out_heat)
        print(f"[textnet] saved heatmap: {out_heat}")
    except Exception as e:
        print(f"[textnet] heatmap not saved: {e}")

    # 8) Save matrices + metadata
    save_matrices(output_prefix, names, S, D)
    save_metadata(
        output_prefix,
        {
            "method": "tfidf_cosine_from_responses",
            "inputs": list(inputs),
            "knn_k": knn_k,
            "leiden_resolution": leiden_resolution,
            "bridge_threshold": bridge_threshold,
            "vectorizer": {
                "stop_words": stop_words,
                "ngram_range": list(ngram_range),
                "min_df": min_df,
            },
        },
    )
    print(f"[textnet] saved matrices and metadata under '{output_prefix}'")

    return {
        "names": names,
        "S": S,
        "D": D,
        "pos": pos,
        "backbone_edges": list(G_backbone.edges()),
        "node_to_comm": node_to_comm,
        "modularity_Q": Q,
        "conductance": cond,
        "participation": part,
    }


# ------------------------------
# CLI
# ------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a model network from raw responses (TF-IDF cosine)")
    ap.add_argument("inputs", nargs="+", help="Input files (CSV/JSON/JSONL). Typically the combined responses JSON.")
    ap.add_argument("--save_prefix", required=True, help="Output directory/prefix for results (e.g., results/textnet_0c_1b_2b)")
    ap.add_argument("--knn_k", type=int, default=8)
    ap.add_argument("--leiden_resolution", type=float, default=1.0)
    ap.add_argument("--bridge_threshold", type=float, default=0.5)
    ap.add_argument("--stop_words", type=str, default="english")
    ap.add_argument("--min_df", type=int, default=1)
    ap.add_argument("--ngram_max", type=int, default=2, help="Use unigrams..ngram_max in TF-IDF")
    ap.add_argument("--fig_width", type=float, default=20.0, help="Figure width in inches (network chart)")
    ap.add_argument("--fig_height", type=float, default=15.0, help="Figure height in inches (network chart)")

    args = ap.parse_args(list(argv) if argv is not None else None)

    ngram_range = (1, max(1, int(args.ngram_max)))

    build_text_network(
        args.inputs,
        output_prefix=args.save_prefix,
        knn_k=args.knn_k,
        leiden_resolution=args.leiden_resolution,
        bridge_threshold=args.bridge_threshold,
        stop_words=(None if args.stop_words.lower() == "none" else args.stop_words),
        ngram_range=ngram_range,
        min_df=args.min_df,
        fig_width=args.fig_width,
        fig_height=args.fig_height,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
