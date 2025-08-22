# pipeline.py
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Optional
from .api import MODEL_CONFIGS, SCORING_CONFIG, OUTPUT_FORMAT, annotate_reasoning_support, chat_one, model_supports_reasoning
from .data import build_sycophancy_battery
from .scoring import (
    PromptMeta, PromptScores, score_response, score_response_llm,
    build_sss, ModelSSS, fit_elasticity, compute_sycophancy_index,
)
from .visualization import plot_network, altair_heatmap, create_network_sidecar
from .visualization.network import (
    similarity_from_vectors, _symmetrize_clip, _dist_from_sim,
    umap_layout, mst_backbone, knn_graph_from_similarity,
    detect_communities, weighted_modularity,
    conductance_per_community, participation_coefficient
)
from .data import (
    collect_responses, load_all_responses, save_responses, load_sss, save_sss, load_vectors, save_vectors,
    load_matrices, save_matrices, load_metadata, save_metadata, get_latest_responses,
    create_comprehensive_metadata, save_scored_rows
)

def run_sycophancy_pipeline(
    *,
    api_key: str,
    model_configs: List[Dict] = MODEL_CONFIGS,
    samples_per_prompt: int = 1,
    temperature: float = 0.2,
    system_prompt: Optional[str] = None,
    knn_k: int = 8,
    leiden_resolution: float = 1.0,
    bridge_threshold: float = 0.5,
    save_prefix: Optional[str] = None,
) -> Dict:
    # 1) Prompts
    prompts_df = build_sycophancy_battery()
    print(f"Battery: {len(prompts_df)} prompts.")

    # 2) Responses
    # Collect new responses
    new_responses_df = collect_responses(
        model_configs=model_configs,
        prompts_df=prompts_df,
        api_key=api_key,
        samples_per_prompt=samples_per_prompt,
        base_temperature=temperature,
        system_prompt=system_prompt,
        save_prefix=save_prefix, # Pass save_prefix to collect_responses
    )
    
    # Load all responses and get the latest for each model/prompt combination
    if save_prefix:
        responses_df = get_latest_responses(save_prefix)
    else:
        responses_df = new_responses_df
        
    print(f"Using {len(responses_df)} responses for SSS calculation.")

    # 3) SSS + Similarity
    # Build SSS for the current set of responses
    sss_df, per_vec, scored_rows = build_sss(prompts_df, responses_df, api_key=api_key)
    
    # Load existing SSS vectors if available
    existing_per_vec = {}
    if save_prefix:
        existing_per_vec = load_vectors(save_prefix)
    
    # Combine existing and new vectors
    # This ensures that the similarity matrix includes all models analyzed so far
    combined_per_vec = {**existing_per_vec, **per_vec}
    
    # Save SSS, per-response judged rows, and vectors
    if save_prefix:
        save_sss(save_prefix, sss_df)
        # New: persist per-response judged rows next to SSS
        if scored_rows is not None and not scored_rows.empty:
            save_scored_rows(save_prefix, scored_rows)
        save_vectors(save_prefix, combined_per_vec)
        
    names, S = similarity_from_vectors(combined_per_vec)
    S = _symmetrize_clip(S)
    D = _dist_from_sim(S)
    
    # Save matrices
    if save_prefix:
        save_matrices(save_prefix, names, S, D)

    # 4) Layout + Backbone
    pos = umap_layout(names, D)
    G_backbone = mst_backbone(names, D)

    # 5) Communities on k-NN graph
    G_comm = knn_graph_from_similarity(names, S, k=knn_k, sym_mode="max")
    node_to_comm, method_used = detect_communities(
        G_comm, method="leiden", resolution=leiden_resolution, seed=42
    )
    Q = weighted_modularity(G_comm, node_to_comm)
    cond = conductance_per_community(G_comm, node_to_comm)
    part = participation_coefficient(G_comm, node_to_comm)

    # 6) Plot
    fig = plot_network(
        names,
        S,
        pos,
        G_backbone,
        node_to_comm,
        Q=Q,
        conductance=cond,
        participation=part,
        title="Sycophancy Stylometry • Backbone + Admixture-ready",
        bridge_threshold=bridge_threshold,
    )
    if save_prefix:
        fig.savefig(f"{save_prefix}_network.png", dpi=180, bbox_inches="tight")
        # Create network sidecar metadata
        create_network_sidecar(
            names=names,
            pos=pos,
            G_backbone=G_backbone,
            node_to_comm=node_to_comm,
            participation=part,
            bridge_threshold=bridge_threshold,
            admixtures=None,  # Can be extended later with admixture detection
            save_path=f"{save_prefix}_network.meta.json"
        )

    # 7) Heatmap (optional)
    # Note: Altair might have issues with large datasets or in some environments
    # Leaving this as is for now, but it might need adjustments
    if save_prefix:  # Only try to save if save_prefix is provided
        try:
            chart = altair_heatmap(names, S, order="spectral")
            chart.save(f"{save_prefix}_heatmap.html")
        except Exception as e:
            print(f"Could not save heatmap: {e}")

    # 8) Sycophancy Index table
    # For the SI table, we only include models from the current run
    # But we could modify this to include all models if desired
    rank_df = sss_df.copy()
    
    # Add intermediate terms used in SI calculation
    rank_df["ca"] = 1.0 - rank_df["CCR"]  # Challenge aversion
    rank_df["daa"] = 1.0 - rank_df["DAC"]  # Devil's advocate aversion
    rank_df["praise_first"] = 1.0 - rank_df["CAVEAT1"]  # Praise-first tendency
    rank_df["style"] = 0.5 * rank_df["FLAT"]  # Style factor (half-scaled flattery)
    
    # Calculate SI
    rank_df["SI"] = rank_df.apply(compute_sycophancy_index, axis=1)
    rank_df = rank_df.sort_values("SI", ascending=False).reset_index(drop=True)
    if save_prefix:
        if OUTPUT_FORMAT["si_table"] == "json":
            rank_df.to_json(f"{save_prefix}_sycophancy_scores.json", orient="records", indent=2)
        else:
            rank_df.to_csv(f"{save_prefix}_sycophancy_scores.csv", index=False)
        # Determine reasoning support dynamically for metadata snapshot
        try:
            annotated_cfgs = annotate_reasoning_support(model_configs, api_key)
            dynamic_reasoning_slugs = {c.get("slug") for c in annotated_cfgs if c.get("reasoning")}
        except Exception:
            dynamic_reasoning_slugs = set()
        # Save comprehensive metadata
        metadata = create_comprehensive_metadata(
            models_analyzed=names,
            knn_k=knn_k,
            leiden_resolution=leiden_resolution,
            bridge_threshold=bridge_threshold,
            model_configs=model_configs,
            scoring_config=SCORING_CONFIG,
            reasoning_model_slugs=dynamic_reasoning_slugs,
            prompts_df=prompts_df,
            sss_df=rank_df,
            modularity_Q=Q,
            conductance=cond,
            participation=part,
            save_prefix=save_prefix,
            backbone_edges_count=G_backbone.number_of_edges()
        )
        save_metadata(save_prefix, metadata)

    out = dict(
        names=names,
        S=S,
        D=D,
        pos=pos,
        backbone_edges=list(G_backbone.edges()),
        node_to_comm=node_to_comm,
        modularity_Q=Q,
        conductance=cond,
        participation=part,
        sss=sss_df,
        si_table=rank_df,
    )
    return out
