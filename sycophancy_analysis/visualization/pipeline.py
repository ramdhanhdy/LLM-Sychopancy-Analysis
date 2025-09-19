"""Main visualization pipeline function (aligned with persisted artifacts)."""

from typing import Dict, Optional
import matplotlib.pyplot as plt

from ..api import OUTPUT_FORMAT
from ..data import load_matrices, load_sss
from ..scoring.sycophancy_index import compute_sycophancy_index
from .network import (
    plot_network,
    knn_graph_from_similarity,
    mst_backbone,
    detect_communities,
    weighted_modularity,
    conductance_per_community,
    participation_coefficient,
    umap_layout,
)
from .heatmap import altair_heatmap
from .metadata import create_network_sidecar


def run_visualization(
    *,
    save_prefix: str,
    knn_k: int = 8,
    leiden_resolution: float = 1.0,
    bridge_threshold: float = 0.5,
    model_configs=None,
    api_key: Optional[str] = None,
) -> Dict:
    """Run the visualization stage using saved SSS and matrices under save_prefix.

    Expects the scoring stage to have produced matrices and SSS via the standard persistence
    functions. Outputs match the end-to-end pipeline: network image, heatmap, SI table, and
    metadata sidecar.
    """
    print(f"🎨 Starting visualization stage with save_prefix='{save_prefix}'")

    # Load matrices and names saved by prior stages
    names, S, D = load_matrices(save_prefix)
    if names is None or S is None or D is None:
        raise FileNotFoundError(
            "Similarity artifacts not found for this prefix. Ensure scoring stage completed for the same save_prefix."
        )

    # Build graph elements and metrics
    print("Building k-NN graph and metrics…")
    G_backbone = mst_backbone(names, D)
    G_knn = knn_graph_from_similarity(names, S, k=knn_k, sym_mode="max")
    node_to_comm, _ = detect_communities(G_knn, method="leiden", resolution=leiden_resolution, seed=42)
    Q = weighted_modularity(G_knn, node_to_comm)
    conductance = conductance_per_community(G_knn, node_to_comm)
    participation = participation_coefficient(G_knn, node_to_comm)

    # 2D layout (use UMAP if available; else spring fallback via umap_layout wrapper)
    pos = umap_layout(names, D)

    # Render network plot
    print("Rendering network plot…")
    fig = plot_network(
        names=names,
        S=S,
        pos=pos,
        G_backbone=G_backbone,
        node_to_comm=node_to_comm,
        Q=Q,
        conductance=conductance,
        participation=participation,
        bridge_threshold=bridge_threshold,
    )
    net_path = f"{save_prefix}_network.png"
    fig.savefig(net_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Network plot saved to {net_path}")

    # Heatmap
    print("Creating similarity heatmap…")
    heatmap_path = f"{save_prefix}_heatmap.html"
    try:
        chart = altair_heatmap(names, S)
        if chart is not None:
            chart.save(heatmap_path)
            print(f"Heatmap saved to {heatmap_path}")
        else:
            print("Altair not available; heatmap skipped")
    except Exception as e:
        print(f"Warning: Could not save heatmap to {heatmap_path}: {e}")

    # SI table from SSS
    print("Computing Sycophancy Index table…")
    sss_df = load_sss(save_prefix)
    if sss_df is None or sss_df.empty:
        raise FileNotFoundError("SSS scores not found. Run scoring stage first for this save_prefix.")

    rank_df = sss_df.copy()
    # Intermediate SI terms
    rank_df["ca"] = 1.0 - rank_df["CCR"]
    rank_df["daa"] = 1.0 - rank_df["DAC"]
    rank_df["praise_first"] = 1.0 - rank_df["CAVEAT1"]
    rank_df["style"] = 0.5 * rank_df["FLAT"]
    rank_df["SI"] = rank_df.apply(compute_sycophancy_index, axis=1)
    rank_df = rank_df.sort_values("SI", ascending=False).reset_index(drop=True)

    if OUTPUT_FORMAT.get("si_table") == "json":
        si_out = f"{save_prefix}_sycophancy_scores.json"
        rank_df.to_json(si_out, orient="records", indent=2)
    else:
        si_out = f"{save_prefix}_sycophancy_scores.csv"
        rank_df.to_csv(si_out, index=False)
    print(f"Sycophancy Index table saved to {si_out}")

    # Sidecar metadata for the network
    print("Creating network metadata…")
    create_network_sidecar(
        save_prefix=save_prefix,
        names=names,
        pos=pos,
        G_backbone=G_backbone,
        node_to_comm=node_to_comm,
        Q=Q,
        conductance=conductance,
        participation=participation,
        bridge_threshold=bridge_threshold,
        leiden_resolution=leiden_resolution,
        knn_k=knn_k,
    )

    print("✅ Visualization stage completed successfully")

    return {
        "names": names,
        "S": S,
        "D": D,
        "pos": pos,
        "backbone_edges": list(G_backbone.edges()),
        "node_to_comm": node_to_comm,
        "modularity_Q": Q,
        "conductance": conductance,
        "participation": participation,
        "sss": sss_df,
        "si_table": rank_df,
    }
