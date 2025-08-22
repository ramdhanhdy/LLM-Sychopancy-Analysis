# visualization/pipeline.py
"""Main visualization pipeline function."""

import os
import pickle
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

from .network import plot_network
from .heatmap import altair_heatmap
from .metadata import create_network_sidecar
from .metrics import create_sycophancy_ranking


def run_visualization(
    *,
    save_prefix: str,
    knn_k: int = 8,
    leiden_resolution: float = 1.0,
    bridge_threshold: float = 0.5,
    model_configs = None,
    api_key: Optional[str] = None,
) -> Dict:
    """Run the visualization stage of the pipeline."""
    print(f"🎨 Starting visualization stage with save_prefix='{save_prefix}'")
    
    # Load persisted SSS scores
    sss_path = f"{save_prefix}_sss_scores.csv"
    if not os.path.exists(sss_path):
        raise FileNotFoundError(f"SSS scores not found at {sss_path}. Run scoring stage first.")
    
    sss_df = pd.read_csv(sss_path)
    print(f"Loaded SSS scores for {len(sss_df)} models from {sss_path}")
    
    # Load similarity matrix and vectors
    sim_path = f"{save_prefix}_similarity_matrix.pkl"
    vec_path = f"{save_prefix}_similarity_vectors.pkl"
    
    if not os.path.exists(sim_path) or not os.path.exists(vec_path):
        raise FileNotFoundError(f"Similarity data not found. Expected {sim_path} and {vec_path}")
    
    with open(sim_path, "rb") as f:
        S = pickle.load(f)
    with open(vec_path, "rb") as f:
        D = pickle.load(f)
    
    names = sss_df["model_name"].tolist()
    print(f"Loaded similarity matrix {S.shape} and vectors {D.shape}")
    
    # Build k-NN graph
    print(f"Building k-NN graph with k={knn_k}")
    nbrs = NearestNeighbors(n_neighbors=knn_k + 1, metric="precomputed")
    nbrs.fit(1 - S)  # Convert similarity to distance
    distances, indices = nbrs.kneighbors(1 - S)
    
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(names)
    
    for i in range(len(names)):
        for j in range(1, knn_k + 1):  # Skip self (index 0)
            neighbor_idx = indices[i, j]
            if neighbor_idx < len(names):
                G.add_edge(names[i], names[neighbor_idx])
    
    print(f"Created k-NN graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Extract backbone (keep all edges for now, could add filtering later)
    G_backbone = G.copy()
    
    # Compute layout
    print("Computing spring layout...")
    pos = nx.spring_layout(G_backbone, k=1.5, iterations=50, seed=42)
    
    # Community detection using Leiden
    print(f"Running Leiden community detection with resolution={leiden_resolution}")
    try:
        import leidenalg
        import igraph as ig
        
        # Convert to igraph
        ig_graph = ig.Graph.from_networkx(G_backbone)
        
        # Run Leiden algorithm
        partition = leidenalg.find_partition(ig_graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=leiden_resolution)
        
        # Map back to node names
        node_to_comm = {}
        for i, comm_id in enumerate(partition.membership):
            node_name = names[i] if i < len(names) else list(G_backbone.nodes())[i]
            node_to_comm[node_name] = comm_id
            
        Q = partition.modularity
        print(f"Leiden found {len(set(partition.membership))} communities with modularity Q={Q:.3f}")
        
    except ImportError:
        print("Warning: leidenalg not available, using simple connected components")
        components = list(nx.connected_components(G_backbone))
        node_to_comm = {}
        for comm_id, component in enumerate(components):
            for node in component:
                node_to_comm[node] = comm_id
        Q = nx.algorithms.community.modularity(G_backbone, components)
    
    # Compute conductance for each community
    conductance = {}
    communities = {}
    for node, comm_id in node_to_comm.items():
        if comm_id not in communities:
            communities[comm_id] = set()
        communities[comm_id].add(node)
    
    for comm_id, community in communities.items():
        if len(community) > 1:
            try:
                conductance[comm_id] = nx.algorithms.cuts.conductance(G_backbone, community)
            except:
                conductance[comm_id] = 0.0
        else:
            conductance[comm_id] = 0.0
    
    # Compute participation coefficients
    participation = {}
    for node in G_backbone.nodes():
        node_comm = node_to_comm.get(node, 0)
        degree = G_backbone.degree(node)
        if degree == 0:
            participation[node] = 0.0
            continue
            
        # Count connections to each community
        comm_degrees = {}
        for neighbor in G_backbone.neighbors(node):
            neighbor_comm = node_to_comm.get(neighbor, 0)
            comm_degrees[neighbor_comm] = comm_degrees.get(neighbor_comm, 0) + 1
        
        # Participation coefficient formula
        p_coeff = 1.0
        for comm_deg in comm_degrees.values():
            p_coeff -= (comm_deg / degree) ** 2
        participation[node] = p_coeff
    
    print(f"Computed conductance and participation coefficients")
    
    # Create network plot
    print("Creating network visualization...")
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
    
    # Save network plot
    network_path = f"{save_prefix}_network.png"
    fig.savefig(network_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Network plot saved to {network_path}")
    
    # Create heatmap
    print("Creating similarity heatmap...")
    heatmap_path = f"{save_prefix}_heatmap.html"
    altair_heatmap(names, S, heatmap_path)
    
    # Compute Sycophancy Index
    print("Computing Sycophancy Index...")
    rank_df = create_sycophancy_ranking(sss_df)
    
    # Save SI ranking
    si_path = f"{save_prefix}_sycophancy_index.csv"
    rank_df.to_csv(si_path, index=False)
    print(f"Sycophancy Index ranking saved to {si_path}")
    
    # Create network metadata sidecar
    print("Creating network metadata...")
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
    
    return dict(
        names=names,
        S=S,
        D=D,
        pos=pos,
        backbone_edges=list(G_backbone.edges()),
        node_to_comm=node_to_comm,
        modularity_Q=Q,
        conductance=conductance,
        participation=participation,
        sss=sss_df,
        si_table=rank_df,
    )
