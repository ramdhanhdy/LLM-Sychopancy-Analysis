# visualization/metadata.py
"""Metadata generation functions for visualization."""

import json
from typing import List, Dict, Tuple, Optional
import numpy as np
import networkx as nx


def create_network_sidecar(
    save_prefix: str,
    names: List[str],
    pos: Dict[str, Tuple[float, float]],
    G_backbone: nx.Graph,
    node_to_comm: Dict[str, int],
    Q: float,
    conductance: Dict[int, float],
    participation: Dict[str, float],
    bridge_threshold: float = 0.5,
    leiden_resolution: float = 1.0,
    knn_k: int = 8,
) -> str:
    """Create network visualization metadata sidecar file."""
    bridges = [nm for nm in names if participation.get(nm, 0.0) >= bridge_threshold]
    
    # Community stats
    comm_stats = {}
    for comm_id in set(node_to_comm.values()):
        members = [nm for nm in names if node_to_comm.get(nm) == comm_id]
        comm_stats[comm_id] = {
            "size": len(members),
            "members": members,
            "conductance": conductance.get(comm_id, 0.0)
        }
    
    # Edge list with weights
    edges_data = []
    for u, v in G_backbone.edges():
        i, j = names.index(u), names.index(v)
        edges_data.append({
            "source": u,
            "target": v,
            "source_index": i,
            "target_index": j
        })
    
    metadata = {
        "schema_version": "1.0",
        "save_prefix": save_prefix,
        "graph_structure": {
            "nodes": len(names),
            "edges": G_backbone.number_of_edges(),
            "node_names": names,
            "edge_list": edges_data
        },
        "layout": {
            "algorithm": "spring_layout",
            "positions": {nm: {"x": float(pos[nm][0]), "y": float(pos[nm][1])} for nm in names}
        },
        "communities": {
            "algorithm": "leiden",
            "resolution": leiden_resolution,
            "count": len(set(node_to_comm.values())),
            "assignments": node_to_comm,
            "stats": comm_stats
        },
        "metrics": {
            "modularity_Q": Q,
            "median_conductance": float(np.median(list(conductance.values()))) if conductance else 0.0,
            "bridge_threshold": bridge_threshold,
            "bridge_count": len(bridges),
            "bridges": bridges
        },
        "participation_coefficients": participation,
        "render_params": {
            "knn_k": knn_k,
            "title": "LLM Sychopantic Behavioral Network",
            "color_scheme": "dark_background"
        }
    }
    
    sidecar_path = f"{save_prefix}_network.meta.json"
    with open(sidecar_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Network metadata saved to {sidecar_path}")
    return sidecar_path
