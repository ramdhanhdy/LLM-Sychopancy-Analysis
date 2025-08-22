# visualization/network.py
"""Complete network analysis and visualization functions."""

import warnings
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from sklearn.neighbors import NearestNeighbors
from .utils import _palette

# Optional deps
try:
    import umap.umap_ as umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

try:
    import igraph as ig
    import leidenalg as la
    HAVE_LEIDEN = True
except Exception:
    HAVE_LEIDEN = False


# Similarity processing functions
def robust_scale_matrix(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Robust scaling using median and IQR."""
    med = np.median(X, axis=0, keepdims=True)
    q1 = np.quantile(X, 0.25, axis=0, keepdims=True)
    q3 = np.quantile(X, 0.75, axis=0, keepdims=True)
    iqr = np.maximum(q3 - q1, eps)
    return (X - med) / iqr


def similarity_from_vectors(V: Dict[str, List[float]]) -> Tuple[List[str], np.ndarray]:
    """Compute similarity matrix from vector dictionary."""
    names = sorted(V.keys())
    X = np.array([V[n] for n in names], dtype=float)
    Xs = robust_scale_matrix(X)
    norms = np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9
    Xn = Xs / norms
    S = Xn @ Xn.T
    S = np.clip((S + 1.0) / 2.0, 0.0, 1.0)  # map [-1,1] -> [0,1]
    return names, S


def _symmetrize_clip(S: np.ndarray) -> np.ndarray:
    """Symmetrize and clip similarity matrix to [0,1]."""
    S = np.array(S, dtype=float)
    S = 0.5 * (S + S.T)
    return np.clip(S, 0.0, 1.0)


def _dist_from_sim(S: np.ndarray) -> np.ndarray:
    """Convert similarity matrix to distance matrix."""
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return D


# Layout algorithms
def umap_layout(
    names: List[str], D: np.ndarray, n_neighbors: int = 25, min_dist: float = 0.08, seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """Compute UMAP layout from distance matrix."""
    if not HAVE_UMAP:
        warnings.warn("umap-learn not installed; falling back to spring.", RuntimeWarning)
        G = nx.from_numpy_array(1.0 - D)
        raw = nx.spring_layout(G, seed=seed, k=3.0, iterations=200, weight=None)
        return {names[i]: (float(raw[i][0]), float(raw[i][1])) for i in range(len(names))}
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="precomputed",
        random_state=seed,
        n_components=2,
        verbose=False,
    )
    Z = reducer.fit_transform(D)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    denom = np.max(np.abs(Zc)) + 1e-9
    Zs = Zc / denom
    return {names[i]: (float(Zs[i, 0]), float(Zs[i, 1])) for i in range(len(names))}


def spring_layout(
    names: List[str], S: np.ndarray, seed: int = 42, k: float = 1.5, iterations: int = 50
) -> Dict[str, Tuple[float, float]]:
    """Compute spring layout from similarity matrix."""
    G = nx.from_numpy_array(S)
    raw = nx.spring_layout(G, seed=seed, k=k, iterations=iterations)
    return {names[i]: (float(raw[i][0]), float(raw[i][1])) for i in range(len(names))}


# Graph construction functions
def knn_graph_from_similarity(
    names: List[str], S: np.ndarray, k: int = 8, sym_mode: str = "max"
) -> nx.Graph:
    """Build k-NN graph from similarity matrix."""
    n = len(names)
    Sc = np.array(S, dtype=float)
    np.fill_diagonal(Sc, -np.inf)
    idx = np.argsort(Sc, axis=1)[:, ::-1]
    knn = [set(row[: min(k, n - 1)]) for row in idx]
    G = nx.Graph()
    for nm in names:
        G.add_node(nm)
    for i in range(n):
        for j in knn[i]:
            if j == i:
                continue
            a, b = (i, j) if i < j else (j, i)
            w_ij, w_ji = float(Sc[i, j]), float(Sc[j, i])
            w = max(w_ij, w_ji) if sym_mode == "max" else 0.5 * (w_ij + w_ji)
            u, v = names[a], names[b]
            if G.has_edge(u, v):
                G[u][v]["weight"] = max(G[u][v]["weight"], w)
            else:
                G.add_edge(u, v, weight=w)
    return G


def mst_backbone(names: List[str], D: np.ndarray) -> nx.Graph:
    """Build minimum spanning tree backbone from distance matrix."""
    Gd = nx.Graph()
    for n in names:
        Gd.add_node(n)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            Gd.add_edge(names[i], names[j], weight=float(D[i, j]))
    return nx.minimum_spanning_tree(Gd, weight="weight")


# Community detection functions
def detect_communities(
    G: nx.Graph, method: str = "auto", resolution: float = 1.0, seed: int = 42
) -> Tuple[Dict[str, int], str]:
    """Detect communities in graph using Leiden or greedy modularity."""
    if method in ("auto", "leiden"):
        if HAVE_LEIDEN and G.number_of_edges() > 0:
            try:
                id_map = {u: i for i, u in enumerate(G.nodes())}
                inv = {i: u for u, i in id_map.items()}
                edges = [(id_map[u], id_map[v]) for u, v in G.edges()]
                weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
                igg = ig.Graph(n=len(id_map), edges=edges)
                igg.es["weight"] = weights
                part = la.find_partition(
                    igg,
                    la.RBConfigurationVertexPartition,
                    weights="weight",
                    resolution_parameter=float(resolution),
                    seed=int(seed),
                )
                node_to_comm = {}
                for cid, comp in enumerate(part):
                    for nid in comp:
                        node_to_comm[inv[int(nid)]] = cid
                return node_to_comm, "leiden"
            except Exception as e:
                warnings.warn(f"Leiden failed ({e}); falling back to greedy.", RuntimeWarning)
        elif method == "leiden":
            warnings.warn("leidenalg/igraph not available; using greedy.", RuntimeWarning)
    if G.number_of_edges() == 0:
        return ({u: i for i, u in enumerate(G.nodes())}, "greedy")
    comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    node_to_comm = {}
    for cid, comp in enumerate(comms):
        for u in comp:
            node_to_comm[u] = cid
    return node_to_comm, "greedy"


def weighted_modularity(G: nx.Graph, node_to_comm: Dict[str, int]) -> float:
    """Compute weighted modularity for community assignment."""
    comm_map: Dict[int, set] = {}
    for n, c in node_to_comm.items():
        comm_map.setdefault(c, set()).add(n)
    communities = list(comm_map.values())
    if G.number_of_edges() == 0 or len(communities) <= 1:
        return 0.0
    return float(nx.algorithms.community.modularity(G, communities, weight="weight"))


def conductance_per_community(G: nx.Graph, node_to_comm: Dict[str, int]) -> Dict[int, float]:
    """Compute conductance for each community."""
    deg_w = {u: 0.0 for u in G.nodes()}
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 1.0))
        deg_w[u] += w
        deg_w[v] += w
    total_vol = sum(deg_w.values())
    comm_nodes: Dict[int, set] = {}
    for n, c in node_to_comm.items():
        comm_nodes.setdefault(c, set()).add(n)
    cond: Dict[int, float] = {}
    for c, Sset in comm_nodes.items():
        if not Sset or len(Sset) == G.number_of_nodes():
            cond[c] = 0.0
            continue
        vol_S = sum(deg_w[u] for u in Sset)
        vol_not = total_vol - vol_S
        cut = 0.0
        for u, v, d in G.edges(data=True):
            if (u in Sset) ^ (v in Sset):
                cut += float(d.get("weight", 1.0))
        denom = max(min(vol_S, vol_not), 1e-9)
        cond[c] = float(cut / denom)
    return cond


def participation_coefficient(G: nx.Graph, node_to_comm: Dict[str, int]) -> Dict[str, float]:
    """Compute participation coefficient for each node."""
    comms = set(node_to_comm.values())
    P: Dict[str, float] = {}
    for u in G.nodes():
        k = 0.0
        per_c = {c: 0.0 for c in comms}
        for v, d in G[u].items():
            w = float(d.get("weight", 1.0))
            k += w
            c = node_to_comm.get(v, -1)
            if c != -1:
                per_c[c] += w
        if k <= 1e-12:
            P[u] = 0.0
        else:
            P[u] = 1.0 - sum((per_c[c] / k) ** 2 for c in comms)
    return P


# Main network analysis function
def analyze_network(
    names: List[str],
    S: np.ndarray,
    *,
    knn_k: int = 8,
    leiden_resolution: float = 1.0,
    layout_method: str = "spring",
    umap_neighbors: int = 25,
    umap_min_dist: float = 0.08,
    seed: int = 42
) -> Dict:
    """Complete network analysis pipeline."""
    # Build k-NN graph
    G_backbone = knn_graph_from_similarity(names, S, k=knn_k)
    
    # Compute layout
    if layout_method == "umap":
        D = _dist_from_sim(S)
        pos = umap_layout(names, D, n_neighbors=umap_neighbors, min_dist=umap_min_dist, seed=seed)
    else:
        pos = spring_layout(names, S, seed=seed)
    
    # Community detection
    node_to_comm, method_used = detect_communities(G_backbone, resolution=leiden_resolution, seed=seed)
    
    # Compute metrics
    Q = weighted_modularity(G_backbone, node_to_comm)
    conductance = conductance_per_community(G_backbone, node_to_comm)
    participation = participation_coefficient(G_backbone, node_to_comm)
    
    return {
        "graph": G_backbone,
        "layout": pos,
        "communities": node_to_comm,
        "modularity": Q,
        "conductance": conductance,
        "participation": participation,
        "method": method_used
    }


def plot_network(
    names: List[str],
    S: np.ndarray,
    pos: Dict[str, Tuple[float, float]],
    G_backbone: nx.Graph,
    node_to_comm: Dict[str, int],
    *,
    Q: float,
    conductance: Dict[int, float],
    participation: Dict[str, float],
    admixtures: Optional[List[Dict]] = None,
    title: str = "LLM Sychopantic Behavioral Network",
    bridge_threshold: float = 0.5,
) -> plt.Figure:
    """Create network visualization plot."""
    G = G_backbone.copy()
    # Edge thickness based on similarity
    ew = []
    for u, v in G.edges():
        i, j = names.index(u), names.index(v)
        ew.append(S[i, j])
    if len(ew):
        w_arr = np.array(ew)
        w_lo, w_hi = float(w_arr.min()), float(w_arr.max())
        w_norm = (w_arr - w_lo) / (w_hi - w_lo) if w_hi > w_lo else w_arr * 0 + 1
    else:
        w_norm = np.array([])

    bg, fg, accent = "#0A0E1A", "#E6F1FF", "#64FFDA"
    comm_ids = sorted(set(node_to_comm.values()))
    comm_colors = _palette(len(comm_ids))
    color_map = {nm: comm_colors[node_to_comm.get(nm, 0) % len(comm_colors)] for nm in names}
    node_colors = [color_map[nm] for nm in G.nodes()]
    node_sizes = [950 + 380 * G.degree[nm] for nm in G.nodes()]

    P = participation or {}
    bridges = [nm for nm in G.nodes() if P.get(nm, 0.0) >= bridge_threshold]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(20, 15), facecolor=bg)
    ax.set_facecolor(bg)

    # Vignette
    r = 200
    yy, xx = np.mgrid[-r:r, -r:r]
    dist = np.sqrt(xx**2 + yy**2)
    vign = np.clip((dist / dist.max()) ** 1.6, 0, 1)
    ax.imshow(
        vign,
        extent=[-1.2, 1.2, -1.1, 1.1],
        cmap=LinearSegmentedColormap.from_list(
            "vign",
            [(0.0, (10 / 255, 14 / 255, 26 / 255, 0.0)), (1.0, (10 / 255, 14 / 255, 26 / 255, 1.0))],
        ),
        interpolation="bicubic",
    )

    # Backbone edges
    if G.number_of_edges() > 0:
        segments, widths, alphas = [], [], []
        idx = 0
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            segments.append([(x0, y0), (x1, y1)])
            s = float(w_norm[idx]) if len(w_norm) else 0.7
            widths.append(1.0 + 4.0 * s)
            alphas.append(0.25 + 0.55 * s)
            idx += 1
        edge_colors = [(100 / 255.0, 255 / 255.0, 218 / 255.0, float(a)) for a in alphas]
        lc = LineCollection(segments, colors=edge_colors, linewidths=widths, capstyle="round", joinstyle="round")
        lc.set_zorder(1)
        ax.add_collection(lc)

    # Optional admixture edges (if you add a detector later)
    if admixtures:
        for e in admixtures:
            c = e["child"]
            for p in e["parents"]:
                x0, y0 = pos[c]
                x1, y1 = pos[p]
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    linestyle=(0, (4, 6)),
                    color=(100 / 255, 255 / 255, 218 / 255, 0.45),
                    linewidth=2.0,
                    zorder=1.1,
                )

    # Node glow
    glow_sizes = [s * 1.35 for s in node_sizes]
    glow_colors = [to_rgba(c, 0.18) for c in node_colors]
    glow_artist = nx.draw_networkx_nodes(G, pos, node_color=glow_colors, node_size=glow_sizes, linewidths=0, ax=ax)
    glow_artist.set_zorder(2)

    # Node core
    node_border = (1.0, 1.0, 1.0, 0.95)
    core_artist = nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, linewidths=2.5, edgecolors=[node_border] * G.number_of_nodes(), ax=ax
    )
    core_artist.set_zorder(3)
    try:
        core_artist.set_path_effects([pe.Stroke(linewidth=11, foreground=(1, 1, 1, 0.04)), pe.Normal()])
    except Exception:
        pass

    # Highlight bridges
    if bridges:
        node_size_map = dict(zip(list(G.nodes()), node_sizes))
        ring_sizes = [node_size_map[nm] * 1.12 for nm in bridges]
        ring = nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=bridges,
            node_color="none",
            node_size=ring_sizes,
            linewidths=2.2,
            edgecolors=accent,
            ax=ax,
        )
        ring.set_zorder(3.5)

    # Labels
    labels = {n: n for n in G.nodes()}
    txt = nx.draw_networkx_labels(G, pos, labels=labels, font_size=13, font_weight="bold", font_color=fg, ax=ax)
    for t in txt.values():
        t.set_zorder(4)
        t.set_path_effects([pe.withStroke(linewidth=4, foreground=(0, 0, 0, 0.45)), pe.Normal()])

    # Title
    ax.text(0.5, 1.05, title, transform=ax.transAxes, ha="center", va="top", fontsize=22, fontweight="bold", color=accent)

    # Stats
    cond_vals = list(conductance.values()) if conductance else []
    cond_med = np.median(cond_vals) if cond_vals else 0.0
    bridge_count = len(bridges)
    stats = (
        f"Nodes: {G.number_of_nodes()}  "
        f"Backbone edges: {G.number_of_edges()}  \n"
        f"Communities: {len(set(node_to_comm.values()))} "
        f"(Leiden on k-NN)\n"
        f"Modularity Q: {Q:.3f}  Median conductance: {cond_med:.3f}  \n"
        f"Bridges (P ≥ {bridge_threshold:.2f}): {bridge_count}"
    )
    ax.text(
        0.98,
        0.02,
        stats,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12.5,
        color=fg,
        bbox=dict(
            boxstyle="round,pad=0.55,rounding_size=0.6",
            facecolor="#151B2A",
            alpha=0.92,
            edgecolor=accent,
            linewidth=1.6,
        ),
    )

    # Legend
    legend_handles = []
    for idx in range(len(comm_ids)):
        h = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=comm_colors[idx % len(comm_colors)], markersize=10, linestyle="")
        legend_handles.append(h)
    if legend_handles:
        leg = ax.legend(
            legend_handles,
            [f"Community {i+1}" for i in range(len(legend_handles))],
            loc="upper right",
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            facecolor="#151B2A",
            edgecolor=accent,
            fontsize=11.5,
            title="Detected Communities",
        )
        leg.get_title().set_color(accent)
        leg.get_title().set_fontweight("bold")
        for text in leg.get_texts():
            text.set_color(fg)

    ax.axis("off")
    plt.tight_layout()
    return fig
