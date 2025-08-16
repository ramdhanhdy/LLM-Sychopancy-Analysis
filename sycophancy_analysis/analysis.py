# analysis.py
import warnings
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import networkx as nx

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

def robust_scale_matrix(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    med = np.median(X, axis=0, keepdims=True)
    q1 = np.quantile(X, 0.25, axis=0, keepdims=True)
    q3 = np.quantile(X, 0.75, axis=0, keepdims=True)
    iqr = np.maximum(q3 - q1, eps)
    return (X - med) / iqr


def similarity_from_vectors(V: Dict[str, List[float]]) -> Tuple[List[str], np.ndarray]:
    names = sorted(V.keys())
    X = np.array([V[n] for n in names], dtype=float)
    Xs = robust_scale_matrix(X)
    norms = np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-9
    Xn = Xs / norms
    S = Xn @ Xn.T
    S = np.clip((S + 1.0) / 2.0, 0.0, 1.0)  # map [-1,1] -> [0,1]
    return names, S


def _symmetrize_clip(S: np.ndarray) -> np.ndarray:
    S = np.array(S, dtype=float)
    S = 0.5 * (S + S.T)
    return np.clip(S, 0.0, 1.0)


def _dist_from_sim(S: np.ndarray) -> np.ndarray:
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return D


def umap_layout(
    names: List[str], D: np.ndarray, n_neighbors: int = 25, min_dist: float = 0.08, seed: int = 42
) -> Dict[str, Tuple[float, float]]:
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


def mst_backbone(names: List[str], D: np.ndarray) -> nx.Graph:
    Gd = nx.Graph()
    for n in names:
        Gd.add_node(n)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            Gd.add_edge(names[i], names[j], weight=float(D[i, j]))
    return nx.minimum_spanning_tree(Gd, weight="weight")


def knn_graph_from_similarity(
    names: List[str], S: np.ndarray, k: int = 8, sym_mode: str = "max"
) -> nx.Graph:
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


def detect_communities(
    G: nx.Graph, method: str = "auto", resolution: float = 1.0, seed: int = 42
) -> Tuple[Dict[str, int], str]:
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
    comm_map: Dict[int, set] = {}
    for n, c in node_to_comm.items():
        comm_map.setdefault(c, set()).add(n)
    communities = list(comm_map.values())
    if G.number_of_edges() == 0 or len(communities) <= 1:
        return 0.0
    return float(nx.algorithms.community.modularity(G, communities, weight="weight"))


def conductance_per_community(G: nx.Graph, node_to_comm: Dict[str, int]) -> Dict[int, float]:
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