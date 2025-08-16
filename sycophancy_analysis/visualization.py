# visualization.py
import warnings
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba

# Optional deps
try:
    import altair as alt
    HAVE_ALTAIR = True
except Exception:
    HAVE_ALTAIR = False

def _palette(n: int) -> List[str]:
    base = [
        "#64FFDA",
        "#7C4DFF",
        "#FF6B6B",
        "#FFCA28",
        "#29B6F6",
        "#66BB6A",
        "#F06292",
        "#26A69A",
        "#AB47BC",
        "#FFA726",
    ]
    return [base[i % len(base)] for i in range(n)]

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
    title: str = "Sycophancy Stylometry • Backbone + Communities",
    bridge_threshold: float = 0.5,
) -> plt.Figure:
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


def altair_heatmap(names: List[str], S: np.ndarray, order: str = "spectral"):
    if not HAVE_ALTAIR:
        raise RuntimeError("Altair not installed.")
    alt.data_transformers.disable_max_rows()
    M = _symmetrize_clip(S) # Assuming _symmetrize_clip is defined here or imported
    if order == "spectral":
        # simple Laplacian spectral ordering
        n = M.shape[0]
        W = M.copy()
        np.fill_diagonal(W, 0.0)
        d = W.sum(axis=1)
        L = np.diag(d) - W
        vals, vecs = np.linalg.eigh(L)
        idx = np.argsort(vecs[:, 1]) if len(vals) >= 2 else np.arange(n)
    else:
        idx = np.arange(M.shape[0])
    M = M[idx][:, idx]
    ordered = [names[i] for i in idx]
    rows = []
    for i in range(len(ordered)):
        for j in range(len(ordered)):
            if j > i:
                continue
            rows.append(dict(source=ordered[i], target=ordered[j], value=float(M[i, j]), is_diag=i == j))
    df = pd.DataFrame(rows)
    hover = alt.selection_point(fields=["source", "target"], on="mouseover", nearest=True, empty="none")
    base = alt.Chart(df).properties(width={"step": 24}, height={"step": 24})
    rects = (
        base.mark_rect(stroke="#2A2F45", strokeWidth=0.6)
        .encode(
            x=alt.X("target:N", sort=ordered, axis=alt.Axis(labelColor="#E6F1FF", title="Models", titleColor="#E6F1FF")),
            y=alt.Y("source:N", sort=ordered, axis=alt.Axis(labelColor="#E6F1FF", title="Models", titleColor="#E6F1FF")),
            color=alt.Color(
                "value:Q",
                scale=alt.Scale(scheme="viridis", domain=[0, 1]),
                legend=alt.Legend(title="Similarity", titleColor="#64FFDA", labelColor="#FFFFFF"),
            ),
            opacity=alt.condition(hover, alt.value(1.0), alt.value(0.92)),
            tooltip=[
                alt.Tooltip("source:N", title="Row"),
                alt.Tooltip("target:N", title="Col"),
                alt.Tooltip("value:Q", title="Similarity", format=".3f"),
            ],
        )
        .add_params(hover)
    )
    diag = (
        base.transform_filter("datum.is_diag == true")
        .mark_rect(stroke="#64FFDA", strokeWidth=1, fillOpacity=0)
        .encode(x="target:N", y="source:N")
    )
    chart = alt.layer(rects, diag).properties(
        background="#0A0E1A",
        title=alt.TitleParams(
            text=["Stylometry Similarity • Spectral Ordering"],
            color="#64FFDA",
            fontSize=20,
            anchor="middle",
            fontWeight="bold",
        ),
    )
    chart = (
        chart.configure_axis(grid=False, domain=True, domainColor="#3A3F55", tickColor="#3A3F55")
        .configure_view(stroke=None)
        .configure_title(anchor="middle")
    )
    return chart