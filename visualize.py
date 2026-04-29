import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, works on macOS/Linux without a display
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def _stable_layout(nodes):
    K = nx.Graph()
    K.add_nodes_from(nodes)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            K.add_edge(u, v)
    return nx.spring_layout(K, seed=42)


def _to_p_edge(score: float, query_mode: str) -> float:
    return (1.0 - score) if query_mode == "no_edge" else score


def _draw_edges(
    G,
    pos,
    edgelist,
    colors,
    widths,
    *,
    dashed: bool,
    rad: float,
):
    """
    Force arrows via FancyArrowPatch (arrowstyle), and prevent clipping into nodes.
    """
    if not edgelist:
        return

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edgelist,
        edge_color=colors,
        width=widths,
        arrows=True,
        arrowstyle="-|>",      # <- strong arrowheads
        arrowsize=26,          # <- bigger arrowheads
        min_source_margin=10,  # <- keep arrow from starting inside node
        min_target_margin=18,  # <- keep arrowhead outside node
        connectionstyle=f"arc3,rad={rad}",
        style="dashed" if dashed else "solid",
    )


def _draw_full_edge_colored_graph(
    df: pd.DataFrame,
    dataset_name: str,
    score_col: str,
    out_path: str,
    threshold: float,
    query_mode: str,
    uncertain_band: float = 0.05,
):
    nodes = sorted(set(df["source"]).union(set(df["target"])))
    pos = _stable_layout(nodes)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    for _, r in df.iterrows():
        u, v = r["source"], r["target"]
        s = float(r[score_col])
        G.add_edge(u, v, score=s)

    # Draw nodes FIRST so edges/arrows appear on top (more visible)
    plt.figure(figsize=(16, 12))
    nx.draw_networkx_nodes(G, pos, node_size=1600, linewidths=1.5, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=10)

    lo = 0.5 - uncertain_band
    hi = 0.5 + uncertain_band

    confident_edges, confident_colors, confident_widths = [], [], []
    uncertain_edges, uncertain_colors, uncertain_widths = [], [], []
    edge_labels = {}

    for (u, v, data) in G.edges(data=True):
        s = float(data.get("score", 0.5))
        p_edge = _to_p_edge(s, query_mode)

        # blue=EDGE, red=NO EDGE
        color = "blue" if p_edge >= threshold else "red"

        # thickness based on distance from 0.5 of the STORED score
        width = 1.0 + 3.0 * min(1.0, abs(s - 0.5) * 2.0)

        edge_labels[(u, v)] = f"{s:.2f}"  # label shows stored score

        if lo <= s <= hi:
            uncertain_edges.append((u, v))
            uncertain_colors.append(color)
            uncertain_widths.append(max(3.5, width))
        else:
            confident_edges.append((u, v))
            confident_colors.append(color)
            confident_widths.append(width)

    # Split bidirectional edges so arrows don't overlap (different curvature)
    def _split_bidir(edgelist):
        es = set(edgelist)
        uni, bi = [], []
        for e in edgelist:
            (bi if (e[1], e[0]) in es else uni).append(e)
        return uni, bi

    # Confident edges (solid)
    uni, bi = _split_bidir(confident_edges)
    _draw_edges(G, pos, uni,
                [confident_colors[confident_edges.index(e)] for e in uni],
                [confident_widths[confident_edges.index(e)] for e in uni],
                dashed=False, rad=0.08)
    _draw_edges(G, pos, bi,
                [confident_colors[confident_edges.index(e)] for e in bi],
                [confident_widths[confident_edges.index(e)] for e in bi],
                dashed=False, rad=0.20)

    # Uncertain edges (dashed)
    uni, bi = _split_bidir(uncertain_edges)
    _draw_edges(G, pos, uni,
                [uncertain_colors[uncertain_edges.index(e)] for e in uni],
                [uncertain_widths[uncertain_edges.index(e)] for e in uni],
                dashed=True, rad=0.08)
    _draw_edges(G, pos, bi,
                [uncertain_colors[uncertain_edges.index(e)] for e in bi],
                [uncertain_widths[uncertain_edges.index(e)] for e in bi],
                dashed=True, rad=0.20)

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        rotate=False,
        label_pos=0.55,
        bbox=dict(alpha=0.0),
    )

    plt.title(
        f"{dataset_name} | {score_col} | mode={query_mode}\n"
        f"Blue=EDGE EXISTS, Red=NO EDGE (threshold={threshold}); label=stored score"
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


def draw_graphs(results_csv: str, dataset_name: str, viz_dir: str, threshold: float, query_mode: str, run_id: str = ""):
    os.makedirs(viz_dir, exist_ok=True)
    df = pd.read_csv(results_csv)
    run_id = run_id or (df["run_id"].iloc[0] if "run_id" in df.columns else "")

    nodes = sorted(set(df["source"]).union(set(df["target"])))
    pos = _stable_layout(nodes)

    # Ground truth (directed)
    gt = nx.DiGraph()
    gt.add_nodes_from(nodes)
    for _, r in df[df["ground_truth"] == True].iterrows():
        gt.add_edge(r["source"], r["target"])

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(gt, pos, node_size=1600, linewidths=1.5, edgecolors="black")
    nx.draw_networkx_labels(gt, pos, font_size=10)
    nx.draw_networkx_edges(
        gt, pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=26,
        min_source_margin=10,
        min_target_margin=18,
        edge_color="green",
        width=2.5,
        connectionstyle="arc3,rad=0.12",
    )
    plt.title(f"Ground Truth (Directed): {dataset_name}")
    gt_path = os.path.join(viz_dir, "ground_truth.png")
    plt.tight_layout()
    plt.savefig(gt_path, bbox_inches="tight", dpi=200)
    plt.close()

    exclude = {
        "run_id", "config_hash", "dataset_name", "source", "target",
        "ground_truth", "decision", "query_mode", "threshold",
        "ensemble_enabled", "max_tokens", "temperature", "models",
    }
    score_cols = [c for c in df.columns if c not in exclude]

    for col in sorted(score_cols):
        safe_col = col.replace("/", "_").replace("\\", "_")
        out_path = os.path.join(viz_dir, f"full__{safe_col}.png")
        _draw_full_edge_colored_graph(df, dataset_name, col, out_path, threshold, query_mode)