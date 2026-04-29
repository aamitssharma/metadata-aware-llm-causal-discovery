from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "outputs" / "ExperimentResultsPaper1" / "04_evaluated_runs" / "asia" / "metaData" / "vanilla" / "metadata_noisy" / "asia_L1_names_only" / "run_1775000063"
OUT_DIR = ROOT / "outputs" / "ExperimentResultsPaper1" / "01_paper_assets" / "asia" / "sensitivity" / "two_model_heatmaps"

MODELS = {
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-oss-20b": "gpt-oss-20b",
}


def load_summary() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(RUN_DIR.glob("thr_0.7__a_*__b_*/evaluation_summary.csv")):
        frame = pd.read_csv(csv_path)
        frames.append(frame[frame["model_name"].isin(MODELS)])
    if not frames:
        raise FileNotFoundError(f"No evaluation_summary.csv files found under {RUN_DIR}")
    return pd.concat(frames, ignore_index=True)


def build_grid(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    grid = (
        df.loc[df["model_name"] == model_name, ["alpha", "beta", "f1_adj"]]
        .pivot(index="alpha", columns="beta", values="f1_adj")
        .sort_index()
    )
    grid = grid.reindex(index=sorted(grid.index), columns=sorted(grid.columns))
    return grid


def render_heatmap(grid: pd.DataFrame, model_name: str, vmin: float, vmax: float) -> None:
    label = MODELS[model_name]
    fig, ax = plt.subplots(figsize=(9.6, 7.3))

    heatmap = sns.heatmap(
        grid,
        ax=ax,
        cmap="YlGnBu",
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        linewidths=1.0,
        linecolor="white",
        square=True,
        cbar=True,
        cbar_kws={"label": "F1_adj", "pad": 0.04},
        annot_kws={"fontsize": 10},
    )

    cbar = heatmap.collections[0].colorbar
    cbar.set_label("F1_adj", rotation=90, labelpad=14, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    best_idx = grid.stack().idxmax()
    row = grid.index.get_loc(best_idx[0])
    col = grid.columns.get_loc(best_idx[1])
    ax.add_patch(
        patches.Rectangle(
            (col, row),
            1,
            1,
            fill=False,
            edgecolor="#d7261e",
            linewidth=2.5,
        )
    )

    ax.set_title(label, fontsize=16, pad=10)
    ax.set_xlabel("beta", fontsize=12)
    ax.set_ylabel("alpha", fontsize=12)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    fig.subplots_adjust(left=0.10, right=0.92, top=0.92, bottom=0.11)

    out_png = OUT_DIR / f"paper_asia_two_model_heatmap_l1_names_{label}.png"
    out_svg = OUT_DIR / f"paper_asia_two_model_heatmap_l1_names_{label}.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", context="talk")

    df = load_summary()
    grids = {model: build_grid(df, model) for model in MODELS}
    all_values = pd.concat([grid.stack() for grid in grids.values()], ignore_index=True)
    vmin = float(all_values.min())
    vmax = float(all_values.max())

    for model_name, grid in grids.items():
        render_heatmap(grid, model_name, vmin=vmin, vmax=vmax)
        print(OUT_DIR / f"paper_asia_two_model_heatmap_l1_names_{MODELS[model_name]}.png")
        print(OUT_DIR / f"paper_asia_two_model_heatmap_l1_names_{MODELS[model_name]}.svg")


if __name__ == "__main__":
    main()
