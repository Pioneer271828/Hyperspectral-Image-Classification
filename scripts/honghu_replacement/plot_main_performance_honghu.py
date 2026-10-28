from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from honghu_common import RESULTS_DIR, ensure_dirs, save_figure  # noqa: E402


COLORS = ["#2F6B9A", "#D88436", "#55A868"]


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": 0.65,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "savefig.dpi": 600,
        }
    )


def missing_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.6))
    ax.axis("off")
    ax.text(
        0.5,
        0.58,
        "Missing K=128 replacement results",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.39,
        "No exact VCA-SAHGCN runs were found for paviaU, xuzhou, and whu-honghu.\n"
        "Run the commands in missing_experiments_honghu.md, then rerun collect_honghu_results.py.",
        ha="center",
        va="center",
        fontsize=8,
        transform=ax.transAxes,
    )
    for path in save_figure(fig, "fig02_main_performance_k128_honghu"):
        print(path)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    configure()
    csv_path = RESULTS_DIR / "main_k128_summary_honghu.csv"
    df = pd.read_csv(csv_path)
    datasets = ["PaviaU", "Xuzhou", "WHU-Hi-HongHu"]
    metrics = ["OA", "AA", "Kappa"]
    values = []
    has_any = False
    for dataset in datasets:
        row = df[df["Dataset"] == dataset]
        for metric in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            mean = pd.to_numeric(row[mean_col], errors="coerce").iloc[0] if not row.empty else np.nan
            std = pd.to_numeric(row[std_col], errors="coerce").iloc[0] if not row.empty else np.nan
            values.append({"Dataset": dataset, "Metric": metric, "Mean": mean, "Std": std})
            if not np.isnan(mean):
                has_any = True
    if not has_any:
        missing_figure()
        return

    plot_df = pd.DataFrame(values)
    fig, ax = plt.subplots(figsize=(7.15, 2.75))
    x = np.arange(len(datasets), dtype=float)
    width = 0.22
    for idx, metric in enumerate(metrics):
        subset = plot_df[plot_df["Metric"] == metric].set_index("Dataset")
        means = np.array([subset.loc[d, "Mean"] if d in subset.index else np.nan for d in datasets], dtype=float) * 100.0
        stds = np.array([subset.loc[d, "Std"] if d in subset.index else np.nan for d in datasets], dtype=float) * 100.0
        ax.bar(
            x + (idx - 1) * width,
            means,
            width=width,
            color=COLORS[idx],
            edgecolor="#202020",
            linewidth=0.45,
            label=metric,
            yerr=stds,
            error_kw={"elinewidth": 0.6, "capsize": 2.0, "capthick": 0.6, "ecolor": "#202020"},
        )
    ax.set_xticks(x, datasets)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#B8C0CC", linewidth=0.35, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16))
    for path in save_figure(fig, "fig02_main_performance_k128_honghu"):
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
