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


def main() -> None:
    ensure_dirs()
    configure()
    df = pd.read_csv(RESULTS_DIR / "main_k128_seed_results_honghu.csv")
    series = {
        "PaviaU": pd.to_numeric(df["PaviaU_OA"], errors="coerce") * 100.0,
        "Xuzhou": pd.to_numeric(df["Xuzhou_OA"], errors="coerce") * 100.0,
        "HongHu": pd.to_numeric(df["HongHu_OA"], errors="coerce") * 100.0,
        "Macro": pd.to_numeric(df["Macro_OA"], errors="coerce") * 100.0,
    }
    fig, ax = plt.subplots(figsize=(7.15, 2.75))
    if all(values.isna().all() for values in series.values()):
        ax.axis("off")
        ax.text(
            0.5,
            0.58,
            "Missing seed-stability data",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.39,
            "Seed 0, 2026, and 3407 results for the K=128 paviaU/xuzhou/whu-honghu run are not available.",
            ha="center",
            va="center",
            fontsize=8,
            transform=ax.transAxes,
        )
    else:
        seeds = df["seed"].astype(str).tolist()
        x = np.arange(len(seeds), dtype=float)
        colors = ["#2F6B9A", "#D88436", "#55A868", "#7B6DB5"]
        for idx, (name, values) in enumerate(series.items()):
            ax.plot(x, values.to_numpy(), marker="o", color=colors[idx], linewidth=1.2, label=name)
        ax.set_xticks(x, seeds)
        ax.set_xlabel("Random seed")
        ax.set_ylabel("OA (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="#B8C0CC", linewidth=0.35, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.16))
    for path in save_figure(fig, "fig_seed_stability_k128_honghu"):
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
