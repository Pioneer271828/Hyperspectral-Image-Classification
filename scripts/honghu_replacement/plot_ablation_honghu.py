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


COLORS = ["#2F6B9A", "#D88436", "#55A868", "#7B6DB5"]
DATASETS = ["PaviaU", "Xuzhou", "WHU-Hi-HongHu"]


PANELS = [
    ("Attention mode", ["K=64 wavelength attention + DomainBN", "K=64 index attention", "K=64 uniform attention"]),
    ("Latent channels", ["K=32 wavelength attention lightweight configuration", "K=64 wavelength attention + DomainBN", "K=128 wavelength attention + DomainBN main model"]),
    ("Normalization", ["K=64 wavelength attention + DomainBN", "K=64 wavelength attention + standard BN"]),
    ("Single vs multi", ["single-dataset VCA-SAHGCN K=128: PaviaU", "single-dataset VCA-SAHGCN K=128: Xuzhou", "single-dataset VCA-SAHGCN K=128: WHU-Hi-HongHu", "K=128 wavelength attention + DomainBN main model"]),
    ("Band augmentation", ["K=64 wavelength attention + DomainBN", "BandDrop 0.1", "Band-level Mixup 0.1"]),
]


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": 0.65,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "savefig.dpi": 600,
        }
    )


def short_label(name: str) -> str:
    replacements = {
        "K=128 wavelength attention + DomainBN main model": "K=128",
        "K=64 wavelength attention + DomainBN": "Wave K=64",
        "K=64 wavelength attention + standard BN": "BN K=64",
        "K=32 wavelength attention lightweight configuration": "K=32",
        "K=64 index attention": "Index",
        "K=64 uniform attention": "Uniform",
        "Band-level Mixup 0.1": "Mixup 0.1",
        "single-dataset VCA-SAHGCN K=128: WHU-Hi-HongHu": "Single HongHu",
        "single-dataset VCA-SAHGCN K=128: PaviaU": "Single PaviaU",
        "single-dataset VCA-SAHGCN K=128: Xuzhou": "Single Xuzhou",
    }
    return replacements.get(name, name)


def draw_missing(ax, title: str) -> None:
    ax.axis("off")
    ax.set_title(title, pad=3)
    ax.text(0.5, 0.55, "missing", ha="center", va="center", fontsize=10, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.35, "No exact HongHu replacement run", ha="center", va="center", fontsize=7, transform=ax.transAxes)


def plot_panel(ax, df: pd.DataFrame, title: str, experiments: list[str]) -> None:
    subset = df[df["Experiment_Name"].isin(experiments)].copy()
    subset["OA_mean_num"] = pd.to_numeric(subset["OA_mean"], errors="coerce")
    subset["OA_std_num"] = pd.to_numeric(subset["OA_std"], errors="coerce")
    subset = subset.dropna(subset=["OA_mean_num"])
    if subset.empty:
        draw_missing(ax, title)
        return
    x_labels = [short_label(item) for item in experiments]
    x = np.arange(len(experiments), dtype=float)
    width = 0.22
    for idx, dataset in enumerate(DATASETS):
        rows = subset[subset["Dataset"] == dataset].set_index("Experiment_Name")
        vals = []
        errs = []
        for exp in experiments:
            if exp in rows.index:
                vals.append(float(rows.loc[exp, "OA_mean_num"]) * 100.0)
                errs.append(float(rows.loc[exp, "OA_std_num"]) * 100.0)
            else:
                vals.append(np.nan)
                errs.append(np.nan)
        ax.bar(
            x + (idx - 1) * width,
            vals,
            yerr=errs,
            width=width,
            color=COLORS[idx],
            edgecolor="#202020",
            linewidth=0.35,
            label=dataset if title == "Attention mode" else None,
            error_kw={"elinewidth": 0.45, "capsize": 1.7, "capthick": 0.45},
        )
    ax.set_title(title, pad=3)
    ax.set_xticks(x, x_labels, rotation=24, ha="right")
    ax.set_ylabel("OA (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", color="#B8C0CC", linewidth=0.35, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    ensure_dirs()
    configure()
    df = pd.read_csv(RESULTS_DIR / "ablation_summary_honghu.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 5.2))
    flat = axes.flatten()
    for ax, (title, experiments) in zip(flat, PANELS):
        plot_panel(ax, df, title, experiments)
    aux_ax = flat[-1]
    aux = df[df["Experiment_Name"] == "auxiliary: HongHu K=32 single-dataset VCA-SAHGCN"].copy()
    aux["OA_mean_num"] = pd.to_numeric(aux["OA_mean"], errors="coerce")
    aux["OA_std_num"] = pd.to_numeric(aux["OA_std"], errors="coerce")
    aux_ax.set_title("Available auxiliary", pad=3)
    if aux["OA_mean_num"].notna().any():
        mean = float(aux["OA_mean_num"].dropna().iloc[0]) * 100.0
        std = float(aux["OA_std_num"].fillna(0).iloc[0]) * 100.0
        aux_ax.bar([0], [mean], yerr=[std], color="#55A868", edgecolor="#202020", linewidth=0.4)
        aux_ax.set_xticks([0], ["HongHu K=32\nsingle"])
        aux_ax.set_ylim(0, 100)
        aux_ax.set_ylabel("OA (%)")
        aux_ax.grid(axis="y", color="#B8C0CC", linewidth=0.35, alpha=0.6)
    else:
        draw_missing(aux_ax, "Available auxiliary")
    for ax in flat:
        if ax.has_data() and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.26))
            break
    fig.tight_layout(w_pad=1.0, h_pad=1.15)
    for path in save_figure(fig, "fig_ablation_honghu"):
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
