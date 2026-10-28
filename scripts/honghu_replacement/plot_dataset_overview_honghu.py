from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from honghu_common import (  # noqa: E402
    CLASS_NAMES,
    DISPLAY_DATASETS,
    class_distribution,
    ensure_dirs,
    false_color_rgb,
    label_to_rgb,
    load_dataset,
    save_figure,
)


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": 0.6,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "savefig.dpi": 600,
        }
    )


def main() -> None:
    ensure_dirs()
    configure()
    datasets = ["PaviaU", "Xuzhou", "HongHu"]
    fig, axes = plt.subplots(
        len(datasets),
        3,
        figsize=(7.15, 7.4),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.18], "wspace": 0.12, "hspace": 0.2},
    )
    for row_idx, dataset in enumerate(datasets):
        cube, gt = load_dataset(dataset)
        ax_rgb, ax_gt, ax_bar = axes[row_idx]

        ax_rgb.imshow(false_color_rgb(cube, dataset))
        ax_rgb.set_xticks([])
        ax_rgb.set_yticks([])
        ax_rgb.set_ylabel(DISPLAY_DATASETS[dataset], fontsize=8.5, fontweight="bold")
        if row_idx == 0:
            ax_rgb.set_title("False-color image", pad=3)

        ax_gt.imshow(label_to_rgb(gt))
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])
        if row_idx == 0:
            ax_gt.set_title("Ground truth", pad=3)

        rows = [r for r in class_distribution(gt) if int(r["Class_ID"]) > 0]
        x = np.array([int(r["Class_ID"]) for r in rows])
        y = np.array([int(r["Pixel_Count"]) for r in rows])
        colors = label_to_rgb(x.reshape(1, -1))[0] / 255.0
        ax_bar.bar(x, y, color=colors, edgecolor="#202020", linewidth=0.3)
        ax_bar.set_xlim(0.3, max(x) + 0.7)
        ax_bar.set_ylabel("Pixels")
        ax_bar.grid(axis="y", color="#B8C0CC", linewidth=0.35, alpha=0.6)
        ax_bar.set_axisbelow(True)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        if dataset == "HongHu":
            ax_bar.set_xticks(x)
            ax_bar.tick_params(axis="x", rotation=90)
        else:
            ax_bar.set_xticks(x)
        if row_idx == len(datasets) - 1:
            ax_bar.set_xlabel("Class ID")
        if row_idx == 0:
            ax_bar.set_title("Class-wise pixel distribution", pad=3)

    outputs = save_figure(fig, "fig01_dataset_overview_honghu")
    plt.close(fig)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
