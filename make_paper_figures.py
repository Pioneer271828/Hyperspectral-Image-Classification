import argparse
import json
import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from loadData import data_read, split_data


ROOT = Path(__file__).resolve().parent
DEFAULT_PACKAGE = ROOT / "experiments" / "core_ablation_package_20260604-231410"
DEFAULT_VIZ_DIR = ROOT / "paper_vca_visualizations"
DEFAULT_OUT_DIR = ROOT / "paper_scientific_figures"

DATASET_ORDER = ["paviaU", "xuzhou", "WHUL"]
DATASET_LABELS = {
    "paviaU": "PaviaU",
    "xuzhou": "Xuzhou",
    "WHUL": "WHU-Hi-LongKou",
}

EXPERIMENT_LABELS = {
    "sahgcn_single_dataset": "Single-dataset",
    "vca_batchnorm_k64": "BatchNorm",
    "vca_k128_domainbn": "K=128",
    "vca_k32_domainbn": "K=32",
    "vca_no_wave_attention_k64": "Index attention",
    "vca_shared_domainbn_k64": "Wavelength attention",
    "vca_uniform_attention_k64": "Uniform attention",
}

MACRO_ORDER = [
    "vca_shared_domainbn_k64",
    "vca_no_wave_attention_k64",
    "vca_uniform_attention_k64",
    "vca_k32_domainbn",
    "vca_k128_domainbn",
    "vca_batchnorm_k64",
    "sahgcn_single_dataset",
]

SCI_COLORS = [
    "#2F6B9A",
    "#D88436",
    "#55A868",
    "#7B6DB5",
    "#C84D62",
    "#D1A33D",
    "#5B8C3A",
    "#6AAFCB",
]

CLASS_PALETTE = np.array(
    [
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [170, 110, 40],
    ],
    dtype=np.uint8,
)


class FigureWriter:
    def __init__(self, out_dir, formats, dpi):
        self.out_dir = Path(out_dir)
        self.formats = [fmt.strip().lower().lstrip(".") for fmt in formats if fmt.strip()]
        self.dpi = int(dpi)
        self.records = []
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, fig, stem, sources, size_in, note=""):
        paths = []
        for fmt in self.formats:
            suffix = "tiff" if fmt in {"tif", "tiff"} else fmt
            path = self.out_dir / f"{stem}.{suffix}"
            kwargs = {
                "dpi": self.dpi,
                "bbox_inches": "tight",
                "pad_inches": 0.035,
                "facecolor": "white",
            }
            if suffix in {"tif", "tiff"}:
                kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            fig.savefig(path, format=suffix, **kwargs)
            paths.append(str(path))
        self.records.append(
            {
                "figure": stem,
                "exports": paths,
                "sources": [str(Path(src)) for src in sources],
                "size_inches": list(size_in),
                "dpi": self.dpi,
                "formats": self.formats,
                "note": note,
            }
        )

    def write_manifest(self):
        path = self.out_dir / "figure_manifest.json"
        path.write_text(json.dumps(self.records, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.65,
            "axes.edgecolor": "#202020",
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "lines.linewidth": 1.25,
            "lines.markersize": 3.8,
            "patch.linewidth": 0.55,
            "grid.color": "#B8C0CC",
            "grid.linewidth": 0.35,
            "grid.alpha": 0.55,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.constrained_layout.use": False,
        }
    )


def chart_path(package, name):
    return Path(package) / "chart_data" / name


def as_percent(values):
    return np.asarray(values, dtype=float) * 100.0


def label_for_dataset(name):
    return DATASET_LABELS.get(name, name)


def label_for_exp(name):
    return EXPERIMENT_LABELS.get(name, name)


def order_categorical(df, column, order):
    lookup = {name: idx for idx, name in enumerate(order)}
    return df.assign(_order=df[column].map(lambda value: lookup.get(value, len(lookup)))).sort_values("_order")


def add_panel_label(ax, label, x=0.0, y=1.02):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )


def clean_axis(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    ax.tick_params(width=0.55, length=2.6, color="#202020")


def grouped_bar(
    ax,
    data,
    x_column,
    group_column,
    value_column,
    error_column=None,
    x_order=None,
    group_order=None,
    colors=None,
    width=0.22,
    y_label="OA (%)",
    ylim=None,
):
    if x_order is None:
        x_order = list(pd.unique(data[x_column]))
    if group_order is None:
        group_order = list(pd.unique(data[group_column]))
    colors = colors or SCI_COLORS
    x = np.arange(len(x_order), dtype=float)
    offsets = (np.arange(len(group_order)) - (len(group_order) - 1) / 2.0) * width
    for idx, group in enumerate(group_order):
        rows = data[data[group_column] == group].set_index(x_column)
        vals = []
        errs = []
        for key in x_order:
            if key in rows.index:
                vals.append(float(rows.loc[key, value_column]) * 100.0)
                if error_column and error_column in rows:
                    errs.append(float(rows.loc[key, error_column]) * 100.0)
                else:
                    errs.append(0.0)
            else:
                vals.append(np.nan)
                errs.append(0.0)
        ax.bar(
            x + offsets[idx],
            vals,
            width=width * 0.88,
            yerr=errs if any(errs) else None,
            color=colors[idx % len(colors)],
            edgecolor="#202020",
            linewidth=0.35,
            error_kw={"elinewidth": 0.55, "capsize": 1.7, "capthick": 0.55, "ecolor": "#303030"},
            label=label_for_exp(group),
        )
    ax.set_xticks(x, [label_for_dataset(key) for key in x_order])
    ax.set_ylabel(y_label)
    if ylim is not None:
        ax.set_ylim(*ylim)
    clean_axis(ax)


def label_to_rgb(label):
    label = np.asarray(label, dtype=np.int64)
    palette = CLASS_PALETTE
    if label.max(initial=0) >= len(palette):
        rng = np.random.default_rng(2026)
        extra = rng.integers(25, 235, size=(int(label.max()) + 1 - len(palette), 3), dtype=np.uint8)
        palette = np.vstack([palette, extra])
    return palette[np.clip(label, 0, len(palette) - 1)] / 255.0


def percentile_rgb(cube, dataset_name):
    cube = np.asarray(cube)
    bands = cube.shape[-1]
    try:
        wavelengths, _ = data_read.readWavelengths(dataset_name, bands)
        targets = [650.0, 550.0, 480.0]
        indices = [int(np.argmin(np.abs(wavelengths - target))) for target in targets]
    except Exception:
        indices = [int(bands * 0.72), int(bands * 0.5), int(bands * 0.28)]
    rgb = cube[:, :, indices].astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        channel = rgb[:, :, c]
        lo, hi = np.nanpercentile(channel, [2, 98])
        out[:, :, c] = np.clip((channel - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def class_counts(label):
    label = np.asarray(label, dtype=np.int64)
    classes = [int(v) for v in np.unique(label) if int(v) > 0]
    return pd.DataFrame({"class_id": classes, "count": [int(np.sum(label == cls)) for cls in classes]})


def split_counts(label, seed, train_ratio, val_ratio, train_num, val_num, samples_type):
    label = np.asarray(label, dtype=np.int64)
    flat = label.reshape(-1)
    class_num = int(flat.max())
    np.random.seed(seed)
    train_idx, val_idx, test_idx = split_data.split_data(
        flat.copy(),
        class_num,
        train_ratio,
        val_ratio,
        train_num,
        val_num,
        samples_type,
    )
    rows = []
    for split_name, indices in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        labs = flat[indices]
        for cls in range(1, class_num + 1):
            rows.append({"split": split_name, "class_id": cls, "count": int(np.sum(labs == cls))})
    return pd.DataFrame(rows)


def plot_dataset_overview(writer, args, datasets):
    size = (7.15, 6.6)
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(
        len(datasets),
        3,
        width_ratios=[1.0, 1.0, 1.25],
        wspace=0.08,
        hspace=0.22,
    )
    sources = []
    for row, name in enumerate(datasets):
        cube, gt = data_read.readData(name)
        gt = np.asarray(gt)
        sources.extend([f"dataset/{name}", "loadData/data_read.py"])
        ax_rgb = fig.add_subplot(gs[row, 0])
        ax_gt = fig.add_subplot(gs[row, 1])
        ax_bar = fig.add_subplot(gs[row, 2])
        ax_rgb.imshow(percentile_rgb(cube, name))
        ax_gt.imshow(label_to_rgb(gt), interpolation="nearest")
        for ax in [ax_rgb, ax_gt]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.45)
                spine.set_color("#303030")
        if row == 0:
            ax_rgb.set_title("(a) False-color image", pad=3)
            ax_gt.set_title("(b) Ground truth", pad=3)
            ax_bar.set_title("(c) Class pixels", pad=3)
        ax_rgb.set_ylabel(label_for_dataset(name), fontsize=8.2, fontweight="bold")

        counts = class_counts(gt)
        colors = label_to_rgb(counts["class_id"].to_numpy().reshape(-1, 1))[:, 0, :]
        ax_bar.bar(
            counts["class_id"],
            counts["count"],
            color=colors,
            edgecolor="#202020",
            linewidth=0.35,
            width=0.72,
        )
        ax_bar.set_xlim(0.3, counts["class_id"].max() + 0.7)
        ax_bar.set_xticks(counts["class_id"])
        ax_bar.set_ylabel("Pixels")
        if row == len(datasets) - 1:
            ax_bar.set_xlabel("Class ID")
        clean_axis(ax_bar)
    writer.save(
        fig,
        "fig01_dataset_overview",
        sorted(set(sources)),
        size,
        "Original data cubes, original ground-truth labels, and full-label class counts.",
    )
    plt.close(fig)


def plot_split_distribution(writer, args, datasets):
    size = (7.15, 3.15)
    fig, axes = plt.subplots(1, len(datasets), figsize=size, sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    sources = ["loadData/split_data.py", "loadData/data_read.py"]
    split_order = ["Train", "Val", "Test"]
    split_colors = {"Train": SCI_COLORS[2], "Val": SCI_COLORS[1], "Test": "#C8CFD8"}
    for ax, name in zip(axes, datasets):
        _, gt = data_read.readData(name)
        counts = split_counts(
            gt,
            args.seed,
            args.train_ratio,
            args.val_ratio,
            args.train_num,
            args.val_num,
            args.samples_type,
        )
        pivot = counts.pivot(index="class_id", columns="split", values="count").reindex(columns=split_order)
        left = np.zeros(len(pivot), dtype=float)
        y = np.arange(len(pivot))
        for split_name in split_order:
            vals = pivot[split_name].to_numpy(dtype=float)
            ax.barh(
                y,
                vals,
                left=left,
                color=split_colors[split_name],
                edgecolor="#202020",
                linewidth=0.25,
                label=split_name,
            )
            left += vals
        ax.set_title(label_for_dataset(name), pad=3)
        ax.set_xlabel("Pixels")
        ax.set_yticks(y, [str(cls) for cls in pivot.index])
        clean_axis(ax, grid_axis="x")
        sources.append(f"dataset/{name}")
    axes[0].set_ylabel("Class ID")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    writer.save(
        fig,
        "fig02_train_val_test_split",
        sorted(set(sources)),
        size,
        "Class-wise train/validation/test split generated with the same split function and default seed.",
    )
    plt.close(fig)


def plot_main_metrics(writer, package, datasets):
    path = chart_path(package, "chart_all_core_oa_long.csv")
    raw = pd.read_csv(path)
    base = raw[raw["experiment"] == "vca_shared_domainbn_k64"]
    summary = (
        base.groupby("dataset")[["oa", "aa", "kappa"]]
        .agg(["mean", "std"])
        .reindex(datasets)
    )
    size = (7.15, 2.55)
    fig, ax = plt.subplots(figsize=size)
    metrics = [("oa", "OA"), ("aa", "AA"), ("kappa", "Kappa")]
    x = np.arange(len(datasets))
    width = 0.22
    for idx, (metric, label) in enumerate(metrics):
        means = summary[(metric, "mean")].to_numpy(dtype=float) * 100.0
        errs = summary[(metric, "std")].fillna(0).to_numpy(dtype=float) * 100.0
        ax.bar(
            x + (idx - 1) * width,
            means,
            width=width * 0.88,
            yerr=errs,
            color=SCI_COLORS[idx],
            edgecolor="#202020",
            linewidth=0.35,
            error_kw={"elinewidth": 0.55, "capsize": 1.8, "capthick": 0.55, "ecolor": "#303030"},
            label=label,
        )
    ax.set_xticks(x, [label_for_dataset(key) for key in datasets])
    ax.set_ylim(70, 101.5)
    ax.set_ylabel("Metric (%)")
    clean_axis(ax)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.09))
    writer.save(fig, "fig03_main_metrics", [path], size, "Mean and standard deviation across completed seeds.")
    plt.close(fig)


def plot_seed_curve(writer, package, datasets):
    path = chart_path(package, "chart_all_core_oa_long.csv")
    raw = pd.read_csv(path)
    base = raw[raw["experiment"] == "vca_shared_domainbn_k64"].copy()
    size = (7.15, 2.55)
    fig, ax = plt.subplots(figsize=size)
    for idx, dataset in enumerate(datasets):
        rows = base[base["dataset"] == dataset].sort_values("seed")
        ax.plot(
            rows["seed"],
            rows["oa"] * 100.0,
            marker="o",
            color=SCI_COLORS[idx],
            label=label_for_dataset(dataset),
        )
    ax.set_xlabel("Random seed")
    ax.set_ylabel("OA (%)")
    ax.set_xticks(sorted(base["seed"].unique()))
    ax.set_ylim(max(70, base["oa"].min() * 100.0 - 4), 101.0)
    clean_axis(ax)
    ax.legend(loc="upper center", ncol=len(datasets), bbox_to_anchor=(0.5, 1.09))
    writer.save(fig, "fig04_seed_oa_curve", [path], size, "Baseline/control OA values by random seed.")
    plt.close(fig)


def plot_macro_summary(writer, package):
    path = chart_path(package, "chart_macro_oa_summary.csv")
    df = order_categorical(pd.read_csv(path), "experiment", MACRO_ORDER)
    size = (7.15, 3.2)
    fig, ax = plt.subplots(figsize=size)
    y = np.arange(len(df))
    colors = [SCI_COLORS[idx % len(SCI_COLORS)] for idx in range(len(df))]
    bars = ax.barh(
        y,
        df["macro_oa_mean"] * 100.0,
        xerr=df["macro_oa_std"].fillna(0) * 100.0,
        color=colors,
        edgecolor="#202020",
        linewidth=0.35,
        error_kw={"elinewidth": 0.55, "capsize": 1.7, "capthick": 0.55, "ecolor": "#303030"},
    )
    for bar, (_, row) in zip(bars, df.iterrows()):
        if int(row["n"]) < 3:
            bar.set_hatch("////")
        ax.text(
            row["macro_oa_mean"] * 100.0 + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"n={int(row['n'])}",
            va="center",
            ha="left",
            fontsize=6.8,
        )
    ax.set_yticks(y, [label_for_exp(name) for name in df["experiment"]])
    ax.invert_yaxis()
    ax.set_xlabel("Macro OA (%)")
    ax.set_xlim(0, 104)
    clean_axis(ax, grid_axis="x")
    ax.text(
        0.99,
        -0.18,
        "Hatched bar: incomplete seed set.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        color="#303030",
    )
    writer.save(fig, "fig05_ablation_macro_oa_summary", [path], size, "Macro-OA summary; incomplete K=128 seed set is hatched.")
    plt.close(fig)


def plot_attention_comparison(writer, package, datasets):
    path = chart_path(package, "chart_attention_comparison.csv")
    df = pd.read_csv(path)
    order = ["vca_shared_domainbn_k64", "vca_no_wave_attention_k64", "vca_uniform_attention_k64"]
    size = (7.15, 2.65)
    fig, ax = plt.subplots(figsize=size)
    grouped_bar(
        ax,
        df,
        "dataset",
        "experiment",
        "oa_mean",
        "oa_std",
        x_order=datasets,
        group_order=order,
        colors=[SCI_COLORS[0], SCI_COLORS[2], SCI_COLORS[1]],
        ylim=(40, 102),
    )
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1))
    writer.save(fig, "fig06_ablation_attention_modes", [path], size, "Wavelength, index, and uniform adapter attention comparison.")
    plt.close(fig)


def plot_latent_channels(writer, package, datasets):
    path = chart_path(package, "chart_latent_channels.csv")
    df = pd.read_csv(path)
    order = ["vca_k32_domainbn", "vca_shared_domainbn_k64", "vca_k128_domainbn"]
    size = (7.15, 2.65)
    fig, ax = plt.subplots(figsize=size)
    grouped_bar(
        ax,
        df,
        "dataset",
        "experiment",
        "oa_mean",
        "oa_std",
        x_order=datasets,
        group_order=order,
        colors=[SCI_COLORS[3], SCI_COLORS[0], SCI_COLORS[5]],
        ylim=(84, 101.5),
    )
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.1))
    ax.text(
        0.99,
        -0.2,
        "K=128 has one completed seed.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        color="#303030",
    )
    writer.save(fig, "fig07_ablation_latent_channels", [path], size, "Latent-channel ablation from chart_data.")
    plt.close(fig)


def plot_norm_comparison(writer, package, datasets):
    path = chart_path(package, "chart_norm_comparison.csv")
    df = pd.read_csv(path)
    order = ["vca_shared_domainbn_k64", "vca_batchnorm_k64"]
    size = (7.15, 2.65)
    fig, ax = plt.subplots(figsize=size)
    grouped_bar(
        ax,
        df,
        "dataset",
        "experiment",
        "oa_mean",
        "oa_std",
        x_order=datasets,
        group_order=order,
        colors=[SCI_COLORS[0], SCI_COLORS[4]],
        ylim=(0, 105),
        width=0.28,
    )
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1))
    writer.save(fig, "fig08_ablation_domainbn_vs_batchnorm", [path], size, "DomainBN versus BatchNorm ablation.")
    plt.close(fig)


def plot_single_vs_multi(writer, package, datasets):
    path = chart_path(package, "chart_single_vs_multi.csv")
    df = pd.read_csv(path)
    order = ["vca_shared_domainbn_k64", "sahgcn_single_dataset"]
    size = (7.15, 2.65)
    fig, ax = plt.subplots(figsize=size)
    grouped_bar(
        ax,
        df,
        "dataset",
        "experiment",
        "oa_mean",
        "oa_std",
        x_order=datasets,
        group_order=order,
        colors=[SCI_COLORS[0], SCI_COLORS[2]],
        ylim=(88, 101.5),
        width=0.28,
    )
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.1))
    writer.save(fig, "fig09_ablation_single_vs_multi", [path], size, "Single-dataset and multi-dataset training comparison.")
    plt.close(fig)


def plot_delta_heatmap(writer, package, datasets):
    path = chart_path(package, "chart_delta_vs_baseline.csv")
    df = pd.read_csv(path)
    exp_order = [
        "sahgcn_single_dataset",
        "vca_k32_domainbn",
        "vca_k128_domainbn",
        "vca_no_wave_attention_k64",
        "vca_uniform_attention_k64",
        "vca_batchnorm_k64",
    ]
    mat = (
        df[df["experiment"].isin(exp_order)]
        .pivot(index="experiment", columns="dataset", values="delta_oa_pct_points")
        .reindex(index=exp_order, columns=datasets)
    )
    size = (4.95, 3.2)
    fig, ax = plt.subplots(figsize=size)
    vmax = np.nanmax(np.abs(mat.to_numpy(dtype=float)))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(datasets)), [label_for_dataset(name) for name in datasets])
    ax.set_yticks(np.arange(len(exp_order)), [label_for_exp(name) for name in exp_order])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat.iloc[i, j]
            if pd.isna(value):
                continue
            text_color = "white" if abs(float(value)) > vmax * 0.5 else "#111111"
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=7, color=text_color)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("Delta OA (percentage points)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=6.8, length=2.4, width=0.5)
    writer.save(fig, "fig10_delta_vs_baseline_heatmap", [path], size, "OA change relative to wavelength-attention K=64 baseline.")
    plt.close(fig)


def trim_white_border(img, threshold=248, pad=6):
    arr = np.asarray(img.convert("RGB"))
    mask = np.any(arr < threshold, axis=2)
    if not np.any(mask):
        return img
    ys, xs = np.where(mask)
    left = max(int(xs.min()) - pad, 0)
    upper = max(int(ys.min()) - pad, 0)
    right = min(int(xs.max()) + pad + 1, img.width)
    lower = min(int(ys.max()) + pad + 1, img.height)
    return img.crop((left, upper, right, lower))


def read_image(path, crop_white=False):
    img = Image.open(path).convert("RGB")
    if crop_white:
        img = trim_white_border(img)
    return np.asarray(img)


def plot_classification_maps(writer, viz_dir, datasets):
    viz_dir = Path(viz_dir)
    sources = []
    available = []
    for name in datasets:
        gt_path = viz_dir / name / f"{name}_ground_truth.png"
        pred_path = viz_dir / name / f"{name}_prediction.png"
        if gt_path.exists() and pred_path.exists():
            available.append((name, gt_path, pred_path))
            sources.extend([gt_path, pred_path])
    if not available:
        return
    size = (7.15, 6.9)
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(len(available), 3, width_ratios=[1, 1, 0.22], hspace=0.06, wspace=0.04)
    for row, (name, gt_path, pred_path) in enumerate(available):
        ax_gt = fig.add_subplot(gs[row, 0])
        ax_pred = fig.add_subplot(gs[row, 1])
        for ax, path, title in [
            (ax_gt, gt_path, "Ground truth"),
            (ax_pred, pred_path, "Prediction"),
        ]:
            ax.imshow(read_image(path), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.45)
        if row == 0:
            ax_gt.set_title("Ground truth", pad=3)
            ax_pred.set_title("Prediction", pad=3)
        ax_gt.set_ylabel(label_for_dataset(name), fontsize=8.2, fontweight="bold")
        ax_leg = fig.add_subplot(gs[row, 2])
        ax_leg.set_xlim(0, 1)
        ax_leg.set_ylim(0, 10)
        ax_leg.axis("off")
        for cls in range(1, 10):
            color = CLASS_PALETTE[cls] / 255.0
            ax_leg.add_patch(plt.Rectangle((0.06, 9 - cls + 0.08), 0.25, 0.55, color=color, ec="#202020", lw=0.25))
            ax_leg.text(0.38, 9 - cls + 0.35, f"C{cls}", va="center", fontsize=6.8)
    writer.save(
        fig,
        "fig11_classification_maps",
        sources,
        size,
        "Prediction maps generated by generate_vca_visualizations.py; this script only repanels them.",
    )
    plt.close(fig)


def plot_visual_panel(writer, viz_dir, datasets, kind, stem, title_map):
    viz_dir = Path(viz_dir)
    suffixes = title_map["suffixes"]
    sources = []
    rows = []
    for name in datasets:
        paths = [viz_dir / name / f"{name}_{suffix}.png" for suffix in suffixes]
        if all(path.exists() for path in paths):
            rows.append((name, paths))
            sources.extend(paths)
    if not rows:
        return
    size = title_map.get("size", (7.15, 6.3))
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(len(rows), len(suffixes), hspace=0.16, wspace=0.08)
    for row, (name, paths) in enumerate(rows):
        for col, path in enumerate(paths):
            ax = fig.add_subplot(gs[row, col])
            img = Image.open(path).convert("RGB")
            top_crop = int(title_map.get("top_crop", 0))
            if top_crop > 0 and img.height > top_crop:
                img = img.crop((0, top_crop, img.width, img.height))
            img = trim_white_border(img)
            ax.imshow(np.asarray(img))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.45)
                spine.set_color("#303030")
            if row == 0:
                ax.set_title(title_map["titles"][col], pad=3)
            if col == 0:
                ax.set_ylabel(label_for_dataset(name), fontsize=8.2, fontweight="bold")
    writer.save(fig, stem, sources, size, f"{kind} panel assembled from existing project visualization outputs.")
    plt.close(fig)


def find_default_checkpoint(package):
    metric_json = (
        Path(package)
        / "raw_metrics_json"
        / "baseline_vca_domainbn_k64"
        / "vca_shared_domainbn_k64_seed2026.json"
    )
    if not metric_json.exists():
        return None, None
    data = json.loads(metric_json.read_text(encoding="utf-8"))
    candidates = [
        data.get("eval_checkpoint_path"),
        data.get("checkpoint_paths", {}).get("primary"),
        data.get("checkpoint_paths", {}).get("macro_oa"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate), metric_json
    return None, metric_json


def baseline_config(package):
    metric_json = (
        Path(package)
        / "raw_metrics_json"
        / "baseline_vca_domainbn_k64"
        / "vca_shared_domainbn_k64_seed2026.json"
    )
    defaults = {
        "latent_channels": 64,
        "adapter_embed_dim": 32,
        "adapter_attention_mode": "wavelength",
        "locality_sigma": 0.08,
        "locality_window": 0.16,
        "norm_type": "domain_bn",
        "wavelength_min": 400.0,
        "wavelength_max": 2500.0,
    }
    if not metric_json.exists():
        return defaults, None
    data = json.loads(metric_json.read_text(encoding="utf-8"))
    config = dict(defaults)
    config.update(data.get("config", {}))
    return config, metric_json


def compute_attention_arrays(package, datasets, checkpoint_override=None):
    try:
        import torch
        from model import VCASAHGCN
    except Exception as exc:
        print(f"Skip recomputed attention panels: {exc}")
        return None, []

    config, config_source = baseline_config(package)
    checkpoint = Path(checkpoint_override) if checkpoint_override else None
    checkpoint_source = None
    if checkpoint is None or not checkpoint.exists():
        checkpoint, checkpoint_source = find_default_checkpoint(package)
    if checkpoint is None or not checkpoint.exists():
        print("Skip recomputed attention panels: checkpoint not found.")
        return None, [src for src in [config_source, checkpoint_source] if src]

    class_counts = {}
    dataset_meta = {}
    for name in datasets:
        cube, gt = data_read.readData(name)
        bands = int(np.asarray(cube).shape[-1])
        wavelengths, fwhm = data_read.readWavelengths(name, bands)
        class_counts[name] = int(np.asarray(gt).max())
        dataset_meta[name] = {
            "bands": bands,
            "wavelengths": wavelengths,
            "fwhm": fwhm,
        }

    model = VCASAHGCN.MultiDatasetVCASAHGCN(
        class_counts,
        latent_channels=int(config["latent_channels"]),
        adapter_embed_dim=int(config["adapter_embed_dim"]),
        wavelength_min=float(config.get("wavelength_min", 400.0)),
        wavelength_max=float(config.get("wavelength_max", 2500.0)),
        locality_sigma=float(config.get("locality_sigma", 0.08)),
        locality_window=float(config.get("locality_window", 0.16)),
        adapter_attention_mode=str(config.get("adapter_attention_mode", "wavelength")),
        norm_type=str(config.get("norm_type", "domain_bn")),
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    arrays = {}
    with torch.no_grad():
        for name, meta in dataset_meta.items():
            bands = meta["bands"]
            dummy = torch.zeros((1, bands), dtype=torch.float32)
            wavelengths = torch.as_tensor(meta["wavelengths"], dtype=torch.float32).reshape(-1)
            fwhm = None if meta["fwhm"] is None else torch.as_tensor(meta["fwhm"], dtype=torch.float32).reshape(-1)
            _, info = model.backbone.adapter._adapt_flat(
                dummy,
                wavelengths=wavelengths,
                fwhm=fwhm,
                return_attention=True,
            )
            arrays[name] = {
                "attention": info["attention"].detach().cpu().numpy(),
                "positions": info["band_positions"].detach().cpu().numpy(),
            }

    sources = [checkpoint, "loadData/data_read.py"]
    for src in [config_source, checkpoint_source]:
        if src:
            sources.append(src)
    return arrays, sources


def plot_recomputed_attention(writer, args, package, datasets):
    arrays, sources = compute_attention_arrays(package, datasets, args.checkpoint)
    if not arrays:
        return False
    size = (7.15, 5.2)
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(len(datasets), 2, width_ratios=[1.05, 1.0], hspace=0.3, wspace=0.42)
    heat_axes = []
    for row, name in enumerate(datasets):
        if name not in arrays:
            continue
        attention = arrays[name]["attention"]
        positions = arrays[name]["positions"]
        ax_h = fig.add_subplot(gs[row, 0])
        ax_c = fig.add_subplot(gs[row, 1])
        im = ax_h.imshow(attention, aspect="auto", cmap="viridis", interpolation="nearest")
        heat_axes.append(ax_h)
        ax_h.set_ylabel("Latent anchor")
        if row == len(datasets) - 1:
            ax_h.set_xlabel("Band index")
        else:
            ax_h.set_xticklabels([])
        ax_h.set_title("(a) Anchor-to-band attention" if row == 0 else "", pad=3)
        ax_h.text(
            -0.18,
            0.5,
            label_for_dataset(name),
            transform=ax_h.transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=8.2,
            fontweight="bold",
        )
        for spine in ax_h.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.55)

        k = attention.shape[0]
        anchors = sorted(set([0, k // 8, k // 4, k // 2, 3 * k // 4, k - 1]))
        for idx, anchor in enumerate(anchors):
            ax_c.plot(
                positions,
                attention[anchor],
                color=SCI_COLORS[idx % len(SCI_COLORS)],
                label=f"anchor {anchor}",
            )
        ax_c.set_title("(b) Selected anchor curves" if row == 0 else "", pad=3)
        ax_c.set_ylabel("Attention")
        if row == len(datasets) - 1:
            ax_c.set_xlabel("Normalized wavelength / band position")
        else:
            ax_c.set_xticklabels([])
        clean_axis(ax_c)
        if row == 0:
            ax_c.legend(loc="upper left", ncol=2, fontsize=6.4, handlelength=1.6)
    if heat_axes:
        cbar = fig.colorbar(im, ax=heat_axes, fraction=0.026, pad=0.04)
        cbar.set_label("Weight", fontsize=7.3, labelpad=4)
        cbar.ax.tick_params(labelsize=6.8, length=2.4, width=0.5)
    writer.save(
        fig,
        "fig12_vca_attention_panels",
        sources,
        size,
        "VCA adapter attention recomputed from the baseline checkpoint and dataset band metadata.",
    )
    plt.close(fig)
    return True


def make_contact_sheet(out_dir):
    out_dir = Path(out_dir)
    pngs = sorted(path for path in out_dir.glob("fig*.png") if not path.name.endswith("_contact_sheet.png"))
    if not pngs:
        return None
    thumbs = []
    labels = []
    for path in pngs:
        img = Image.open(path).convert("RGB")
        img.thumbnail((540, 380), Image.Resampling.LANCZOS)
        thumbs.append(img.copy())
        labels.append(path.stem)
    label_h = 30
    pad = 24
    cols = 2
    rows = int(math.ceil(len(thumbs) / cols))
    cell_w = max(img.width for img in thumbs) + pad * 2
    cell_h = max(img.height for img in thumbs) + label_h + pad * 2
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    for idx, (img, label) in enumerate(zip(thumbs, labels)):
        row = idx // cols
        col = idx % cols
        x0 = col * cell_w + pad + (cell_w - pad * 2 - img.width) // 2
        y0 = row * cell_h + pad
        sheet.paste(img, (x0, y0))
        draw.text((col * cell_w + pad, y0 + img.height + 8), label, fill=(25, 35, 50), font=font)
    out_path = out_dir / "_paper_figures_contact_sheet.png"
    sheet.save(out_path)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate publication-style figures from local project data/results.")
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE), help="Core ablation package with chart_data CSV files.")
    parser.add_argument("--viz-dir", default=str(DEFAULT_VIZ_DIR), help="Existing visual outputs from generate_vca_visualizations.py.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for paper figures.")
    parser.add_argument("--datasets", default="paviaU,xuzhou,WHUL", help="Comma-separated dataset order.")
    parser.add_argument("--formats", default="png,pdf,tiff", help="Comma-separated export formats.")
    parser.add_argument("--dpi", type=int, default=600, help="Raster export DPI.")
    parser.add_argument("--checkpoint", default="", help="Optional baseline checkpoint for recomputing VCA attention panels.")
    parser.add_argument("--seed", type=int, default=2026, help="Seed for split-distribution visualization.")
    parser.add_argument("--samples-type", default="ratio", choices=["ratio", "number"])
    parser.add_argument("--train-ratio", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.005)
    parser.add_argument("--train-num", type=int, default=30)
    parser.add_argument("--val-num", type=int, default=10)
    parser.add_argument("--skip-data", action="store_true", help="Skip figures that read original HSI data cubes.")
    parser.add_argument("--skip-existing-viz", action="store_true", help="Skip panels assembled from existing PNG visualizations.")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_style()
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    formats = [fmt.strip() for fmt in args.formats.split(",") if fmt.strip()]
    writer = FigureWriter(args.out_dir, formats, args.dpi)
    package = Path(args.package)

    if not args.skip_data:
        plot_dataset_overview(writer, args, datasets)
        plot_split_distribution(writer, args, datasets)

    plot_main_metrics(writer, package, datasets)
    plot_seed_curve(writer, package, datasets)
    plot_macro_summary(writer, package)
    plot_attention_comparison(writer, package, datasets)
    plot_latent_channels(writer, package, datasets)
    plot_norm_comparison(writer, package, datasets)
    plot_single_vs_multi(writer, package, datasets)
    plot_delta_heatmap(writer, package, datasets)

    if not args.skip_existing_viz:
        plot_classification_maps(writer, args.viz_dir, datasets)
        attention_ok = plot_recomputed_attention(writer, args, package, datasets)
        if not attention_ok:
            plot_visual_panel(
                writer,
                args.viz_dir,
                datasets,
                "VCA attention",
                "fig12_vca_attention_panels",
                {
                    "suffixes": ["vca_attention_heatmap", "vca_anchor_attention_curves"],
                    "titles": ["Anchor-to-band heatmap", "Anchor curves"],
                    "size": (7.15, 6.0),
                },
            )
        plot_visual_panel(
            writer,
            args.viz_dir,
            datasets,
            "Feature embedding",
            "fig13_tsne_feature_panels",
            {
                "suffixes": ["tsne_features"],
                "titles": ["t-SNE feature distribution"],
                "size": (3.65, 6.2),
                "top_crop": 58,
            },
        )

    manifest = writer.write_manifest()
    contact_sheet = make_contact_sheet(args.out_dir)
    print(f"Wrote {len(writer.records)} figures to {Path(args.out_dir).resolve()}")
    print(f"Manifest: {manifest}")
    if contact_sheet:
        print(f"Contact sheet: {contact_sheet}")


if __name__ == "__main__":
    main()
