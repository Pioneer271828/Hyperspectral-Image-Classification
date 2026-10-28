from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from honghu_common import FIGURES_DIR, MAT_DATASETS, ROOT, ensure_dirs, read_json, save_figure  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def resolve_checkpoint(metric: dict) -> Path:
    ckpt = metric.get("eval_checkpoint_path") or metric.get("checkpoint_paths", {}).get("primary")
    path = Path(ckpt)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_attention(device_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import torch

        from model import VCASAHGCN
    except Exception as exc:
        print(f"Skip attention heatmap: {exc}")
        return None
    metric_path = ROOT / "experiments" / "honghu_vca_k32" / "metrics" / "vca_honghu_k32_seed2026.json"
    if not metric_path.exists():
        print("Skip attention heatmap: metric JSON not found.")
        return None
    metric = read_json(metric_path)
    ckpt = resolve_checkpoint(metric)
    if not ckpt.exists():
        print(f"Skip attention heatmap: checkpoint not found: {ckpt}")
        return None
    cfg = metric["config"]
    device = torch.device(device_name)
    model = VCASAHGCN.MultiDatasetVCASAHGCN(
        {"whu-honghu": 22},
        latent_channels=int(cfg["latent_channels"]),
        adapter_embed_dim=int(cfg["adapter_embed_dim"]),
        wavelength_min=float(cfg.get("wavelength_min", 400.0)),
        wavelength_max=float(cfg.get("wavelength_max", 2500.0)),
        locality_sigma=float(cfg.get("locality_sigma", 0.08)),
        locality_window=float(cfg.get("locality_window", 0.16)),
        adapter_attention_mode=str(cfg.get("adapter_attention_mode", "wavelength")),
        adapter_chunk_pixels=int(cfg.get("adapter_chunk_pixels", 0)),
        adapter_checkpoint_chunks=bool(cfg.get("adapter_checkpoint_chunks", True)),
        norm_type=str(cfg.get("norm_type", "domain_bn")),
    ).to(device)
    model.load_state_dict(torch.load(str(ckpt), map_location=device))
    model.eval()
    bands = 270
    start, end = MAT_DATASETS["HongHu"]["spectral_range_nm"]
    wavelengths = torch.linspace(float(start), float(end), steps=bands, device=device)
    dummy = torch.zeros((1, 1, bands), device=device)
    with torch.no_grad():
        _, info = model.backbone.adapter(dummy, wavelengths=wavelengths, fwhm=None, return_attention=True)
    attention = info["attention"].detach().cpu().numpy()
    positions = wavelengths.detach().cpu().numpy()
    np.save(FIGURES_DIR / "honghu_vca_k32_seed2026_attention.npy", attention)
    return attention, positions


def missing_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.8))
    ax.axis("off")
    ax.text(0.5, 0.58, "Missing HongHu attention arrays", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.39, "No usable local checkpoint was found for an attention heatmap.", ha="center", va="center", fontsize=8)
    for path in save_figure(fig, "fig_attention_heatmap_honghu"):
        print(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ensure_dirs()
    configure()
    loaded = load_attention(args.device)
    if loaded is None:
        missing_figure()
        return
    attention, positions = loaded
    anchors = sorted(set([0, attention.shape[0] // 8, attention.shape[0] // 4, attention.shape[0] // 2, 3 * attention.shape[0] // 4, attention.shape[0] - 1]))
    fig = plt.figure(figsize=(7.15, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.32)
    ax_h = fig.add_subplot(gs[0, 0])
    im = ax_h.imshow(attention, aspect="auto", cmap="viridis", interpolation="nearest")
    ax_h.set_title("Anchor-to-band attention")
    ax_h.set_xlabel("Band index")
    ax_h.set_ylabel("Latent anchor")
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=6.5)

    ax_c = fig.add_subplot(gs[0, 1])
    for anchor in anchors:
        ax_c.plot(positions, attention[anchor], linewidth=1.1, label=f"anchor {anchor}")
    ax_c.set_title("Selected anchor attention curves")
    ax_c.set_xlabel("Approx. wavelength (nm)")
    ax_c.set_ylabel("Attention")
    ax_c.grid(axis="y", color="#B8C0CC", linewidth=0.35, alpha=0.6)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    ax_c.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("WHU-Hi-HongHu VCA attention (K=32 auxiliary checkpoint)", fontsize=9.5, y=1.02)
    for path in save_figure(fig, "fig_attention_heatmap_honghu"):
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
