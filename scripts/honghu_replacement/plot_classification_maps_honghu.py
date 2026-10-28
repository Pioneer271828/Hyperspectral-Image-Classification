from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from honghu_common import FIGURES_DIR, RESULTS_DIR, ROOT, ensure_dirs, label_to_rgb, load_dataset, read_json, save_figure  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METHODS = [
    "Ground truth",
    "M3DCNN",
    "HybridSN",
    "SSRN",
    "GCN",
    "MambaHSI",
    "F2HGNN",
    "CEGCN",
    "AMGCFN",
    "DF2Net",
    "VCA-SAHGCN",
]


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 8,
            "savefig.dpi": 600,
        }
    )


def make_prepare_args() -> SimpleNamespace:
    return SimpleNamespace(
        train_ratio=0.01,
        val_ratio=0.005,
        train_num=30,
        val_num=10,
        samples_type="ratio",
        superpixel_scale=100,
        pavia_superpixel_scale=50,
    )


def resolve_checkpoint(metric: dict) -> Path:
    ckpt = metric.get("eval_checkpoint_path") or metric.get("checkpoint_paths", {}).get("primary")
    if not ckpt:
        raise ValueError("Metric JSON has no eval checkpoint path.")
    path = Path(ckpt)
    if not path.is_absolute():
        path = ROOT / path
    return path


def infer_vca_k32(device_name: str) -> Path | None:
    try:
        import torch

        from model import VCASAHGCN
        from train_vca_multidataset import prepare_dataset
    except Exception as exc:
        print(f"Skip VCA inference: {exc}")
        return None

    metric_path = ROOT / "experiments" / "honghu_vca_k32" / "metrics" / "vca_honghu_k32_seed2026.json"
    if not metric_path.exists():
        print("Skip VCA inference: HongHu K=32 metric JSON not found.")
        return None
    metric = read_json(metric_path)
    ckpt = resolve_checkpoint(metric)
    if not ckpt.exists():
        print(f"Skip VCA inference: checkpoint not found: {ckpt}")
        return None
    device = torch.device(device_name)
    cfg = metric["config"]
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

    item = prepare_dataset("whu-honghu", make_prepare_args(), device=device, seed=int(metric["seed"]))
    with torch.no_grad():
        y, _, _ = model(
            "whu-honghu",
            item["net_input"],
            item["Q"],
            item["A"],
            wavelengths=item["wavelengths"],
            fwhm=item["fwhm"],
        )
    pred = torch.argmax(y, dim=1).detach().cpu().numpy().reshape(item["height"], item["width"]) + 1
    _, gt = load_dataset("HongHu")
    pred = np.where(gt > 0, pred, 0).astype(np.int16)
    out_npy = RESULTS_DIR / "honghu_vca_k32_seed2026_full22_prediction.npy"
    out_png = FIGURES_DIR / "honghu_vca_k32_seed2026_full22_prediction.png"
    np.save(out_npy, pred)
    Image.fromarray(label_to_rgb(pred), mode="RGB").save(out_png)
    print(f"Wrote {out_npy}")
    print(f"Wrote {out_png}")
    return out_npy


def panel_image(label_map: np.ndarray | None, title: str) -> np.ndarray:
    if label_map is None:
        h, w = 940, 475
        img = np.full((h, w, 3), 245, dtype=np.uint8)
        yy, xx = np.ogrid[:h, :w]
        border = (xx < 2) | (xx > w - 3) | (yy < 2) | (yy > h - 3)
        img[border] = [170, 170, 170]
        return img
    return label_to_rgb(label_map)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-vca-inference", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    configure()
    _, gt = load_dataset("HongHu")
    pred_path = RESULTS_DIR / "honghu_vca_k32_seed2026_full22_prediction.npy"
    if not args.skip_vca_inference and not pred_path.exists():
        infer_vca_k32(args.device)
    pred = np.load(pred_path) if pred_path.exists() else None

    maps = {"Ground truth": gt, "VCA-SAHGCN": pred}
    fig, axes = plt.subplots(3, 4, figsize=(7.15, 6.25))
    axes = axes.flatten()
    for idx, method in enumerate(METHODS):
        ax = axes[idx]
        image = panel_image(maps.get(method), method)
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        label = chr(ord("a") + idx)
        title = f"({label}) {method}"
        if method == "VCA-SAHGCN" and pred is not None:
            title = f"({label}) VCA-SAHGCN (K=32 aux)"
        ax.set_title(title, pad=2)
        if maps.get(method) is None:
            ax.text(
                0.5,
                0.5,
                "missing",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="#404040",
            )
    for ax in axes[len(METHODS):]:
        ax.axis("off")
    fig.suptitle("Visual classification results on the WHU-Hi-HongHu dataset", fontsize=9.5, y=0.995)
    fig.tight_layout(pad=0.25, w_pad=0.2, h_pad=0.35)
    for path in save_figure(fig, "fig_honghu_classification_maps"):
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
