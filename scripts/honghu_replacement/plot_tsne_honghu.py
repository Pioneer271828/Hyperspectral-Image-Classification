from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from honghu_common import FIGURES_DIR, ROOT, ensure_dirs, label_to_rgb, load_dataset, read_json, save_figure  # noqa: E402

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
            "legend.fontsize": 6.5,
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
    path = Path(ckpt)
    if not path.is_absolute():
        path = ROOT / path
    return path


def sample_indices(gt: np.ndarray, per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    flat = gt.reshape(-1)
    indices = []
    labels = []
    for cls in sorted(int(v) for v in np.unique(flat) if int(v) > 0):
        cls_idx = np.flatnonzero(flat == cls)
        if cls_idx.size > per_class:
            cls_idx = rng.choice(cls_idx, size=per_class, replace=False)
        indices.append(cls_idx)
        labels.append(np.full(cls_idx.shape[0], cls, dtype=np.int16))
    return np.concatenate(indices), np.concatenate(labels)


def compute_features(device_name: str, per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import torch
        from sklearn.manifold import TSNE

        from model import VCASAHGCN
        from train_vca_multidataset import prepare_dataset
    except Exception as exc:
        print(f"Skip t-SNE: {exc}")
        return None
    metric_path = ROOT / "experiments" / "honghu_vca_k32" / "metrics" / "vca_honghu_k32_seed2026.json"
    if not metric_path.exists():
        print("Skip t-SNE: metric JSON not found.")
        return None
    metric = read_json(metric_path)
    ckpt = resolve_checkpoint(metric)
    if not ckpt.exists():
        print(f"Skip t-SNE: checkpoint not found: {ckpt}")
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
    item = prepare_dataset("whu-honghu", make_prepare_args(), device=device, seed=int(metric["seed"]))
    with torch.no_grad():
        fused, _, _ = model.backbone.forward_features(
            item["net_input"],
            item["Q"],
            item["A"],
            wavelengths=item["wavelengths"],
            fwhm=item["fwhm"],
            dataset_name="whu-honghu",
        )
    _, gt = load_dataset("HongHu")
    idx, labels = sample_indices(gt, per_class, seed)
    sampled = fused.detach().cpu().numpy()[idx]
    perplexity = min(30, max(5, sampled.shape[0] // 10))
    emb = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perplexity, random_state=seed).fit_transform(sampled)
    np.savez(FIGURES_DIR / "honghu_vca_k32_seed2026_tsne_features.npz", embedding=emb, labels=labels)
    return emb, labels


def missing_figure() -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.axis("off")
    ax.text(0.5, 0.58, "Missing HongHu t-SNE", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.39, "Feature embeddings could not be generated from a local checkpoint.", ha="center", va="center", fontsize=8)
    for path in save_figure(fig, "fig_tsne_honghu"):
        print(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    ensure_dirs()
    configure()
    npz_path = FIGURES_DIR / "honghu_vca_k32_seed2026_tsne_features.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        emb = data["embedding"]
        labels = data["labels"]
    else:
        computed = compute_features(args.device, args.per_class, args.seed)
        if computed is None:
            missing_figure()
            return
        emb, labels = computed

    colors = label_to_rgb(labels.reshape(1, -1))[0] / 255.0
    fig, ax = plt.subplots(figsize=(3.6, 3.45))
    ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=3.0, linewidths=0, alpha=0.82)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("WHU-Hi-HongHu feature t-SNE (K=32 auxiliary)", pad=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.55)
        spine.set_color("#202020")
    for path in save_figure(fig, "fig_tsne_honghu"):
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
