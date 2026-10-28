import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from model import VCASAHGCN
from train_vca_multidataset import prepare_dataset
from model.checkpoint_io import load_model_checkpoint, preparation_args_from_config, dataset_seed
from model.interaction_config import add_interaction_arguments


PALETTE = np.array([
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
    [255, 250, 200],
    [128, 0, 0],
    [170, 255, 195],
    [128, 128, 0],
    [255, 215, 180],
    [0, 0, 128],
    [128, 128, 128],
], dtype=np.uint8)


def get_font(size=22, bold=False):
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def normalize_minmax(x):
    x = np.asarray(x, dtype=np.float32)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx <= mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def label_to_rgb(label):
    label = np.asarray(label, dtype=np.int64)
    palette = PALETTE
    if label.max(initial=0) >= len(palette):
        rng = np.random.default_rng(1234)
        extra = rng.integers(20, 245, size=(int(label.max()) + 1 - len(palette), 3), dtype=np.uint8)
        palette = np.vstack([palette, extra])
    return palette[np.clip(label, 0, len(palette) - 1)]


def save_label_png(label, path, scale=2):
    rgb = label_to_rgb(label)
    img = Image.fromarray(rgb, mode="RGB")
    if scale and scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
    img.save(path)


def make_panel(images, titles, path):
    font = get_font(24, bold=True)
    pad = 24
    title_h = 42
    max_h = max(im.height for im in images)
    total_w = sum(im.width for im in images) + pad * (len(images) + 1)
    canvas = Image.new("RGB", (total_w, max_h + title_h + pad * 2), "white")
    draw = ImageDraw.Draw(canvas)
    x = pad
    for im, title in zip(images, titles):
        text_w = draw.textbbox((0, 0), title, font=font)[2]
        draw.text((x + max(0, (im.width - text_w) // 2), pad), title, fill=(30, 41, 59), font=font)
        canvas.paste(im, (x, pad + title_h))
        x += im.width + pad
    canvas.save(path)


def heat_color(value):
    value = float(np.clip(value, 0.0, 1.0))
    # dark blue -> cyan -> yellow -> red, easy to read in print.
    stops = [
        (0.00, np.array([18, 36, 92])),
        (0.35, np.array([33, 145, 180])),
        (0.70, np.array([253, 231, 92])),
        (1.00, np.array([180, 45, 38])),
    ]
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        if value <= b:
            t = (value - a) / max(b - a, 1e-9)
            return tuple((ca * (1 - t) + cb * t).astype(np.uint8))
    return tuple(stops[-1][1].astype(np.uint8))


def save_attention_heatmap(attention, path, title, max_width=1400):
    attention = normalize_minmax(attention)
    k, b = attention.shape
    cell_w = max(1, min(10, max_width // max(b, 1)))
    cell_h = max(3, min(12, 720 // max(k, 1)))
    left, top, right_pad, bottom_pad = 110, 82, 30, 78
    w = left + b * cell_w + right_pad
    h = top + k * cell_h + bottom_pad
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(28, bold=True)
    tick_font = get_font(18)
    draw.text((left, 20), title, fill=(30, 41, 59), font=title_font)
    for y in range(k):
        for x in range(b):
            color = heat_color(attention[y, x])
            x0 = left + x * cell_w
            y0 = top + y * cell_h
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill=color)
    draw.rectangle((left, top, left + b * cell_w, top + k * cell_h), outline=(80, 88, 101), width=2)
    draw.text((left, h - 46), "Real spectral band index", fill=(80, 88, 101), font=tick_font)
    draw.text((16, top + k * cell_h // 2 - 10), "Latent anchor", fill=(80, 88, 101), font=tick_font)
    img.save(path)


def save_anchor_curves(attention, positions, path, title):
    attention = np.asarray(attention, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.float32)
    k, _ = attention.shape
    anchors = sorted(set([0, k // 8, k // 4, k // 2, 3 * k // 4, k - 1]))
    colors = [(46, 111, 158), (75, 159, 121), (196, 106, 71), (122, 111, 179), (211, 155, 58), (182, 75, 85)]
    W, H = 1500, 820
    left, top, right, bottom = 110, 100, 1430, 650
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(30, bold=True)
    font = get_font(20)
    draw.text((left, 28), title, fill=(30, 41, 59), font=title_font)
    for i in range(6):
        y = top + (bottom - top) * i / 5
        draw.line((left, y, right, y), fill=(220, 225, 232), width=2)
    draw.line((left, bottom, right, bottom), fill=(100, 116, 139), width=3)
    draw.line((left, top, left, bottom), fill=(100, 116, 139), width=3)
    x_min, x_max = float(positions.min()), float(positions.max())
    y_max = max(float(attention[a].max()) for a in anchors)
    y_max = max(y_max, 1e-6)
    for idx, anchor in enumerate(anchors):
        row = attention[anchor]
        pts = []
        for pos, val in zip(positions, row):
            x = left + (float(pos) - x_min) / max(x_max - x_min, 1e-9) * (right - left)
            y = bottom - float(val) / y_max * (bottom - top)
            pts.append((x, y))
        color = colors[idx % len(colors)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=3)
        draw.rectangle((left + idx * 205, 700, left + idx * 205 + 26, 718), fill=color)
        draw.text((left + idx * 205 + 34, 696), f"anchor {anchor}", fill=(30, 41, 59), font=font)
    draw.text((left + 510, 660), "Band position / normalized wavelength", fill=(80, 88, 101), font=font)
    draw.text((22, top + 210), "Attention", fill=(80, 88, 101), font=font)
    img.save(path)


def save_scatter(points, labels, path, title):
    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    W, H = 1100, 900
    left, top, right, bottom = 90, 90, 1040, 790
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    title_font = get_font(30, bold=True)
    font = get_font(18)
    draw.text((left, 28), title, fill=(30, 41, 59), font=title_font)
    draw.rectangle((left, top, right, bottom), outline=(100, 116, 139), width=2)
    x = normalize_minmax(points[:, 0])
    y = normalize_minmax(points[:, 1])
    for px, py, lab in zip(x, y, labels):
        cx = int(left + px * (right - left))
        cy = int(bottom - py * (bottom - top))
        color = tuple(label_to_rgb(np.array([[lab]], dtype=np.int64))[0, 0])
        draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=color)
    used = sorted(int(v) for v in np.unique(labels) if v > 0)[:20]
    lx, ly = left, 815
    for lab in used:
        color = tuple(label_to_rgb(np.array([[lab]], dtype=np.int64))[0, 0])
        draw.rectangle((lx, ly, lx + 18, ly + 18), fill=color)
        draw.text((lx + 24, ly - 3), str(lab), fill=(30, 41, 59), font=font)
        lx += 62
        if lx > right - 60:
            lx = left
            ly += 26
    img.save(path)


def build_prepare_args(args):
    return preparation_args_from_config(vars(args))


def sample_features(features, gt, samples_per_class, seed):
    rng = np.random.default_rng(seed)
    gt_flat = gt.reshape(-1)
    feats = features.reshape(gt_flat.shape[0], -1)
    xs, ys = [], []
    for cls in sorted(int(v) for v in np.unique(gt_flat) if v > 0):
        idx = np.flatnonzero(gt_flat == cls)
        if idx.size == 0:
            continue
        if idx.size > samples_per_class:
            idx = rng.choice(idx, size=samples_per_class, replace=False)
        xs.append(feats[idx])
        ys.append(np.full(idx.size, cls, dtype=np.int64))
    if not xs:
        return None, None
    return np.vstack(xs), np.concatenate(ys)


def main():
    parser = argparse.ArgumentParser(description="Generate current VCA-SAHGCN paper visualizations.")
    add_interaction_arguments(parser)
    parser.add_argument("--graph-mode", default="static", choices=["static", "learned"])
    parser.add_argument("--checkpoint", type=str, default=r"experiments\paper_ablation_20260604-150049\checkpoints\vca_shared_domainbn_k64_seed2026.pt")
    parser.add_argument("--datasets", type=str, default="paviaU,xuzhou,WHUL")
    parser.add_argument("--out-dir", type=str, default=r"paper_vca_visualizations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--samples-type", type=str, default="ratio", choices=["ratio", "number"])
    parser.add_argument("--train-ratio", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.005)
    parser.add_argument("--train-num", type=int, default=30)
    parser.add_argument("--val-num", type=int, default=10)
    parser.add_argument("--superpixel-scale", type=int, default=100)
    parser.add_argument("--pavia-superpixel-scale", type=int, default=50)
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--adapter-embed-dim", type=int, default=32)
    parser.add_argument("--adapter-attention-mode", type=str, default="wavelength", choices=["wavelength", "index", "uniform"])
    parser.add_argument("--locality-sigma", type=float, default=0.08)
    parser.add_argument("--locality-window", type=float, default=0.16)
    parser.add_argument("--norm-type", type=str, default="domain_bn", choices=["domain_bn", "batch_norm", "group_norm"])
    parser.add_argument("--wavelength-min", type=float, default=400.0)
    parser.add_argument("--wavelength-max", type=float, default=2500.0)
    parser.add_argument("--map-scale", type=int, default=2)
    parser.add_argument("--tsne-samples-per-class", type=int, default=400)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--try-umap", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(os.path.abspath(args.out_dir))
    device = torch.device(args.device)
    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    model = load_model_checkpoint(args.checkpoint, device=device, config=vars(args))
    config = {**vars(args), **model.training_config, "graph_mode": model.backbone.graph_mode}
    prep_args = preparation_args_from_config(config)
    datasets = {}
    for name in dataset_names:
        if name not in model.heads:
            raise ValueError("Checkpoint has no dataset head for " + name)
        datasets[name] = prepare_dataset(name, prep_args, device=device, seed=dataset_seed(config, name, model.heads))

    with torch.no_grad():
        for name, item in datasets.items():
            ds_dir = ensure_dir(os.path.join(out_dir, name))
            y, _, _ = model(
                name,
                item["net_input"],
                item["Q"],
                item["A"],
                wavelengths=item["wavelengths"],
                fwhm=item["fwhm"],
                superpixel_scale=item["superpixel_scale"],
            )
            pred = torch.argmax(y, dim=1).detach().cpu().numpy().reshape(item["height"], item["width"]) + 1
            gt = item["test_samples_gt"].detach().cpu().numpy().reshape(item["height"], item["width"]).astype(np.int64)
            full_gt = gt.copy()
            pred_masked = np.where(full_gt > 0, pred, 0)

            gt_path = os.path.join(ds_dir, f"{name}_ground_truth.png")
            pred_path = os.path.join(ds_dir, f"{name}_prediction.png")
            panel_path = os.path.join(ds_dir, f"{name}_gt_vs_prediction.png")
            save_label_png(full_gt, gt_path, scale=args.map_scale)
            save_label_png(pred_masked, pred_path, scale=args.map_scale)
            make_panel(
                [Image.open(gt_path).convert("RGB"), Image.open(pred_path).convert("RGB")],
                ["Ground Truth", "Prediction"],
                panel_path,
            )

            bands = int(item["bands"])
            dummy = torch.zeros((1, bands), device=device, dtype=torch.float32)
            _, attn_info = model.backbone.adapter._adapt_flat(
                dummy,
                wavelengths=item["wavelengths"],
                fwhm=item["fwhm"],
                return_attention=True,
            )
            attention = attn_info["attention"].detach().cpu().numpy()
            positions = attn_info["band_positions"].detach().cpu().numpy()
            save_attention_heatmap(
                attention,
                os.path.join(ds_dir, f"{name}_vca_attention_heatmap.png"),
                f"{name} VCA Attention Map",
            )
            save_anchor_curves(
                attention,
                positions,
                os.path.join(ds_dir, f"{name}_vca_anchor_attention_curves.png"),
                f"{name} Anchor-to-Band Attention",
            )

            if not args.skip_tsne:
                fused, _, _ = model.backbone.forward_features(
                    item["net_input"],
                    item["Q"],
                    item["A"],
                    wavelengths=item["wavelengths"],
                    fwhm=item["fwhm"],
                    dataset_name=name,
                    superpixel_scale=item["superpixel_scale"],
                )
                features = fused.detach().cpu().numpy()
                sample_x, sample_y = sample_features(features, full_gt, args.tsne_samples_per_class, args.seed)
                if sample_x is not None and sample_x.shape[0] > 2:
                    from sklearn.manifold import TSNE
                    perplexity = min(30, max(5, sample_x.shape[0] // 20))
                    points = TSNE(n_components=2, random_state=args.seed, perplexity=perplexity).fit_transform(sample_x)
                    save_scatter(points, sample_y, os.path.join(ds_dir, f"{name}_tsne_features.png"), f"{name} t-SNE Features")
                    if args.try_umap:
                        try:
                            import umap
                            points = umap.UMAP(n_components=2, random_state=args.seed).fit_transform(sample_x)
                            save_scatter(points, sample_y, os.path.join(ds_dir, f"{name}_umap_features.png"), f"{name} UMAP Features")
                        except Exception as exc:
                            print(f"skip {name} UMAP: {exc}")

            print(f"wrote {name} visualizations to {ds_dir}")


if __name__ == "__main__":
    main()
