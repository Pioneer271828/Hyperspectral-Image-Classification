import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from loadData import data_read
from model import VCASAHGCN
from train_vca_multidataset import prepare_dataset
from model.checkpoint_io import load_model_checkpoint, preparation_args_from_config, dataset_seed


HONGHU_19_PALETTE = np.array([
    [0, 0, 0],
    [222, 0, 0],
    [205, 133, 63],
    [64, 128, 33],
    [255, 255, 0],
    [0, 255, 128],
    [0, 128, 255],
    [128, 0, 0],
    [0, 188, 0],
    [255, 192, 203],
    [192, 192, 192],
    [139, 69, 19],
    [0, 100, 0],
    [128, 128, 0],
    [128, 0, 128],
    [0, 0, 128],
    [128, 128, 128],
    [155, 205, 0],
    [55, 90, 250],
    [60, 128, 25],
], dtype=np.uint8)


PAVIAU_PALETTE = np.array([
    [0, 0, 0],
    [255, 128, 0],
    [0, 128, 255],
    [255, 0, 255],
    [255, 255, 0],
    [0, 255, 128],
    [128, 0, 128],
    [0, 0, 255],
    [0, 128, 0],
    [255, 192, 203],
], dtype=np.uint8)


XUZHOU_PALETTE = np.array([
    [0, 0, 0],
    [222, 0, 0],
    [205, 133, 63],
    [64, 128, 33],
    [255, 255, 0],
    [0, 255, 128],
    [0, 128, 255],
    [128, 0, 0],
    [0, 188, 0],
    [255, 192, 203],
], dtype=np.uint8)


def make_prepare_args(config=None):
    return preparation_args_from_config(config or {})


def label_to_rgb(label, palette):
    label = np.asarray(label, dtype=np.int64)
    clipped = np.where((label >= 0) & (label < len(palette)), label, 0)
    return palette[clipped]


def save_label_map(label, palette, path, scale=2):
    rgb = label_to_rgb(label, palette)
    image = Image.fromarray(rgb, mode="RGB")
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    image.save(path)


def read_class_counts(dataset_names):
    counts = {}
    for name in dataset_names:
        _, gt = data_read.readData(name)
        counts[name] = int(np.max(gt))
    return counts


def build_model(class_counts, config, device):
    config = {"norm_type": "domain_bn", "adapter_checkpoint_chunks": True, **config}
    return load_model_checkpoint(config["checkpoint"], device=device, config=config, class_counts=class_counts)


def predict_label(model, dataset_name, item):
    with torch.no_grad():
        y, _, _ = model(
            dataset_name,
            item["net_input"],
            item["Q"],
            item["A"],
            wavelengths=item["wavelengths"],
            fwhm=item["fwhm"],
            superpixel_scale=item["superpixel_scale"],
        )
    pred = torch.argmax(y, dim=1).detach().cpu().numpy().reshape(item["height"], item["width"]) + 1
    return pred.astype(np.int16)


def remap_honghu_full_to_19(label):
    cropped = np.asarray(label)[450:, :]
    out = np.zeros_like(cropped, dtype=np.int16)
    for old_label in range(1, 5):
        out[cropped == old_label] = old_label
    for old_label in range(6, 21):
        out[cropped == old_label] = old_label - 1
    return out


def mask_prediction(pred, dataset_name):
    _, gt = data_read.readData(dataset_name)
    gt = np.asarray(gt)
    return np.where(gt > 0, pred, 0).astype(np.int16)


def export_paviau_xuzhou(out_dir, device, scale):
    dataset_order = ["paviaU", "xuzhou", "WHUL"]
    prepared_names = ["paviaU", "xuzhou"]
    class_counts = read_class_counts(dataset_order)
    model = build_model(
        class_counts,
        {
            "checkpoint": r"experiments\paper_ablation_20260604-150049\checkpoints\vca_shared_domainbn_k64_seed2026.pt",
            "latent_channels": 64,
            "adapter_embed_dim": 32,
            "adapter_chunk_pixels": 0,
        },
        device,
    )
    args = make_prepare_args({**model.training_config, "graph_mode": model.backbone.graph_mode})
    palettes = {"paviaU": PAVIAU_PALETTE, "xuzhou": XUZHOU_PALETTE}
    output_names = {
        "paviaU": "University_of_Pavia_classification_result.png",
        "xuzhou": "Xuzhou_classification_result.png",
    }
    for index, name in enumerate(prepared_names):
        item = prepare_dataset(name, args, device, seed=dataset_seed(model.training_config, name, model.heads))
        pred = mask_prediction(predict_label(model, name, item), name)
        save_label_map(pred, palettes[name], os.path.join(out_dir, output_names[name]), scale=scale)


def export_honghu(out_dir, device, scale):
    class_counts = {"whu-honghu": 22}
    model = build_model(
        class_counts,
        {
            "checkpoint": r"experiments\honghu_vca_k32\checkpoints\vca_honghu_k32_seed2026.pt",
            "latent_channels": 32,
            "adapter_embed_dim": 16,
            "adapter_chunk_pixels": 65536,
        },
        device,
    )
    args = make_prepare_args({**model.training_config, "graph_mode": model.backbone.graph_mode})
    item = prepare_dataset("whu-honghu", args, device, seed=dataset_seed(model.training_config, "whu-honghu", model.heads))
    pred = predict_label(model, "whu-honghu", item)
    _, gt = data_read.readData("whu-honghu")
    pred_19 = remap_honghu_full_to_19(pred)
    gt_19 = remap_honghu_full_to_19(gt)
    pred_19 = np.where(gt_19 > 0, pred_19, 0).astype(np.int16)
    save_label_map(
        pred_19,
        HONGHU_19_PALETTE,
        os.path.join(out_dir, "WHU-Hi-HongHu_classification_result.png"),
        scale=scale,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=r"paper_table_palette_classification_maps")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--skip-honghu", action="store_true")
    parser.add_argument("--skip-paviau-xuzhou", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    if not args.skip_paviau_xuzhou:
        export_paviau_xuzhou(args.out_dir, device, args.scale)
    if not args.skip_honghu:
        export_honghu(args.out_dir, device, args.scale)
    print("wrote classification maps to {}".format(os.path.abspath(args.out_dir)))


if __name__ == "__main__":
    main()
