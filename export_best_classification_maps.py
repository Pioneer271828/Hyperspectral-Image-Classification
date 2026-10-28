import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from loadData import data_read
from model import VCASAHGCN
from train_vca_multidataset import prepare_dataset
from model.checkpoint_io import load_model_checkpoint, preparation_args_from_config


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


BEST_SPECS = {
    "paviaU": {
        "label": "University of Pavia",
        "metric_json": r"experiments\ablation_no_wavelength_attention\metrics\vca_no_wave_attention_k64_seed0.json",
        "seed_offset": 0,
        "palette": PAVIAU_PALETTE,
        "output": "University_of_Pavia_best_classification_result.png",
        "mask_dataset": "paviaU",
    },
    "xuzhou": {
        "label": "Xuzhou",
        "metric_json": r"experiments\ablation_latent_channels_no_baseline\metrics\vca_k128_domainbn_seed3407.json",
        "seed_offset": 1,
        "palette": XUZHOU_PALETTE,
        "output": "Xuzhou_best_classification_result.png",
        "mask_dataset": "xuzhou",
    },
    "whu-honghu": {
        "label": "WHU-Hi-HongHu",
        "metric_json": r"experiments\honghu_vca_k32\metrics\vca_honghu_k32_seed2026.json",
        "seed_offset": 0,
        "palette": HONGHU_19_PALETTE,
        "output": "WHU-Hi-HongHu_best_classification_result.png",
        "mask_dataset": "whu-honghu",
        "remap": "honghu_19",
    },
}


def prepare_args(config=None):
    return preparation_args_from_config(config or {})


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_path(path):
    return str(Path(path).expanduser().resolve())


def class_counts(dataset_names):
    counts = {}
    for name in dataset_names:
        _, gt = data_read.readData(name)
        counts[name] = int(np.max(gt))
    return counts


def build_model(metric, device):
    cfg = metric["config"]
    ckpt = metric.get("eval_checkpoint_path") or metric["checkpoint_paths"]["primary"]
    cfg = {"norm_type": "domain_bn", "adapter_checkpoint_chunks": True, **cfg}
    return load_model_checkpoint(resolve_path(ckpt), device=device, config=cfg)


def predict(model, dataset_name, item):
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
    return (torch.argmax(y, dim=1).detach().cpu().numpy().reshape(item["height"], item["width"]) + 1).astype(np.int16)


def label_to_rgb(label, palette):
    label = np.asarray(label, dtype=np.int64)
    safe = np.where((label >= 0) & (label < len(palette)), label, 0)
    return palette[safe]


def save_label(label, palette, path, scale):
    image = Image.fromarray(label_to_rgb(label, palette), mode="RGB")
    if scale != 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    image.save(path)


def remap_honghu_full_to_19(label):
    cropped = np.asarray(label)[450:, :]
    remapped = np.zeros_like(cropped, dtype=np.int16)
    for old_label in range(1, 5):
        remapped[cropped == old_label] = old_label
    for old_label in range(6, 21):
        remapped[cropped == old_label] = old_label - 1
    return remapped


def mask_and_remap(pred, dataset_name, remap=None):
    _, gt = data_read.readData(dataset_name)
    gt = np.asarray(gt)
    if remap == "honghu_19":
        pred = remap_honghu_full_to_19(pred)
        gt = remap_honghu_full_to_19(gt)
    return np.where(gt > 0, pred, 0).astype(np.int16)


def export_one(name, spec, out_dir, device, scale):
    metric = read_json(spec["metric_json"])
    seed = int(metric["seed"]) + int(spec.get("seed_offset", 0))
    model = build_model(metric, device)
    config = {**metric["config"], **model.training_config, "graph_mode": model.backbone.graph_mode}
    seed = int(config.get("dataset_seeds", {}).get(name, seed))
    item = prepare_dataset(name, prepare_args(config), device=device, seed=seed)
    pred = predict(model, name, item)
    pred = mask_and_remap(pred, spec["mask_dataset"], remap=spec.get("remap"))
    output_path = os.path.join(out_dir, spec["output"])
    save_label(pred, spec["palette"], output_path, scale)
    return {
        "dataset": spec["label"],
        "output": os.path.abspath(output_path),
        "metric_json": os.path.abspath(spec["metric_json"]),
        "checkpoint": os.path.abspath(metric.get("eval_checkpoint_path") or metric["checkpoint_paths"]["primary"]),
        "test": metric["test"][name],
        "config": metric["config"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=r"paper_best_classification_maps")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--datasets", default="whu-honghu,paviaU,xuzhou")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    selected = [name.strip() for name in args.datasets.split(",") if name.strip()]
    manifest = []
    for name in selected:
        manifest.append(export_one(name, BEST_SPECS[name], args.out_dir, device, args.scale))
    manifest_path = os.path.join(args.out_dir, "best_classification_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    print("wrote best classification maps to {}".format(os.path.abspath(args.out_dir)))
    print("manifest:", os.path.abspath(manifest_path))


if __name__ == "__main__":
    main()
