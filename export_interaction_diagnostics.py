"""Export predictions, directional gate maps and sparse edge changes for any new run."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model.checkpoint_io import load_model_checkpoint, preparation_args_from_config, dataset_seed
from train_vca_multidataset import prepare_dataset


def export_dataset(model, name, item, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        outputs = model(
            name, item["net_input"], item["Q"], item["A"], wavelengths=item["wavelengths"],
            fwhm=item["fwhm"], superpixel_scale=item["superpixel_scale"],
            return_graph_info=True, return_diagnostics=True,
        )
    height, width = item["height"], item["width"]
    arrays = {key: output.argmax(-1).reshape(height, width).cpu().numpy() + 1
              for key, output in zip(("prediction", "graph_prediction", "cnn_prediction"), outputs[:3])}
    stats = {}
    for key, value in outputs[3].get("interaction", {}).items():
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
            if value.ndim == 0:
                stats[key] = float(value.reshape(-1)[0])
            else:
                arrays[key] = value
        else:
            stats[key] = value
    for key in ("graph_to_cnn_gate", "cnn_to_graph_gate_pixels"):
        if key in arrays:
            gate = arrays[key].reshape(height, width)
            # Values have a fixed [0,1] scale across experiments; save raw values in NPZ too.
            Image.fromarray(np.uint8(np.clip(gate, 0, 1) * 255)).save(out_dir / (key + ".png"))
    np.savez_compressed(out_dir / "interaction_arrays.npz", **arrays)
    (out_dir / "interaction_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metrics-json", help="Model/training config for a legacy checkpoint without a sidecar")
    parser.add_argument("--datasets", help="Comma-separated subset; default: all checkpoint heads")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    metric = json.loads(Path(args.metrics_json).read_text(encoding="utf-8")) if args.metrics_json else {}
    config = metric.get("config")
    model = load_model_checkpoint(args.checkpoint, device=args.device, config=config)
    training_config = {**(config or {}), **model.training_config, "graph_mode": model.backbone.graph_mode}
    prep_args = preparation_args_from_config(training_config)
    order = list(model.heads.keys())
    selected = [name.strip() for name in args.datasets.split(",") if name.strip()] if args.datasets else order
    training_config.setdefault("seed", metric.get("seed", 2026))
    for name in selected:
        if name not in order:
            parser.error("Checkpoint has no head for " + name)
        item = prepare_dataset(name, prep_args, torch.device(args.device), dataset_seed(training_config, name, order))
        export_dataset(model, name, item, Path(args.out_dir) / name)
        print("Exported interaction diagnostics:", name)


if __name__ == "__main__":
    main()
