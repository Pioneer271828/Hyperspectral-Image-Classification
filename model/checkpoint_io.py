"""Keep legacy state_dict files while saving exact architecture in .pt.json sidecars."""

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from model.VCASAHGCN import MultiDatasetVCASAHGCN, VCASAHGCNBackbone


def backbone_kwargs_from_config(config):
    fields = inspect.signature(VCASAHGCNBackbone.__init__).parameters
    return {key: value for key, value in config.items() if key in fields and key != "self"}


def save_model_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    metadata = model.export_config()
    metadata["training_config"] = getattr(model, "training_config", {})
    Path(str(path) + ".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def read_checkpoint_metadata(path):
    sidecar = Path(str(path) + ".json")
    return json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else None


def load_model_checkpoint(path, device="cpu", config=None, class_counts=None):
    metadata = read_checkpoint_metadata(path)
    state = torch.load(path, map_location=device, weights_only=True)
    if metadata is not None:
        class_counts = metadata["class_counts"]
        kwargs = metadata["backbone_kwargs"]
    else:
        if config is None:
            raise ValueError("Legacy checkpoint needs explicit model config (no .pt.json sidecar)")
        kwargs = backbone_kwargs_from_config(config)
        if class_counts is None:
            class_counts = {key[len("heads."):-len(".weight")]: value.shape[0]
                            for key, value in state.items()
                            if key.startswith("heads.") and key.endswith(".weight")}
        if not class_counts:
            raise ValueError("Expected a MultiDatasetVCASAHGCN checkpoint with dataset heads")
    model = MultiDatasetVCASAHGCN(class_counts, **kwargs).to(device)
    model.load_state_dict(state, strict=True)
    model.training_config = metadata.get("training_config", {}) if metadata else dict(config)
    return model.eval()


def preparation_args_from_config(config):
    defaults = dict(samples_type="ratio", train_num=30, val_num=10, train_ratio=0.01,
                    val_ratio=0.005, superpixel_scale=100, pavia_superpixel_scale=50, graph_mode="static")
    return SimpleNamespace(**{key: config.get(key, value) for key, value in defaults.items()})


def dataset_seed(config, name, dataset_order):
    """Preserve split seeds even if preparation skipped an earlier missing dataset."""
    recorded = config.get("dataset_seeds", {})
    return int(recorded.get(name, int(config.get("seed", 2026)) + list(dataset_order).index(name)))
