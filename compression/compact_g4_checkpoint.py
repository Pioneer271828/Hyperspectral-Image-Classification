"""Pinned initialization checkpoints for physically narrowed G4 CNN stage one."""
import hashlib
import io
import os
from pathlib import Path
import tempfile

import torch

from joint_training import _tree_fingerprint
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest

from .g4_cnn_stage1 import _CompactModel, _template


KIND = "g4_compact_cnn_stage1_initialization"


def _restore_model(architecture, mapping, state):
    if mapping["policy"] != "mon_explicit_compact_channel_map_v1" or mapping["optimizer_state_transferred"] is not False:
        raise ValueError("Initial compact channel mapping required")
    first_axis = mapping["axes"][0]
    removed = [entry["index"] for entry in first_axis["removed"]]
    with torch.random.fork_rng(devices=[]):
        stock = MultiDatasetVCASAHGCN(architecture["class_counts"], **architecture["backbone_kwargs"])
        compact, retained, axes = _template(stock, removed)
    old_to_new = [None] * 128
    for new, old in enumerate(retained):
        old_to_new[old] = new
    expected_axes = [{"tensor": name, "axis": axis, "old_to_new": old_to_new,
                      "new_to_old": retained, "removed": first_axis["removed"]} for name, axis in axes]
    if mapping["axes"] != expected_axes:
        raise ValueError("Complete compact topology and channel mapping must match")
    template = compact.state_dict()
    if set(template) != set(state) or any(
        not torch.is_tensor(state[name]) or (state[name].shape, state[name].dtype, state[name].layout)
        != (value.shape, value.dtype, value.layout) for name, value in template.items()
    ):
        raise ValueError("Compact checkpoint tensor schema mismatch")
    if _tree_fingerprint(state) != mapping["compact_state"]:
        raise ValueError("Checkpoint tensors differ from the mapped initialization")
    compact.load_state_dict(state, strict=True)
    return compact


def save_compact_initialization(path, model, mapping, *, expected_mapping_sha256):
    """Save the initial mapped state as one atomic file with its topology and mapping."""
    if type(model) is not _CompactModel or digest(mapping) != expected_mapping_sha256:
        raise ValueError("Compact model and exact pinned channel mapping required")
    architecture = MultiDatasetVCASAHGCN.export_config(model)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    _restore_model(architecture, mapping, state)
    payload = {"schema_version": 1, "kind": KIND, "architecture": architecture, "mapping": mapping,
               "state": state}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".compact-init-", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        raw_sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.link(temporary, path)
    finally:
        temporary.unlink()
    return raw_sha


def load_compact_initialization(path, *, expected_sha256, expected_mapping_sha256, device="cpu"):
    """Read exact pinned bytes and rebuild the same compact model in evaluation mode."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Compact checkpoint file hash mismatch")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("kind") != KIND:
        raise ValueError("Unsupported compact checkpoint schema")
    mapping = payload["mapping"]
    if digest(mapping) != expected_mapping_sha256:
        raise ValueError("Compact checkpoint mapping hash mismatch")
    model = _restore_model(payload["architecture"], mapping, payload["state"])
    return model.to(device), mapping
