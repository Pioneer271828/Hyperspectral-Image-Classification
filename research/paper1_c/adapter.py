"""Isolated DSNet input/output adapter; no upstream source is redistributed.

The external model remains GPL-3.0 at the pinned author revision. This file
does not replace a project model, dataset reader, graph builder or trainer.
"""
from pathlib import Path
import hashlib
import importlib.util
import subprocess

import torch
import torch.nn.functional as F

UPSTREAM = "e910795073b52373ce0771ac85139a17f250b236"
MODEL_BLOB = "86d6843ade5a18b2187e20854a0b66d183baf5df"


def load_author_model(root):
    root = Path(root).resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    raw = (root / "model.py").read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if head != UPSTREAM or dirty or blob != MODEL_BLOB:
        raise ValueError("Author revision, clean state or model blob differs")
    spec = importlib.util.spec_from_file_location("paper1_c_dsnet_author", root / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patches(values, points, *, size=7):
    """Extract HWC tile-local patches at local y/x indices; never read a halo.

    Padding repeats the current tile edge, including when a tile is smaller
    than the patch. No global image, label raster, or cross-role lookup exists.
    """
    if values.ndim != 3 or not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError("Expected finite floating HWC observation")
    if size < 5 or size % 2 != 1 or not points:
        raise ValueError("Expected nonempty points and odd patch size >=5")
    h, w, _ = values.shape
    if any(type(y) is not int or type(x) is not int or not 0 <= y < h or not 0 <= x < w
           for y, x in points) or len(set(points)) != len(points):
        raise ValueError("Patch centers repeat or escape the admitted tile")
    radius = size // 2
    tensor = F.pad(values.permute(2, 0, 1).unsqueeze(0), (radius,) * 4, mode="replicate")
    return torch.stack([tensor[0, :, y:y+size, x:x+size] for y, x in points])


class Adapter(torch.nn.Module):
    def __init__(self, author, bands, classes, size=7):
        super().__init__()
        if bands < 4 or classes < 2 or size < 5 or size % 2 != 1:
            raise ValueError("Unsupported DSNet dimensions")
        self.model = author.DSNet(bands, classes, size, "conv2d_unmix")
        self.bands, self.classes, self.size = bands, classes, size

    def forward(self, inputs):
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (self.bands, self.size, self.size):
            raise ValueError("Patch dimensions differ from frozen model")
        if not torch.isfinite(inputs).all():
            raise ValueError("Nonfinite input")
        nonlinear, linear, logits = self.model(inputs)
        if any(not torch.isfinite(x).all() for x in (nonlinear, linear, logits)):
            raise FloatingPointError("Original DSNet emitted nonfinite outputs; no silent epsilon repair")
        return nonlinear, linear, logits


def objective(outputs, inputs, labels):
    """Author demo's CE + SAD arithmetic, independently expressed.

    Reconstruction is nonlinear + first-L + last-L decoder channels. Keeping
    the unclamped cosine intentionally exposes the original singularities.
    """
    nonlinear, linear, logits = outputs
    bands = inputs.shape[1]
    if linear.shape[1] != 2 * bands or labels.dtype != torch.long:
        raise ValueError("Decoder or label contract differs")
    reconstructed = nonlinear + (linear[:, :bands] + linear[:, bands:])
    cosine = (inputs * reconstructed).sum(1) / (
        torch.linalg.vector_norm(reconstructed, dim=1) * torch.linalg.vector_norm(inputs, dim=1))
    loss = F.cross_entropy(logits, labels) + cosine.acos().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Original unclamped SAD objective is nonfinite")
    return loss


def first_minimum(events):
    """Select only finite development CE NLL; ties keep the earliest event."""
    if not events or any(not torch.isfinite(torch.tensor(row[1])) for row in events):
        raise ValueError("Missing or nonfinite development selection")
    if any(type(row[0]) is not int or row[0] < 0 for row in events):
        raise ValueError("Invalid update index")
    if [x[0] for x in events] != sorted(set(x[0] for x in events)):
        raise ValueError("Selection events must be strictly ordered")
    return min(events, key=lambda row: row[1])[0]
