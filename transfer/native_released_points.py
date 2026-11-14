"""Equal-weight supervision for independently identified released original pixels."""
from dataclasses import dataclass

import numpy as np
import torch

from model.research_config import digest
from transfer.development_blocks import array_hash, sample_identity
from transfer.native_scale_observation import _immutable
from transfer.native_scale_supervision import Targets


MODE = "released_original_points_v1"
HEADS = ("fused", "graph", "cnn")
LOSS = {"mode": MODE, "head_weights": [1, 1, 1], "class_balance": False,
        "normalization": "per_original_released_point_per_head", "probability_floor": 1e-8,
        "duplicate_policy": "reject", "unknown_pixels": "excluded",
        "graph_regularization": "native_scale_existing_zero_weights"}


@dataclass(frozen=True)
class ReleasedPoints:
    points: tuple
    cells: np.ndarray
    targets: Targets
    source: str

    def weights(self, mode):
        if mode != MODE:
            raise ValueError("Released points require the registered loss mode")
        hist = self.targets.histogram
        counts = hist.sum(axis=1)
        if not counts.any():
            raise ValueError("No released points for native supervision")
        # Counts times a binary mask use the unchanged three-head probability
        # loss. Supply the true number of points separately as its denominator.
        values = hist / np.maximum(counts[:, None], 1)
        return torch.tensor(values, dtype=torch.float32), torch.tensor(
            np.broadcast_to(counts[:, None], hist.shape).copy(), dtype=torch.float32)

    @property
    def histogram(self):
        return self.targets.histogram

    @property
    def area(self):
        return self.targets.area

    def summary(self):
        hist = self.histogram
        conflicting = (hist > 0).sum(axis=1) > 1
        max_count = hist.max(axis=1, keepdims=True)
        minority = (hist < max_count) & conflicting[:, None]
        return {"mode": MODE, "source": self.source, "points": len(self.points),
                "point_set": digest(self.points), "mapping": array_hash(self.cells),
                "histogram": array_hash(hist), "per_class": hist.sum(axis=0).tolist(),
                "known_cells": int((hist.sum(axis=1) > 0).sum()), "conflicting_cells": int(conflicting.sum()),
                "conflicting_points_per_class": hist[conflicting].sum(axis=0).tolist(),
                "minority_points_per_class": np.where(minority, hist, 0).sum(axis=0).tolist(),
                "unknown_pixels": int(self.area.sum()) - len(self.points), "complete_purity": None}


def bind(actual, row, producer, role, *, synthetic_labels=None):
    receipt = actual.verify(producer, role)
    source = receipt["source"]
    if (row["id"], row["role"], list(row["box"])) != (source["tile"], role, receipt["box"]):
        raise ValueError("Released point geometry or role differs")
    ident = sample_identity(row, (*actual.lattice.scene_shape, 0), actual.classes)
    points = tuple(tuple(point) for point in row["samples"])
    if synthetic_labels is None:
        if receipt["synthetic_profile"] is not None or source["samples"] != ident:
            raise ValueError("Released point set differs from the authorized bank")
    else:
        if receipt["synthetic_profile"] is None or array_hash(synthetic_labels) != source["labels"]:
            raise ValueError("Synthetic sparse labels differ from their generator")
    y0, _, x0, _ = actual.lattice.box
    width = actual.lattice.scene_shape[1]
    members = actual.lattice.membership()
    hist = np.zeros((len(actual.lattice.cells), actual.classes), dtype=np.int64)
    cells = []
    for pixel, label in points:
        y, x = divmod(pixel, width)
        cell = members[y-y0, x-x0]
        if synthetic_labels is not None and synthetic_labels[y-y0, x-x0] != label:
            raise ValueError("Synthetic released label differs from its source")
        hist[cell, label-1] += 1
        cells.append(cell)
    return ReleasedPoints(points, _immutable(np.asarray(cells, dtype=np.int64)),
                          Targets(hist, actual.lattice.areas(), False), ident["sha256"])


def scores(output, released, *, detailed):
    hist = released.histogram
    classes = hist.shape[1]
    truth = np.asarray([label-1 for _, label in released.points], dtype=np.int64)
    count = len(truth)
    if not count:
        raise ValueError("No released devval points to score")
    result = {}
    for name, tensor in zip(HEADS, output[:3], strict=True):
        probs = tensor.detach().cpu().numpy()
        if probs.shape != hist.shape or not np.isfinite(probs).all():
            raise ValueError("Native output differs from released point lattice")
        choices = probs.argmax(axis=1)
        nll_sum = float(-(hist * np.log(np.clip(probs.astype(np.float64), 1e-8, 1.0))).sum())
        row = dict(nll_sum=nll_sum, count=count)
        if detailed:
            assigned = choices[released.cells]
            point = np.bincount(truth * classes + assigned, minlength=classes**2).reshape(classes, classes)
            coarse = np.zeros((classes, classes), dtype=np.int64)
            np.add.at(coarse.T, choices, hist)
            if not np.array_equal(coarse, point):
                raise RuntimeError("Point and known-count coarse confusion disagree")
            row.update(point_confusion=point.tolist(), coarse_known_count_confusion=coarse.tolist(),
                       class_support=hist.sum(axis=0).tolist(), coarse_known_cells=int((hist.sum(axis=1)>0).sum()),
                       point_prediction_sha256=array_hash(probs[released.cells]),
                       independence_unit="original_spatial_block")
        result[name] = row
    return result
