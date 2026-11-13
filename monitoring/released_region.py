"""Observed boundaries and interventions on representations from released tiles."""
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from transfer.multiscale_region_graph import MultiRangeBackbone


NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))
STATES = ("observed_boundary", "observed_interior", "unknown")
RECOVERY = "wp06_fixed_released_region_recovery_v1"


def neighbors(point):
    y, x = point
    return ((y+dy, x+dx) for dy, dx in NEIGHBORS)


def observed_boundary(shape, released):
    labels = {(y, x): int(label) for y, x, label in released}
    if (len(labels) != len(released) or any(not (0 <= y < shape[0] and 0 <= x < shape[1] and label > 0)
                                           for (y, x), label in labels.items())):
        raise ValueError("Expected unique released points inside the original footprint")
    status = {}
    for point, label in labels.items():
        adjacent = [labels.get(other) for other in neighbors(point)]
        status[point] = ("observed_boundary" if any(value is not None and value != label for value in adjacent)
                         else "observed_interior" if adjacent == [label]*4 else "unknown")
    boundary = {point for point, value in status.items() if value == "observed_boundary"}
    bands = {}
    for radius in (1, 2, 3):
        offsets = [(dy, dx) for dy in range(1-radius, radius) for dx in range(1-radius, radius)
                   if abs(dy)+abs(dx) < radius]
        bands[radius] = {point for point in status if any((point[0]+dy, point[1]+dx) in boundary
                                                         for dy, dx in offsets)}
    return status, bands


def narrow_components(released):
    """Only fully enclosed components have known extent. Width is bounding-box width."""
    labels = {(y, x): int(label) for y, x, label in released}
    remaining, observed, unknown = set(labels), [], 0
    while remaining:
        seed = min(remaining)
        label = labels[seed]
        component, stack, enclosed = set(), [seed], True
        remaining.remove(seed)
        while stack:
            point = stack.pop()
            component.add(point)
            for other in neighbors(point):
                if other not in labels:
                    enclosed = False
                elif other in remaining and labels[other] == label:
                    remaining.remove(other)
                    stack.append(other)
        if enclosed:
            ys, xs = zip(*component, strict=True)
            observed.append({"class_id": label, "area": len(component),
                             "minimum_box_extent": min(max(ys)-min(ys)+1, max(xs)-min(xs)+1)})
        else:
            unknown += 1
    return {"enclosed_components": observed, "unknown_extent_components": unknown,
            "width_one_components": sum(row["minimum_box_extent"] == 1 for row in observed) if observed else None}


def point_geometry(actual, targets, *, include_points=False):
    y0, y1, x0, x1 = actual.lattice.box
    w = actual.lattice.scene_shape[1]
    local = [(pixel//w-y0, pixel%w-x0, label) for pixel, label in targets.points]
    states, bands = observed_boundary((y1-y0, x1-x0), local)
    class_support = {name: [0]*actual.classes for name in STATES}
    for y, x, label in local:
        class_support[states[y, x]][label-1] += 1
    bands_by_cell = {}
    for radius, band in bands.items():
        counts = Counter(int(targets.cells[i]) for i, (y, x, _) in enumerate(local) if (y, x) in band)
        bands_by_cell[str(radius)] = counts
    footprints = [{"cell": cell, "box": list(actual.lattice.cells[cell]),
                   "known_counts": targets.histogram[cell].tolist(),
                   "band_counts": [counts.get(cell, 0) for counts in bands_by_cell.values()]}
                  for cell in sorted({cell for counts in bands_by_cell.values() for cell in counts})]
    result = {"support": {name: sum(counts) for name, counts in class_support.items()}, "by_class": class_support,
              "radius_support": {str(radius): len(rows) for radius, rows in bands.items()},
              "coarse_footprints": footprints,
              "fine_narrow_region": narrow_components(local)}
    if include_points:
        result["point_state"] = [states[y, x] for y, x, _ in local]
        result["point_bands"] = [[radius for radius, band in bands.items() if (y, x) in band] for y, x, _ in local]
    return result


@dataclass(frozen=True)
class RegionAction:
    operation: str
    target: int = 0
    donor: int | None = None

    def apply(self, value):
        if value.ndim != 2 or not 0 <= self.target < len(value):
            raise ValueError("Registered region target differs from the actual representation")
        if self.operation == "zero" and self.donor is None:
            replacement = value[self.target].new_zeros(value.shape[-1])
        elif (self.operation == "replace" and self.donor is not None
              and 0 <= self.donor < len(value) and self.donor != self.target):
            replacement = value[self.donor]
        else:
            raise ValueError("No legal donor or intervention operation")
        return value.index_copy(0, torch.tensor([self.target], device=value.device), replacement[None])


@contextmanager
def region_action(backbone, action):
    """Patch the existing graph stage on a private model and always restore dispatch."""
    method = "graph_stage2"
    if method in vars(backbone):
        raise ValueError("Graph stage already has an observer or intervention")
    stage = getattr(backbone, method)
    hits = []

    def apply(values, adjacency, dataset_name=None):
        result = stage(values, adjacency, dataset_name)
        if hits:
            raise ValueError("Region intervention observed multiple unregistered calls")
        hits.append(True)
        if isinstance(backbone, MultiRangeBackbone):
            narrow, wide = result.chunk(2, dim=0)
            return torch.cat((action.apply(narrow), action.apply(wide)), dim=0)
        return action.apply(result)

    setattr(backbone, method, apply)
    try:
        yield
        if not hits:
            raise ValueError("Declared region intervention did not execute")
    finally:
        delattr(backbone, method)


def recovery_contract(classes, seed, arm, factor, producer, operation, numeric):
    from transfer import native_released_training

    if (classes, seed, arm, factor) not in ((9, 64, "joint", 2), (22, 64, "joint", 2)):
        raise ValueError("Outside the bounded recovery fixture")
    if operation not in ("baseline", "zero"):
        raise ValueError("Unknown fixed region recovery arm")
    spec = native_released_training.contract(classes, seed, arm, factor, producer, numeric=numeric)
    return {**spec, "format": RECOVERY, "region_operation": operation,
            "intervention_schedule": "fixed_region_zero_through_train_and_devval_or_off"}
