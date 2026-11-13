"""Block-local resolution degradation on a fixed supervised pixel lattice."""

from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
import math

import numpy as np

from loadData.observation_sampling import ObservationView
from model.research_config import digest
from transfer.development_blocks import array_hash
from transfer.g3_training import Scene


MODES = ("identity", "gaussian", "area2", "area4")


def transform_spec(mode):
    if mode == "identity":
        return {"method": "identity"}
    if mode == "gaussian":
        return {"method": "separable_gaussian", "sigma_pixels": 1.0, "radius_pixels": 3, "boundary": "edge"}
    if mode in ("area2", "area4"):
        return {"method": "area_cell_mean_nearest", "factor": int(mode[-1]), "partial_cells": "retain"}
    raise ValueError("Unknown spatial observation mode")


@dataclass(frozen=True)
class SpatialObservation:
    scene: Scene
    tile: dict
    view: ObservationView
    _receipt: dict
    _scene_identity: tuple

    @property
    def receipt(self):
        return deepcopy(self._receipt)

    @property
    def observation_id(self):
        return self._receipt["observation"]

    def verify(self, condition):
        obs = self.view.observation
        if (condition is not None and self._receipt["profile"] != "synthetic"):
            raise ValueError("Physical intervention requires a synthetic observation")
        partition = {"devtrain": "train", "devval": "val"}[self._receipt["role"]]
        scene_identity = (self.scene.dataset, self.scene.classes, self.scene.plan["plan_hash"])
        if (scene_identity != self._scene_identity or obs.plan_hash != self._scene_identity[-1]
                or self.tile["id"] != self._receipt["tile"] or self.tile["box"] != self._receipt["box"]
                or self.tile["partition"] != partition or obs.partition != partition
                or self.view.bands.fingerprint != self._receipt["band_order"]
                or array_hash(obs.spectral) != self._receipt["transformed"]
                or array_hash(obs.coordinates) != self._receipt["coordinates"]
                or array_hash(obs.global_ids) != self._receipt["ids"]):
            raise ValueError("Materialized spatial observation changed after admission")

    def bank_view(self):
        return self.scene, self.tile, self.view, self.receipt


@lru_cache(maxsize=32)
def _cells(length, factor):
    return tuple(range(0, length, factor))


def degrade(values, mode):
    """Gaussian replicates block edges; area uses fixed anchored partial cells."""
    if (not isinstance(mode, str) or mode not in MODES or not isinstance(values, np.ndarray)
            or values.dtype != np.float32 or values.ndim != 3 or min(values.shape) == 0
            or not np.isfinite(values).all()):
        raise ValueError("Finite nonempty FP32 HWC and registered spatial mode required")
    if mode == "identity":
        return values
    if mode == "gaussian":
        weights = np.array([math.exp(-x*x/2) for x in range(-3, 4)], dtype=np.float64)
        weights /= weights.sum()
        out = values.astype(np.float64)
        for axis in (0, 1):
            indexes = np.arange(values.shape[axis])
            out = sum(weight * np.take(out, np.clip(indexes + shift, 0, len(indexes)-1), axis=axis)
                      for shift, weight in zip(range(-3, 4), weights, strict=True))
        return out.astype(np.float32)
    factor = transform_spec(mode)["factor"]
    y, x = (_cells(length, factor) for length in values.shape[:2])
    total = np.add.reduceat(np.add.reduceat(values.astype(np.float64), y, axis=0), x, axis=1)
    ny, nx = (np.minimum(factor, length-np.asarray(starts)) for length, starts in zip(values.shape, (y,x)))
    pooled = total / (ny[:, None, None] * nx[None, :, None])
    return pooled[np.arange(values.shape[0])[:, None]//factor,
                  np.arange(values.shape[1])[None, :]//factor].astype(np.float32)


def materialize(bank, row, bands, mode, producer):
    transform = transform_spec(mode)
    if not isinstance(producer, str) or not producer:
        raise ValueError("Spatial observation producer required")
    scene, tile, view, original = bank.materialize(row, bands)
    old = view.observation
    # The bank has already retained the actual band subset and applied frozen per-band statistics.
    spectral = degrade(old.spectral, mode)
    spectral.setflags(write=False)
    if mode != "identity":
        # Bind the original retained block and the actually transformed observation independently.
        view = replace(view, observation=replace(old, spectral=spectral))
    body = dict(original, profile=bank.release.profile, spatial=mode, transform=transform, producer=producer,
                order="retain_bands_then_frozen_normalization_then_spatial",
                transformed=array_hash(spectral), lattice="original_coordinates_and_ids",
                physical_scale="unverified_no_gsd_conversion")
    body["observation"] = digest(body)
    return SpatialObservation(scene, tile, view, deepcopy(body),
                              (scene.dataset, scene.classes, scene.plan["plan_hash"]))


def observability(row, mode, scene_width, classes):
    """Count released labels in each cell; every other pixel is unknown."""
    y0,y1,x0,x1 = row["box"]
    h,w = y1-y0,x1-x0
    factor = transform_spec(mode).get("factor", 1)
    cells = {(y//factor,x//factor): [] for y in _cells(h,factor) for x in _cells(w,factor)}
    for pixel,label in row["samples"]:
        y,x = divmod(pixel,scene_width)
        if not (y0 <= y < y1 and x0 <= x < x1 and 1 <= label <= classes):
            raise ValueError("Released sample escapes spatial lattice")
        cells[(y-y0)//factor,(x-x0)//factor].append(label)
    known = [0]*classes
    conflicts = [0]*classes
    out = []
    for (y,x), labels in cells.items():
        count = len(labels)
        if count:
            for label in labels:
                known[label-1] += 1
                if len(set(labels)) > 1:
                    conflicts[label-1] += 1
        pixels = min(factor,h-y*factor)*min(factor,w-x*factor)
        out.append((pixels,count,len(set(labels))))
    # Gaussian has overlapping receptive fields, no disjoint coarse cells.
    if mode == "gaussian":
        return dict(lattice="original", coarse_cells=None, known_per_class=known, conflict_per_class=None,
                    unknown_pixels=h*w-sum(known), pixels=h*w)
    return dict(lattice="anchored_cells", coarse_cells=len(out), known_per_class=known,
                conflict_per_class=conflicts, unknown_pixels=h*w-sum(known), pixels=h*w,
                known_count_histogram=np.bincount([n for _,n,_ in out]).tolist(),
                conflicting_cells=sum(c>1 for _,_,c in out),
                fully_known_cells=sum(n==size for size,n,_ in out))
