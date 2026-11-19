"""Frozen native representations and support-only readouts on registered blocks."""

from contextlib import ExitStack
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from evaluation.frozen_training_stages import HEADS, probabilities, require
from model.region_pixel_alignment import broadcast_regions


LAYERS = ("sensor", "vca", "pre_pool", "pooled", "cnn", "graph", "fused")
SHARED_LAYERS = ("vca", "pooled", "fused")


def as_pixels(value):
    return value.detach().reshape(-1, value.shape[-1]).cpu().numpy()


def moment(x):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(0)
    return len(x), mean, float(np.square(x - mean).sum())


def combine_moments(values):
    count = sum(n for n, _, _ in values)
    mean = sum(n * mu for n, mu, _ in values) / count
    within = sum(ss for _, _, ss in values) / count
    between = sum(n * np.square(mu - mean).sum() for n, mu, _ in values) / count
    return dict(pixels=count, blocks=len(values), within_block=within,
                between_block=float(between), total=float(within + between))


def assignment_variance(q, pixels, pooled):
    """Decompose weighted pixel variance over the actual sparse Q memberships."""
    q = q.detach().to_sparse_coo().coalesce().cpu()
    row, col = q.indices().numpy()
    weight = q.values().double().numpy()
    x = np.asarray(pixels, dtype=np.float64)
    row_weight = np.bincount(row, weights=weight, minlength=q.shape[0])
    mass = np.bincount(col, weights=weight, minlength=q.shape[1])
    require(q.shape[0] == len(x) and np.isfinite(weight).all() and (weight >= 0).all()
            and np.allclose(row_weight, 1, rtol=0, atol=2e-5), "Invalid native Q membership")
    centers = np.zeros((q.shape[1], x.shape[1]), dtype=np.float64)
    np.add.at(centers, col, weight[:, None] * x[row])
    nonempty = mass > 0
    centers[nonempty] /= mass[nonempty, None]
    require(pooled.shape == centers.shape and np.allclose(pooled[nonempty], centers[nonempty],
            rtol=2e-4, atol=3e-5), "Captured pooling does not match the actual pre-pooling tensor")
    count = mass.sum()
    mean = (mass @ centers) / count
    within = weight @ np.square(x[row] - centers[col]).sum(1) / count
    between = mass @ np.square(centers - mean).sum(1) / count
    total = row_weight @ np.square(x - mean).sum(1) / count
    require(np.isclose(total, within + between, rtol=1e-10, atol=1e-12), "Q variance is inconsistent")
    maxima = np.zeros(len(x))
    np.maximum.at(maxima, row, weight)
    hard = np.full(len(x), q.shape[1], dtype=np.int64)
    tied = weight == maxima[row]
    np.minimum.at(hard, row[tied], col[tied])
    return dict(pixels=len(x), regions=int(q.shape[1]), nonempty_regions=int(nonempty.sum()),
                occupied_hard_regions=int(len(np.unique(hard))), nnz=len(weight),
                mean_max_assignment=float(maxima.mean()), within_region=float(within),
                between_region=float(between), total=float(total))


def capture(model, spectral, forward, *, layers=LAYERS):
    """Attach read-only hooks to the original calculation, before any readout fit."""
    backbone = model.backbone
    features = {}
    state = {}
    stages, fuse = backbone.forward_stages, backbone.fuse_features

    def adapter(_module, _args, output):
        features["vca"] = output[0] if isinstance(output, tuple) else output

    def clean(_module, _args, output):
        features["pre_pool"] = output[0].permute(1, 2, 0)

    def forward_stages(cnn_input, region_input, q, a, *args, **kwargs):
        state.update(q=q, regions=region_input)
        cnn, graph, extra = stages(cnn_input, region_input, q, a, *args, **kwargs)
        features.update(cnn=cnn, graph=graph)
        return cnn, graph, extra

    def fuse_features(cnn, graph):
        value = fuse(cnn, graph)
        features["fused"] = value
        return value

    device = next(model.parameters()).device

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode(), ExitStack() as stack:
        stack.callback(backbone.adapter.register_forward_hook(adapter).remove)
        if "pre_pool" in layers:
            stack.callback(backbone.CNN_denoise[-1].register_forward_hook(clean).remove)
        stack.enter_context(patch.object(backbone, "forward_stages", new=forward_stages))
        stack.enter_context(patch.object(backbone, "fuse_features", new=fuse_features))
        sync()
        start = perf_counter()
        output, graph = forward()
        sync()
        seconds = perf_counter() - start
        features["pooled"] = broadcast_regions(state["q"], state["regions"])
        if "sensor" in layers:
            features["sensor"] = torch.as_tensor(np.array(spectral, copy=True))
        values = {key: as_pixels(features[key]) for key in layers}
        n = spectral.shape[0] * spectral.shape[1]
        require(all(len(value) == n and np.isfinite(value).all() for value in values.values()),
                "Native feature capture lost pixel alignment")
        q = dict(pixels=n) if "pre_pool" not in values else assignment_variance(
            state["q"], values["pre_pool"], as_pixels(state["regions"]))
        heads = {head: as_pixels(value) for head, value in zip(HEADS, output[:3], strict=True)}
    return values, heads, q, graph, seconds


class ReleasedBlocks:
    def __init__(self, role, classes, *, layers=LAYERS):
        require(role in ("devtrain", "devval"), "Only released development roles are admissible")
        self.role = role
        self.classes = classes
        self.identities = []
        self.truth = []
        self.labels = {layer: [] for layer in layers}
        self.moments = {layer: [] for layer in layers} if layers == LAYERS else {}
        self.q = []
        self.seconds = 0.0

    def add(self, block, cells, identities, truth, data):
        values, heads, q, _, seconds = data
        cells, truth = np.asarray(cells, dtype=np.int64), np.asarray(truth, dtype=np.int64)
        require(len(cells) == len(identities) == len(truth)
                and ((cells >= 0) & (cells < q["pixels"])).all()
                and ((truth >= 1) & (truth <= self.classes)).all(), "Invalid released sample binding")
        self.identities.extend(identities)
        self.truth.extend(truth.tolist())
        for key in self.labels:
            self.labels[key].append(values[key][cells])
        for key in self.moments:
            self.moments[key].append(moment(values[key]))
        self.q.append(dict(block=block, **q))
        self.seconds += seconds
        return {name: probabilities(value, q["pixels"], self.classes)[cells] for name, value in heads.items()}

    def summary(self):
        require(len(self.identities) == len(set(self.identities)), "Repeated released labels")
        return dict(role=self.role, label_counts=np.bincount(self.truth, minlength=self.classes+1)[1:].tolist(),
                    q=self.q, moments={key: combine_moments(v) for key, v in self.moments.items()},
                    inference_seconds=self.seconds)


def point_metrics(values, class_ids, truth, rivals):
    """Keep the true class, fixed rival and actual highest competing class separate."""
    require(len(class_ids) >= 2 and np.isfinite(values).all(), "Invalid readout scores")
    columns = {cls: index for index, cls in enumerate(class_ids)}
    result = []
    for row, label in zip(values, truth, strict=True):
        order = np.argsort(-row, kind="stable")
        predicted = int(class_ids[order[0]])
        top_other = int(next(class_ids[index] for index in order if class_ids[index] != label))
        rival = rivals.get(label)
        margin = float(row[columns[label]] - row[columns[rival]]) if label in columns and rival in columns else None
        result.append(dict(predicted=predicted, highest_other=top_other, fixed_rival=rival,
                           margin=margin, true_class_fitted=label in columns))
    return result


def readout(train, development, rivals):
    require(train.role == "devtrain" and development.role == "devval"
            and set(train.identities).isdisjoint(development.identities), "Readout support and scoring roles overlap")
    y = np.asarray(train.truth)
    classes = np.unique(y)
    require(len(classes) >= 2, "Need at least two support classes")
    targets = (y[:, None] == classes).astype(np.float64)
    output = {}
    require(set(development.labels).issubset(train.labels), "Readout layers differ")
    for key in development.labels:
        x, val = (np.concatenate(role.labels[key]).astype(np.float64) for role in (train, development))
        require(x.shape[1] == val.shape[1], "Fixed readout requires a common feature dimension")
        mean, std = x.mean(0), x.std(0)
        scale = np.where(std > 1e-10, std, 1.0)
        x, val = (x - mean) / scale, (val - mean) / scale
        centers = np.asarray([x[y == cls].mean(0) for cls in classes])
        centroid = -np.square(val[:, None] - centers).mean(2)
        base = targets.mean(0)
        ridge = (val @ x.T) @ np.linalg.solve(x @ x.T + np.eye(len(x)), targets - base) + base
        output[key] = {method: point_metrics(value, classes, development.truth, rivals)
                       for method, value in (("centroid", centroid), ("ridge", ridge))}
    return dict(fit_class_ids=classes.tolist(), ridge_penalty=1.0,
                scale_floor=1e-10, classes_missing_from_fit=sorted(set(development.truth)-set(classes.tolist())),
                layers=output)
