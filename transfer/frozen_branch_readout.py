"""Intervene at the fixed fusion input and score same-width frozen references."""

import numpy as np
import torch

from evaluation.frozen_band_region import HEADS

BRANCH_TREATMENTS = ("native", "mask_cnn", "mask_graph", "exchange", "restored")


@torch.no_grad()
def fusion_interventions(model, dataset, features):
    cnn, graph = (torch.as_tensor(features[head]) for head in ("cnn", "graph"))
    fixed = model.backbone.alph
    expected = model.backbone.fuse_features(cnn, graph)
    if not torch.equal(expected, torch.as_tensor(features["fused"])):
        raise ValueError("Frozen features do not reconstruct native fusion")
    zero = torch.zeros_like(cnn)
    inputs = ((cnn, graph), (zero, graph), (cnn, zero), (graph, cnn), (cnn, graph))
    layer = model.heads[dataset]
    logits = {key: layer(model.backbone.fuse_features(a, b)).numpy()
              for key, (a, b) in zip(BRANCH_TREATMENTS, inputs, strict=True)}
    if not np.array_equal(logits["native"], logits["restored"]):
        raise ValueError("The native fusion readout could not be restored")
    sides = dict(cnn=(fixed * (cnn @ layer.weight.T)).numpy(),
                 graph=((1 - fixed) * (graph @ layer.weight.T)).numpy(),
                 bias=layer.bias.numpy())
    if not np.allclose(logits["native"], sides["cnn"] + sides["graph"] + sides["bias"], atol=2e-6):
        raise ValueError("Branch logit decomposition differs from the original classifier")
    return logits, sides, dict(cnn=float(fixed), graph=float(1 - fixed))


def centroid_inputs(features, labels, classes):
    """Fit one centroid per class on released support. Validation labels are absent."""
    support = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if (support.ndim != 2 or labels.shape != (len(support),)
            or not np.isfinite(support).all() or set(labels.tolist()) != set(range(1, classes + 1))):
        raise ValueError("Each class requires nonempty finite released support")
    counts = [int(np.count_nonzero(labels == cls)) for cls in range(1, classes + 1)]
    centroids = np.stack([support[labels == cls].mean(axis=0) for cls in range(1, classes + 1)])
    return centroids, counts


def centroid_distances(centroids, validation):
    values = np.asarray(validation, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != centroids.shape[1] or not np.isfinite(values).all():
        raise ValueError("Validation feature width or finiteness differs from the fitted reference")
    return np.square(values[:, None, :] - centroids[None, :, :]).sum(axis=-1)


def centroid_reference(train, labels, validation, classes):
    if set(train) != set(validation) or set(train) != set(HEADS):
        raise ValueError("Three matching frozen feature heads required")
    arrays = {}
    for head in HEADS:
        centroids, counts = centroid_inputs(train[head], labels, classes)
        arrays[f"{head}__distance"] = centroid_distances(centroids, validation[head])
        arrays[f"{head}__between"] = centroid_distances(centroids, centroids)
        arrays[f"{head}__counts"] = np.asarray(counts)
    return arrays
