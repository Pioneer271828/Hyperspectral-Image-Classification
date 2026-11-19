"""Default-off, training-support-only readout for a frozen three-head model."""

from dataclasses import dataclass

import torch

from evaluation.frozen_band_region import HEADS

SCHEMA = "frozen_support_shrinkage_v1"
BRANCHES = ("cnn", "graph")


def tensor(value, device=None):
    return torch.as_tensor(value, dtype=torch.float64, device=device).contiguous()


def probabilities(value, device=None):
    result = tensor(value, device)
    if (result.ndim != 2 or not torch.isfinite(result).all() or (result < 0).any()
            or not torch.allclose(result.sum(1), torch.ones_like(result[:, 0]), atol=2e-6, rtol=2e-6)):
        raise ValueError("Expected finite normalized probabilities [points, classes]")
    return result


def distance(centers, points):
    values = tensor(points, centers.device)
    if (values.ndim != 2 or values.shape[1] != centers.shape[1] or not torch.isfinite(values).all()):
        raise ValueError("Frozen feature width and finiteness must match support")
    return (values[:, None] - centers[None]).square().sum(dim=-1)


def support_weights(train_probabilities, labels, classes, device=None):
    truth = torch.as_tensor(labels, dtype=torch.long, device=device)
    if truth.ndim != 1 or not len(truth) or not ((truth >= 1) & (truth <= classes)).all():
        raise ValueError("Expected declared one-based support labels")
    scores = []
    for head in BRANCHES:
        p = probabilities(train_probabilities[head], truth.device)
        if p.shape != (len(truth), classes):
            raise ValueError("Training scores and support labels differ")
        correct = p.argmax(1) + 1 == truth
        per_class = [(correct[truth == cls].sum().double() + 1.) / ((truth == cls).sum().double() + 2.)
                     for cls in truth.unique()]
        scores.append(torch.stack(per_class).mean())
    reliability = torch.stack(scores)
    return reliability / reliability.sum()


def support_scale(between, counts):
    valid = (counts[:, None] > 0) & (counts[None, :] > 0) & (between > 1e-12)
    values = between[torch.triu(valid, diagonal=1)]
    if not len(values):
        return between.new_zeros(())
    return torch.quantile(values, .5)


def apply_distance(native, distances, counts, scale):
    p = probabilities(native)
    d, n, spread = (tensor(v, p.device) for v in (distances, counts, scale))
    if (d.shape != p.shape or n.shape != p.shape[1:] or spread.ndim != 0
            or not torch.isfinite(d).all() or (d < 0).any()
            or not torch.isfinite(n).all() or (n < 0).any() or (n != n.floor()).any()
            or not torch.isfinite(spread) or spread < 0):
        raise ValueError("Distance, support count and scale must match the native scores")
    mask = n > 0
    if mask.sum() < 2 or spread <= 1e-12:
        return p
    relative = (d[:, mask].mean(1, keepdim=True) - d) / spread
    correction = relative.clamp(-3., 3.) * n / (n + 1)
    # Preserve zero-probability entries and do not invent class evidence for missing support.
    weights = p * correction.exp()
    return weights / weights.sum(1, keepdim=True)


def output_scores(native, distances, counts, scales, weights):
    result = {head: apply_distance(native[head], distances[head], counts, scales[head]) for head in HEADS}
    w = tensor(weights, result["fused"].device)
    if w.shape != (2,) or not torch.isfinite(w).all() or (w < 0).any() or abs(w.sum() - 1.) > 1e-10:
        raise ValueError("Expected normalized train-only CNN and graph weights")
    if (tensor(counts, w.device) > 0).sum() >= 2 and all(float(scales[h]) > 1e-12 for h in HEADS):
        pooled = sum(w[i] * result[head] for i, head in enumerate(BRANCHES))
        result["fused"] = (result["fused"] + pooled) * .5
    return result


@dataclass(frozen=True)
class SupportReadout:
    binding: dict
    centers: dict
    counts: torch.Tensor
    scales: dict
    weights: torch.Tensor

    @classmethod
    def fit(cls, binding, features, train_probabilities, labels, classes):
        if set(features) != set(HEADS) or set(train_probabilities) != set(HEADS):
            raise ValueError("Expected all three released support heads")
        first = tensor(features["fused"])
        truth = torch.as_tensor(labels, device=first.device)
        if (truth.ndim != 1 or truth.dtype not in (torch.int32, torch.int64) or not len(truth)
                or not ((truth >= 1) & (truth <= classes)).all()):
            raise ValueError("Expected nonempty one-based integer support labels")
        counts = torch.bincount(truth, minlength=classes + 1)[1:]
        centers, scales = {}, {}
        for head in HEADS:
            values = tensor(features[head], first.device)
            if (values.ndim != 2 or len(values) != len(truth) or not values.shape[1]
                    or not torch.isfinite(values).all()):
                raise ValueError("Support feature and label roster differs")
            means = values.new_zeros((classes, values.shape[1]))
            means.index_add_(0, truth.long() - 1, values)
            means /= counts.clamp_min(1)[:, None]
            centers[head], scales[head] = means, support_scale(distance(means, means), counts)
        return cls.from_state(dict(schema=SCHEMA, binding=binding, counts=counts, centers=centers,
                                   scales=scales, weights=support_weights(train_probabilities, truth, classes, first.device)),
                              binding, device=first.device)

    def state(self):
        return dict(schema=SCHEMA, binding=dict(self.binding), counts=self.counts.cpu().clone(),
                    centers={key: value.cpu().clone() for key, value in self.centers.items()},
                    scales={key: value.cpu().clone() for key, value in self.scales.items()},
                    weights=self.weights.cpu().clone())

    @classmethod
    def from_state(cls, state, binding, device="cpu"):
        if (set(state) != {"schema", "binding", "counts", "centers", "scales", "weights"}
                or state["schema"] != SCHEMA or state["binding"] != binding
                or set(binding) != {"scene", "checkpoint", "observation", "support"}
                or any(not isinstance(value, str) or not value for value in binding.values())
                or set(state["centers"]) != set(HEADS) or set(state["scales"]) != set(HEADS)):
            raise ValueError("Support packet binding or schema differs")
        counts = tensor(state["counts"], device).clone()
        weights = tensor(state["weights"], device).clone()
        if (counts.ndim != 1 or len(counts) < 2 or counts.sum() <= 0 or (counts != counts.floor()).any()
                or not torch.isfinite(counts).all() or (counts < 0).any() or weights.shape != (2,)
                or not torch.isfinite(weights).all() or (weights < 0).any() or abs(weights.sum() - 1.) > 1e-10):
            raise ValueError("Support packet counts or weights differ")
        centers = {head: tensor(state["centers"][head], device).clone() for head in HEADS}
        scales = {head: tensor(state["scales"][head], device).clone() for head in HEADS}
        for head, value in centers.items():
            if (value.ndim != 2 or value.shape[0] != len(counts) or not value.shape[1]
                    or not torch.isfinite(value).all() or not torch.equal(
                        support_scale(distance(value, value), counts), scales[head])):
                raise ValueError("Support packet features or distance scale differ")
        return cls(dict(binding), centers, counts, scales, weights)

    def predict(self, native, features, binding):
        if binding != self.binding or set(native) != set(HEADS) or set(features) != set(HEADS):
            raise ValueError("Inference binding or heads differ from the training support")
        distances = {head: distance(self.centers[head], features[head]) for head in HEADS}
        return output_scores(native, distances, self.counts, self.scales, self.weights)
