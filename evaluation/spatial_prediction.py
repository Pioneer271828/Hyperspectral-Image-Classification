"""Posthoc diagnostics on immutable, explicitly indexed prediction packets."""
import numpy as np

from .classification_metrics import integer_values, class_indices, classification_metrics
from .branch_diagnostics import branch_diagnostics
from .spatial_diagnostics import spatial_support
from .spatial_metrics import boundary_bands


def spatial_prediction_report(labels, class_ids, *, pixel_indices, truth, heads,
                              geometry_mask, widths=(1, 2, 3)):
    """Packet labels/predictions are one-based IDs; pixels are sorted C-order.

    No threshold/model selection. Geometry outside scoring must be explicitly
    permitted by the caller. Unlabeled pixels never become negative examples.
    """
    labels = np.asarray(labels)
    indices = integer_values(pixel_indices, 'Pixel indices')
    if labels.ndim != 2 or indices.ndim != 1 or (indices >= labels.size).any():
        raise ValueError('Invalid 2D labels or pixel indices')
    if len(indices) > 1 and not (np.diff(indices) > 0).all():
        raise ValueError('Pixel indices must be strictly increasing')
    truth = np.asarray(truth)
    if truth.shape != indices.shape or not np.array_equal(truth, labels.ravel()[indices]):
        raise ValueError('Packet truth does not match raw indexed labels')
    class_indices(truth, class_ids, 'Packet truth')
    if set(heads) != {'cnn', 'graph', 'fused'}:
        raise ValueError('Exactly cnn/graph/fused predictions are required')
    predictions = {}
    for key, prediction in heads.items():
        prediction = np.asarray(prediction)
        if prediction.shape != truth.shape:
            raise ValueError('Head prediction shape mismatch')
        class_indices(prediction, class_ids, key)
        predictions[key] = prediction
    score = np.zeros(labels.shape, dtype=bool)
    score.ravel()[indices] = True
    support = spatial_support(labels, class_ids, geometry_mask=geometry_mask,
                              score_mask=score, widths=widths)
    safe = np.zeros(labels.shape, dtype=np.int64)
    safe[geometry_mask] = labels[geometry_mask]

    def metrics(selected):
        return {'heads': {h: classification_metrics(truth, p, class_ids, selected)
                          for h, p in predictions.items()},
                'branches': branch_diagnostics(truth, predictions['cnn'], predictions['graph'],
                                               predictions['fused'], class_ids, selected)}

    rows = {}
    for width, band in boundary_bands(safe, geometry_mask, widths).items():
        selected = band.ravel()[indices]
        rows[str(width)] = {'boundary': metrics(selected), 'observed_nonboundary': metrics(~selected)}
    return {'support': support, 'whole_score': metrics(np.ones(truth.shape, dtype=bool)),
            'widths': rows, 'purpose': 'posthoc_descriptive_only'}


def paired_difference(candidate, reference, seeds=(64, 512, 2048)):
    """Pair by exact seed IDs, no silent subset selection or null->zero conversion."""
    seeds = tuple(seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('At least two distinct predefined seeds are required')
    if set(candidate) != set(seeds) or set(reference) != set(seeds):
        raise ValueError('Missing or unexpected paired seed')
    values = [candidate[s] for s in seeds] + [reference[s] for s in seeds]
    if all(v is None for v in values):
        return {'seeds': list(seeds), 'values_pp': [None] * len(seeds),
                'mean_pp': None, 'sample_sd_pp': None, 'positive_seeds': None}
    if any(v is None or isinstance(v, (bool, np.bool_)) or not np.isscalar(v)
           or not np.isfinite(v) or v < 0 or v > 1 for v in values):
        raise ValueError('Metrics must be finite fractions or all undefined')
    delta = np.array([candidate[s] - reference[s] for s in seeds], dtype=float) * 100
    return {'seeds': list(seeds), 'values_pp': delta.tolist(), 'mean_pp': float(delta.mean()),
            'sample_sd_pp': float(delta.std(ddof=1)), 'positive_seeds': int((delta > 0).sum())}
