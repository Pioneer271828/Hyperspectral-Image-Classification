"""Four-neighbor boundaries inside a fixed evaluation mask; no unlabeled background."""
import numpy as np

from .classification_metrics import evaluation_mask, classification_metrics


def boundary_pixels(labels, mask=None):
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError('Spatial diagnostics require a 2D label map')
    valid = evaluation_mask(labels, mask)
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = valid[:, 1:] & valid[:, :-1] & (labels[:, 1:] != labels[:, :-1])
    vertical = valid[1:, :] & valid[:-1, :] & (labels[1:, :] != labels[:-1, :])
    boundary[:, 1:] |= horizontal; boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical; boundary[:-1, :] |= vertical
    return boundary


def boundary_bands(labels, mask=None, widths=(1, 2, 3)):
    labels = np.asarray(labels)
    valid = evaluation_mask(labels, mask)
    widths = tuple(widths)
    if not widths or any(type(w) is not int or w <= 0 for w in widths) or len(set(widths)) != len(widths):
        raise ValueError('Boundary widths must be distinct positive integers')
    current = boundary_pixels(labels, valid)
    result = {}
    # Width 1 means the two incident boundary pixels; each extra width adds one
    # four-neighbor step. Do not propagate through unlabeled or excluded holes.
    for width in range(1, max(widths) + 1):
        if width in widths:
            result[width] = current.copy()
        expanded = current.copy()
        expanded[1:, :] |= current[:-1, :]; expanded[:-1, :] |= current[1:, :]
        expanded[:, 1:] |= current[:, :-1]; expanded[:, :-1] |= current[:, 1:]
        current = expanded & valid
    return result


def spatial_metrics(labels, predictions, class_ids, mask=None, widths=(1, 2, 3)):
    valid = evaluation_mask(labels, mask)
    bands = boundary_bands(labels, valid, widths)
    return {'boundary_policy': 'labeled_four_neighbor_geodesic_width1_incident_pixels',
            'widths': {str(width): {'boundary': classification_metrics(labels, predictions, class_ids, band),
                                   'interior': classification_metrics(labels, predictions, class_ids, valid & ~band)}
                       for width, band in bands.items()}}
