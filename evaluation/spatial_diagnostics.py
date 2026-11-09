"""Frozen descriptive spatial support; geometry access is separate from scoring."""
import numpy as np

from .classification_metrics import class_order, class_indices, integer_values
from .spatial_metrics import boundary_bands


def spatial_support(labels, class_ids, *, geometry_mask, score_mask, widths=(1, 2, 3)):
    """Return counts only, never infer true interiors from sparse annotations.

    Both masks are explicit label-access contracts. Labels outside geometry are
    ignored; score_mask must be contained in geometry_mask. Width one contains
    incident pixels of observed four-neighbor class transitions. Wider bands
    propagate through observed labeled geometry only, including across classes.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2 or not all(labels.shape):
        raise ValueError('A nonempty 2D label map is required')
    ids = class_order(class_ids)
    geometry, score = np.asarray(geometry_mask), np.asarray(score_mask)
    for mask in (geometry, score):
        if mask.dtype != np.bool_ or mask.shape != labels.shape:
            raise ValueError('Masks must be matching boolean maps')
    if (score & ~geometry).any():
        raise ValueError('Scoring access must be contained in geometry access')
    safe = np.zeros(labels.shape, dtype=np.int64)
    safe[geometry] = integer_values(labels[geometry], 'Geometry labels')
    class_indices(safe[geometry & (safe > 0)], ids, 'Geometry labels')
    valid = geometry & (safe > 0)
    selected = score & valid

    def counts(mask):
        support = [int(np.count_nonzero(mask & (safe == c))) for c in ids]
        present = [c for c, n in zip(ids, support) if n]
        return {'count': sum(support), 'class_ids': ids, 'per_class_count': support,
                'present_class_ids': present, 'missing_class_ids': [c for c in ids if c not in present],
                'full_class_coverage': len(present) == len(ids)}

    rows = {}
    for width, band in boundary_bands(safe, valid, widths).items():
        boundary = counts(selected & band)
        rows[str(width)] = {'boundary': boundary,
                           'observed_nonboundary': counts(selected & ~band),
                           'boundary_evaluation_available': boundary['count'] > 0}
    return {'schema_version': 1, 'policy': 'labeled_four_neighbor_geodesic_width1_incident_pixels',
            'geometry': counts(valid), 'scoring': counts(selected),
            'geometry_label_count_outside_scoring': int(np.count_nonzero(valid & ~score)),
            'score_unlabeled_count': int(np.count_nonzero(score & (safe == 0))),
            'widths': rows,
            'limitation': 'Observed nonboundary is not certified interior; missing labels can hide transitions.'}
