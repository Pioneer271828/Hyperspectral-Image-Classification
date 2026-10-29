"""Explicit class order, supported-class macro averages, JSON-safe undefined values."""
import numpy as np


def class_order(class_ids):
    ids = list(class_ids)
    if (not ids or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                       or v <= 0 or v > np.iinfo(np.int64).max for v in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('class_ids must be distinct positive integers in fixed reporting order')
    return [int(v) for v in ids]


def integer_values(values, name):
    values = np.asarray(values)
    if values.dtype.kind not in 'iuf' or not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(name + ' must contain finite integers')
    if values.size and (int(values.min()) < 0 or int(values.max()) > np.iinfo(np.int64).max):
        raise ValueError(name + ' must contain nonnegative int64 values')
    return values.astype(np.int64, copy=False)


def evaluation_mask(labels, mask=None):
    labels = np.asarray(labels)
    if labels.ndim not in (1, 2):
        raise ValueError('labels must be a vector or 2D map')
    if mask is None:
        mask = np.ones(labels.shape, dtype=bool)
    mask = np.asarray(mask)
    if mask.shape != labels.shape or mask.dtype != np.bool_:
        raise ValueError('Evaluation mask must be a matching boolean array')
    integer_values(labels[mask], 'Selected labels')
    return mask & (labels > 0)


def class_indices(values, class_ids, name):
    values = integer_values(values, name)
    ids = np.asarray(class_order(class_ids), dtype=np.int64)
    order = np.argsort(ids)
    sorted_ids = ids[order]
    positions = np.searchsorted(sorted_ids, values)
    safe = np.minimum(positions, len(ids) - 1)
    if ((positions >= len(ids)) | (sorted_ids[safe] != values)).any():
        raise ValueError(name + ' contain IDs outside declared classes')
    return order[positions]


def confusion_matrix(labels, predictions, class_ids, mask=None):
    ids = class_order(class_ids)
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if predictions.shape != labels.shape:
        raise ValueError('Prediction shape differs from labels; pass hard label IDs, not scores')
    valid = evaluation_mask(labels, mask)
    truth = class_indices(labels[valid], ids, 'Labels')
    predicted = class_indices(predictions[valid], ids, 'Predictions')
    return np.bincount(truth * len(ids) + predicted, minlength=len(ids) ** 2).reshape(len(ids), len(ids))


def metrics_from_confusion(matrix, class_ids):
    ids = class_order(class_ids)
    matrix = np.asarray(matrix)
    if matrix.shape != (len(ids), len(ids)):
        raise ValueError('Confusion matrix shape differs from declared classes')
    matrix = integer_values(matrix, 'Confusion counts')
    support, predicted = matrix.sum(1), matrix.sum(0)
    true_positive = np.diag(matrix)
    count = int(matrix.sum())
    per_class = []
    recalls, f1s, included, missing = [], [], [], []
    for i, class_id in enumerate(ids):
        precision = float(true_positive[i] / predicted[i]) if predicted[i] else None
        recall = float(true_positive[i] / support[i]) if support[i] else None
        f1 = float(2 * true_positive[i] / (support[i] + predicted[i])) if support[i] else None
        if support[i]:
            included.append(class_id); recalls.append(recall); f1s.append(f1)
        else:
            missing.append(class_id)
        per_class.append(dict(class_id=class_id, support=int(support[i]), predicted_count=int(predicted[i]),
                              precision=precision, recall=recall, f1=f1))
    oa = float(true_positive.sum() / count) if count else None
    expected = float(np.dot(support.astype(float), predicted.astype(float)) / count ** 2) if count else None
    kappa = (oa - expected) / (1.0 - expected) if count and abs(1.0 - expected) > 1e-15 else None
    aa, macro_f1 = (float(np.mean(recalls)), float(np.mean(f1s))) if included else (None, None)
    return dict(class_ids=ids, confusion_matrix=matrix.tolist(), count=count, oa=oa, aa=aa, kappa=kappa,
                macro_f1=macro_f1, macro_class_ids=included, missing_class_ids=missing,
                macro_policy='ground_truth_supported_classes', full_class_coverage=not missing,
                aa_full=aa if not missing else None, macro_f1_full=macro_f1 if not missing else None,
                per_class=per_class)


def classification_metrics(labels, predictions, class_ids, mask=None):
    return metrics_from_confusion(confusion_matrix(labels, predictions, class_ids, mask), class_ids)
