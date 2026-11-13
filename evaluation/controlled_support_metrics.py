"""Released-label paired error migration for frozen-support counterfactuals."""

from collections import Counter

import numpy as np

from evaluation.classification_metrics import metrics_from_confusion
from transfer.xuzhou_multiscale_audit import HEADS


def released_samples(record, tile):
    y0, y1, x0, x1 = tile["box"]
    if tile["role"] not in ("devtrain", "devval"):
        raise ValueError("Only released development support labels are admitted")
    group = "train_support" if tile["role"] == "devtrain" else "validation_support"
    release = record[group]
    width = record["shape"][1]
    rows = []
    for identity, label in zip(release["indices"], release["labels"], strict=True):
        y, x = divmod(identity, width)
        if y0 <= y < y1 and x0 <= x < x1:
            rows.append((identity, (y - y0) * (x1 - x0) + x - x0, label))
    return rows


def paired_metrics(record, role, before, after):
    samples = released_samples(record, role)
    classes = record["classes"]
    if not samples:
        return dict(count=0, heads={}, admission="NOT_ASSESSED")
    training = Counter(record["train_support"]["labels"])
    # The study has no candidate gate. Least represented released training
    # classes (ties retained) are reported as a descriptive cohort only.
    smallest = {key for key, n in training.items() if n == min(training.values())}
    outcomes = {}
    for name, a, b in zip(HEADS, before[:3], after[:3], strict=True):
        ids, positions, labels = map(np.asarray, zip(*samples, strict=True))
        old = a.detach().cpu().numpy()[positions].argmax(1) + 1
        new = b.detach().cpu().numpy()[positions].argmax(1) + 1
        reference = np.zeros((len(classes), len(classes)), dtype=np.int64)
        alternative = np.zeros_like(reference)
        np.add.at(reference, (labels - 1, old - 1), 1)
        np.add.at(alternative, (labels - 1, new - 1), 1)
        migrations = []
        counts = Counter()
        for identity, truth, previous, current in zip(ids.tolist(), labels.tolist(), old.tolist(), new.tolist(), strict=True):
            relation = ("new_error" if previous == truth and current != truth else
                        "corrected" if previous != truth and current == truth else
                        "changed_error" if previous != truth and current != truth and previous != current else
                        "unchanged")
            migrations.append(dict(sample=identity, truth=truth, reference=previous, intervention=current,
                                   status=relation, least_represented=truth in smallest))
            counts[relation] += 1
        outcomes[name] = dict(reference=metrics_from_confusion(reference, classes),
                              intervention=metrics_from_confusion(alternative, classes),
                              migrations=migrations, counts=counts,
                              least_represented_new_errors=sum(row["least_represented"] and row["status"] == "new_error"
                                                               for row in migrations))
    return dict(count=len(samples), heads=outcomes, least_represented_classes=sorted(smallest),
                cohort_rule="lowest_released_training_count_all_ties", admission="NOT_ASSESSED")
