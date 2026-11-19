"""Paired descriptive metrics for identical released development points."""

from collections import Counter, defaultdict
from statistics import fmean, stdev

import numpy as np

from evaluation.classification_metrics import classification_metrics


HEADS = ("fused", "graph", "cnn")


def _summary(matrix):
    return {str(index + 1): int(count) for index, count in enumerate(matrix) if count}


def paired(target, source, labels, classes):
    """Compare hard predictions indexed by a separately authenticated population."""
    target, source, labels = (np.asarray(values, dtype=np.int64)
                              for values in (target, source, labels))
    order = list(range(1, classes + 1))
    old, new = (classification_metrics(labels, prediction, order)
                for prediction in (source, target))
    lost = (source == labels) & (target != labels)
    gained = (source != labels) & (target == labels)
    rows = []
    for cls in order:
        mask = labels == cls
        transitions = Counter(zip(source[mask].tolist(), target[mask].tolist(), strict=True))
        rows.append(dict(
            class_id=cls, denominator=int(mask.sum()),
            source_correct=int(((source == cls) & mask).sum()),
            target_correct=int(((target == cls) & mask).sum()),
            source_recall=old["per_class"][cls - 1]["recall"],
            target_recall=new["per_class"][cls - 1]["recall"],
            correct_to_error=int((lost & mask).sum()), error_to_correct=int((gained & mask).sum()),
            source_predictions=_summary(old["confusion_matrix"][cls - 1]),
            target_predictions=_summary(new["confusion_matrix"][cls - 1]),
            lost_to=dict(sorted(Counter(target[lost & mask].tolist()).items())),
            gained_from=dict(sorted(Counter(source[gained & mask].tolist()).items())),
            transitions=[[a, b, count] for (a, b), count in sorted(transitions.items())],
        ))
    return dict(denominator=len(labels), flips=int((target != source).sum()),
                correct_to_error=int(lost.sum()), error_to_correct=int(gained.sum()),
                source_oa=old["oa"], target_oa=new["oa"], source_aa=old["aa"], target_aa=new["aa"],
                delta_oa=new["oa"] - old["oa"] if len(labels) else None,
                delta_aa=new["aa"] - old["aa"] if len(labels) else None,
                source_confusion=old["confusion_matrix"], target_confusion=new["confusion_matrix"],
                classes=rows)


def hard(probabilities, count, classes):
    values = np.asarray(probabilities)
    if (values.shape != (count, classes) or not np.isfinite(values).all()
            or (values < 0).any() or not np.allclose(values.sum(1), 1, atol=2e-6, rtol=2e-6)):
        raise ValueError("Expected one finite normalized probability row for every released point")
    return (values.argmax(1) + 1).tolist()


def comparisons(predictions, labels, classes, pairs, *, group_masks=None):
    if not predictions or any(set(heads) != set(HEADS) for heads in predictions.values()):
        raise ValueError("Expected all three heads in every condition")
    output = {}
    for name, (target, source) in pairs.items():
        output[name] = {head: paired(predictions[target][head], predictions[source][head],
                                     labels, classes) for head in HEADS}
        for group, mask in (group_masks or {}).items():
            output[f"{name}/{group}"] = {
                head: paired(np.asarray(predictions[target][head])[mask],
                             np.asarray(predictions[source][head])[mask], np.asarray(labels)[mask],
                             classes) for head in HEADS}
    return output


def summarize(cases):
    """Compute sample SD only within one explicitly matched model-seed cohort."""
    groups = defaultdict(list)
    for case in cases:
        groups[(case["scene"], case["cohort"], case["population"])].append(case)
    output = []
    for (scene, cohort, population), group in sorted(groups.items()):
        seeds = [row["model_seed"] for row in group]
        if len(set(seeds)) != len(seeds) or len({tuple(sorted(row["paired"])) for row in group}) != 1:
            raise ValueError("Cohort contains duplicated seeds or incompatible comparisons")
        for comparison in group[0]["paired"]:
            for head in HEADS:
                rows = [row["paired"][comparison][head] for row in group]
                stats = {}
                for key in ("source_oa", "target_oa", "source_aa", "target_aa", "delta_oa", "delta_aa"):
                    values = [row[key] for row in rows if row[key] is not None]
                    stats[key] = dict(mean=fmean(values) if values else None,
                                      sample_sd=stdev(values) if len(values) > 1 else None,
                                      n=len(values))
                class_stats = []
                for aligned in zip(*(row["classes"] for row in rows), strict=True):
                    base = aligned[0]
                    if any((row["class_id"], row["denominator"]) !=
                           (base["class_id"], base["denominator"]) for row in aligned):
                        raise ValueError("Class populations differ inside the seed cohort")
                    gains = [row["target_recall"] - row["source_recall"] for row in aligned
                             if row["source_recall"] is not None]
                    class_stats.append(dict(
                        class_id=base["class_id"], denominator_per_seed=base["denominator"],
                        mean_delta_recall=fmean(gains) if gains else None,
                        sample_sd_delta_recall=stdev(gains) if len(gains) > 1 else None,
                        correct_to_error=sum(row["correct_to_error"] for row in aligned),
                        error_to_correct=sum(row["error_to_correct"] for row in aligned),
                    ))
                output.append(dict(scene=scene, cohort=cohort, population=population,
                                   model_seeds=seeds, comparison=comparison, head=head, metrics=stats,
                                   classes=class_stats))
    return output
