"""De-identified fixed-row metrics and orthogonal mechanism contrasts."""

from statistics import fmean, stdev

import numpy as np

from evaluation.frozen_band_region import HEADS, paired


NATIVE_CONTRASTS = {
    "missing_vs_full": ("clean", "full"),
    "target_vs_matched": ("target", "matched"),
    "target_vs_random": ("target", "random"),
    "graph_only_clean_features": ("clean_target_graph", "clean"),
    "feature_only_clean_graph": ("target_clean_graph", "clean"),
    "joint_vs_clean": ("target", "clean"),
    "graph_only_target_features": ("target", "target_clean_graph"),
    "feature_only_target_graph": ("target", "clean_target_graph"),
}
FULL_INFORMATION_CONTRASTS = {
    "full_graph_counterfactual_clean": ("clean_full_graph", "clean"),
    "full_graph_counterfactual_target": ("target_full_graph", "target_clean_graph"),
}


def validate_probabilities(values, rows, classes):
    values = np.asarray(values)
    if (values.shape != (rows, classes) or values.dtype.kind != "f" or not np.isfinite(values).all()
            or (values < 0).any() or (values > 1).any()
            or not np.allclose(values.sum(axis=1), 1., rtol=2e-6, atol=2e-6)):
        raise ValueError("Expected one normalized distribution per released point in each head")
    return values.astype(np.float64)


def fit_algebraic_temperature(scores, truth):
    """One train-only log-odds readout. No optimizer or model mutation."""
    classes = scores.shape[1]
    values = validate_probabilities(scores, len(truth), classes)
    count = len(values)
    if not count:
        raise ValueError("Temperature needs released training support")
    accuracy = (np.sum(values.argmax(1) + 1 == truth) + 1.) / (count + 2.)
    confidence = (values.max(1).sum() + 1.) / (count + 2.)
    ratio = (confidence / (1. - confidence)) / (accuracy / (1. - accuracy))
    return dict(temperature=float(np.clip(ratio, .5, 4.)), train_count=count,
                smoothed_accuracy=float(accuracy), smoothed_confidence=float(confidence),
                applied_scope="released_train_fit_once_no_parameter_updates")


def apply_temperature(probabilities, temperature):
    if not .5 <= temperature <= 4.:
        raise ValueError("Temperature is outside the frozen readout bounds")
    weights = np.power(probabilities.astype(np.float64), 1. / temperature)
    return weights / weights.sum(axis=1, keepdims=True)


def calibration(values, truth):
    n, classes = values.shape
    if not n:
        return dict(n=0, nll=None, brier=None, ece10=None, bins=[0] * 10)
    p = validate_probabilities(values, n, classes)
    true = p[np.arange(n), np.asarray(truth) - 1]
    confidence, correct = p.max(1), p.argmax(1) + 1 == truth
    bins = np.minimum((confidence * 10).astype(np.int64), 9)
    counts = np.bincount(bins, minlength=10)
    gaps = np.bincount(bins, weights=correct.astype(float)-confidence, minlength=10)
    return dict(n=n, nll=float(-np.log(np.maximum(true, 1e-30)).mean()),
                brier=float((np.square(p).sum(1) - 2*true + 1).mean()),
                ece10=float(np.abs(gaps).sum() / n), bins=counts.tolist())


def point_scores(probabilities, labels, foils):
    """Rank by descending score and ascending output class column on ties."""
    values = validate_probabilities(probabilities, len(labels), probabilities.shape[1])
    order = np.argsort(-values, axis=1, kind="stable")
    ranks = np.argsort(order, axis=1) + 1
    results = []
    for index, label in enumerate(labels):
        row, col = values[index], label - 1
        runner = int(next(k for k in order[index] if k != col))
        comparisons = dict.fromkeys((runner+1, *foils.get(label, ())))
        results.append(dict(
            true_probability=float(row[col]), rank=int(ranks[index, col]), predicted=int(order[index, 0])+1,
            top_probability=float(row[order[index, 0]]), best_nontrue=runner+1,
            true_minus_best=float(row[col]-row[runner]),
            foils={str(k): dict(probability=float(row[k-1]), margin=float(row[col]-row[k-1]),
                                log_odds=float(np.log(max(row[col], 1e-30))-np.log(max(row[k-1], 1e-30))))
                   for k in comparisons},
        ))
    return results


def _margin(values, labels, foils):
    raw = []
    for scores, label in zip(values, labels, strict=True):
        adversary = foils.get(label)
        if adversary:
            alternative = max(scores[k-1] for k in adversary)
        else:
            alternative = max(np.delete(scores, label-1))
        raw.append(scores[label-1] - alternative)
    return np.asarray(raw, dtype=np.float64)


def contrasts(windows, labels, arrays, temperatures, foils, groups):
    """Return protected native comparisons and labelled graph mechanisms."""
    count = len(labels)
    classes = arrays["full"][HEADS[0]].shape[1]
    result = []
    for window in windows:
        window_id = window["id"]
        slots = {"full": arrays["full"], **arrays[window_id]}
        masks = {"all": np.ones(count, dtype=bool), **groups}
        pairs = {**NATIVE_CONTRASTS, **FULL_INFORMATION_CONTRASTS}
        for name, (target, source) in pairs.items():
            if target not in slots or source not in slots:
                continue
            channel = ("counterfactual_full_information" if name in FULL_INFORMATION_CONTRASTS
                       else "native_performance" if name == "missing_vs_full" else "mechanism")
            for head in HEADS:
                a = validate_probabilities(slots[source][head], count, classes)
                b = validate_probabilities(slots[target][head], count, classes)
                warm = temperatures[head]["temperature"]
                t_a, t_b = apply_temperature(a, warm), apply_temperature(b, warm)
                guess_a, guess_b = a.argmax(1)+1, b.argmax(1)+1
                for group, mask in masks.items():
                    aligned = labels[mask]
                    desc = paired(guess_b[mask], guess_a[mask], aligned, classes)
                    raw_margin = _margin(b[mask], aligned, foils)-_margin(a[mask], aligned, foils)
                    calibrated_margin = _margin(t_b[mask], aligned, foils)-_margin(t_a[mask], aligned, foils)
                    for cls in desc["classes"]:
                        selected = aligned == cls["class_id"]
                        cls["delta_margin"] = float(raw_margin[selected].mean()) if selected.any() else None
                        cls["delta_calibrated_margin"] = (
                            float(calibrated_margin[selected].mean()) if selected.any() else None)
                    result.append(dict(window=window_id, contrast=name, head=head, group=group, evidence=channel,
                                       metrics=desc,
                                       before=[calibration(a[mask], aligned), calibration(b[mask], aligned)],
                                       after=[calibration(t_a[mask], aligned), calibration(t_b[mask], aligned)]))
    return result


def interaction(labels, window, foils):
    """Difference of differences for independent graph and feature substitution."""
    out = {}
    for head in HEADS:
        values = {key: _margin(window[key][head], labels, foils)
                  for key in ("clean", "target", "target_clean_graph", "clean_target_graph")}
        diff = values["target"] - values["target_clean_graph"] - values["clean_target_graph"] + values["clean"]
        out[head] = {str(k): dict(n=int((labels == k).sum()), mean=float(diff[labels == k].mean()))
                     for k in sorted(set(labels.tolist()))}
    return out


def leave_one(values):
    """Descriptive held-out summaries for named model seeds or scenes."""
    combined = list(values.values())
    return dict(n=len(combined), mean=fmean(combined) if combined else None,
                sample_sd=stdev(combined) if len(combined) > 1 else None,
                leave_one={key: fmean(v for k, v in values.items() if k != key)
                           for key in values} if len(combined) > 1 else {})
