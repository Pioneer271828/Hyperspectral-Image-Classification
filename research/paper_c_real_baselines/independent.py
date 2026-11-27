"""Independent scalar arithmetic for future sealed endpoints, not a training launcher.

This module imports neither the producer model nor its metric/feature functions.
Cached-feature replay validates the optimizer, not original-image provenance.
The latter remains an additional, separately required acceptance check.
"""
import itertools
import math

import numpy as np


def probability(z):
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def score(theta, row):
    return math.fsum(float(a) * float(b) for a, b in zip(theta[:8], row)) + theta[8]


def replay(X, y, *, seed, steps, start=None):
    """Reference GD using explicit examples/coordinates and math.fsum."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.shape != (len(y), 8) or not len(y) or not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Finite binary eight-feature fitting examples required')
    if type(steps) is not int or not 0 < steps <= 128:
        raise ValueError('Fixed short replay bound')
    theta = list(np.random.default_rng(seed).normal(0, .01, 8)) + [0.] if start is None else list(start)
    if len(theta) != 9 or not all(math.isfinite(v) for v in theta):
        raise ValueError('Nine finite coefficients required')
    for _ in range(steps):
        residual = [probability(score(theta, row)) - float(label) for row, label in zip(X, y)]
        gradient = [math.fsum(float(row[j]) * r for row, r in zip(X, residual)) / len(y) + .01 * theta[j] for j in range(8)]
        gradient.append(math.fsum(residual) / len(y))
        theta = [v - .5 * g for v, g in zip(theta, gradient)]
    return np.asarray(theta, np.float64)


def centroid_reference(X, y):
    means = [[math.fsum(float(row[j]) for row, label in zip(X, y) if label == c) / sum(int(label == c) for label in y) for j in range(8)] for c in (0, 1)]
    return np.asarray([2 * (b - a) for a, b in zip(*means)] + [math.fsum(v*v for v in means[0]) - math.fsum(v*v for v in means[1])])


def quantile(values, fraction):
    values = sorted(float(v) for v in values)
    if not values or not 0 <= fraction <= 1:
        raise ValueError('Nonempty ordered quantile required')
    pos = (len(values) - 1) * fraction
    lower = math.floor(pos)
    upper = math.ceil(pos)
    return values[lower] + (pos-lower) * (values[upper]-values[lower])


def descriptor_reference(patches):
    """Sort/interpolate each pixel explicitly; no producer quantile calls."""
    patches = np.asarray(patches)
    if patches.ndim != 4 or patches.shape[2:] != (7, 7) or not np.isfinite(patches).all():
        raise ValueError('Finite N,C,7,7 patch matrix required')
    result = []
    for patch in patches:
        pixels = []
        for yy in range(7):
            for xx in range(7):
                s = [float(v) for v in patch[:, yy, xx]]
                q05, q25, q50, q75, q95 = [quantile(s, f) for f in (.05, .25, .5, .75, .95)]
                spread = q75-q25
                if spread <= 1e-12:
                    pixels.append([0.]*4)
                else:
                    pixels.append([min(max(-(q05-q50)/spread, 0), 8)/8,
                                   min(max((q95-q50)/spread, 0), 8)/8,
                                   min(max((q75+q25-2*q50)/spread, -1), 1),
                                   math.fsum(min(max((v-q50)/spread, -4), 4)**2 for v in s)/len(s)/16])
        result.append(pixels[24] + [math.fsum(p[j] for p in pixels)/49 for j in range(4)])
    return np.asarray(result, dtype=np.float64)


def metrics(y, z, p, pred, weights=None):
    if not (len(y) == len(z) == len(p) == len(pred)) or not len(y):
        raise ValueError('Aligned outcomes required')
    weights = [1]*len(y) if weights is None else list(weights)
    if len(weights) != len(y) or any(v not in (0, 1) for v in [*y, *pred]) or any(not math.isfinite(float(v)) or int(v) != v or v < 0 for v in weights) or not sum(weights):
        raise ValueError('Binary outcomes and nonnegative integer weights required')
    if any(not math.isfinite(float(v)) for v in [*z, *p]) or any(not 0 <= v <= 1 for v in p):
        raise ValueError('Finite scores and probabilities required')
    cm = [[0, 0], [0, 0]]
    for truth, guess, w in zip(y, pred, weights):
        cm[int(truth)][int(guess)] += int(w)
    den = [sum(row) for row in cm]
    total = sum(den)
    recalls = [cm[c][c]/den[c] if den[c] else None for c in (0, 1)]
    f1 = []
    for c in (0, 1):
        bottom = 2*cm[c][c]+cm[c][1-c]+cm[1-c][c]
        f1.append(2*cm[c][c]/bottom if bottom else 0.)
    oa = (cm[0][0]+cm[1][1])/total
    chance = sum(den[c]*(cm[0][c]+cm[1][c]) for c in (0, 1))/(total*total)
    losses = [max(v, 0)-truth*v+math.log1p(math.exp(-abs(v))) for truth, v in zip(y, z)]
    return dict(denominator=total, class_denominators=den, confusion=cm, OA=oa,
                AA=sum(recalls)/2 if all(v is not None for v in recalls) else None,
                recall=recalls, macroF1=sum(f1)/2,
                NLL=math.fsum(w*l for w, l in zip(weights, losses))/total,
                binary_Brier=math.fsum(w*(v-truth)**2 for w, v, truth in zip(weights, p, y))/total,
                kappa=(oa-chance)/(1-chance) if chance < 1 else None)


def resamples(parents):
    clusters = sorted(set(parents))
    if not 1 <= len(clusters) <= 3:
        raise ValueError('One to three observed parents required')
    for selected in itertools.product(clusters, repeat=len(clusters)):
        yield [selected.count(parent) for parent in parents]


def interval(values):
    valid = [v for v in values if v is not None and math.isfinite(v)]
    return dict(valid_draws=len(valid), range_2_5_97_5=[quantile(valid, f) for f in (.025, .975)] if valid else [None, None])


def sensitivity_reference(row):
    draws = [metrics(row['truth'], row['logits'], row['probabilities'], row['predictions'], w) for w in resamples(row['parent_ids'])]
    return dict(original_parent_clusters=len(set(row['parent_ids'])), enumerated_draws=len(draws),
                OA=interval([d['OA'] for d in draws]), AA=interval([d['AA'] for d in draws]),
                recall_ground=interval([d['recall'][0] for d in draws]), recall_tree=interval([d['recall'][1] for d in draws]))


def pair_reference(a, b):
    for key in ('scene', 'point_ids', 'parent_ids', 'truth'):
        if a[key] != b[key]:
            raise ValueError('Original point identity/order required for pairing')
    ca = [int(v == y) for v, y in zip(a['predictions'], a['truth'])]
    cb = [int(v == y) for v, y in zip(b['predictions'], b['truth'])]
    delta = [v-w for v, w in zip(ca, cb)]
    return dict(A=a['id'], B=b['id'], scene=a['scene'], points=len(delta),
                A_correct_B_wrong=sum(v == 1 and w == 0 for v, w in zip(ca, cb)),
                A_wrong_B_correct=sum(v == 0 and w == 1 for v, w in zip(ca, cb)),
                OA_delta=sum(delta)/len(delta), NLL_delta=a['metrics']['NLL']-b['metrics']['NLL'],
                class_OA_delta=[sum(d for d, y in zip(delta, a['truth']) if y == c)/sum(y == c for y in a['truth']) for c in (0, 1)],
                parent_resampling_OA_delta=interval([sum(w*d for w, d in zip(weights, delta))/sum(weights) for weights in resamples(a['parent_ids'])]),
                pointwise_correctness_delta=delta)


def assert_close(actual, expected, *, atol=1e-12, rtol=0, path='root'):
    """Compare nested evidence while preserving undefined values and count keys."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or not set(expected) <= set(actual):
            raise ValueError(f'Missing evidence keys: {path}')
        for key, value in expected.items():
            assert_close(actual[key], value, atol=atol, rtol=rtol, path=f'{path}.{key}')
    elif isinstance(expected, (list, tuple, np.ndarray)):
        if len(actual) != len(expected):
            raise ValueError(f'Evidence length differs: {path}')
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_close(a, b, atol=atol, rtol=rtol, path=f'{path}[{i}]')
    elif expected is None or isinstance(expected, (str, bool, int, np.integer)):
        if actual != expected:
            raise ValueError(f'Exact evidence differs: {path}')
    elif not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
        raise ValueError(f'Numerical evidence differs: {path}')
