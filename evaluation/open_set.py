"""E4c-1: frozen array-level open-set baselines, without model or label I/O.

High scores indicate unknown. Declared provenance is not an authenticated data
release: callers must separately establish model, semantics and split legality.
No score/temperature is selected from evaluation labels by this module.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _sha(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Expected a lowercase SHA256 identity')
    return value


def _matrix(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not 0 < a.shape[0] <= 200000 or
            not 0 < a.shape[1] <= 512 or a.dtype.kind not in 'fi'):
        raise ValueError('Expected a bounded numeric matrix')
    a = a.astype(np.float64, copy=True)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite data')
    return a


def _labels(value, n):
    a = np.asarray(value)
    if a.shape != (n,) or a.dtype.kind not in 'iu':
        raise ValueError('Labels must be integer vector')
    if (a < 0).any() or (a > 2147483647).any():
        raise ValueError('Invalid semantic ID')
    return a.astype(np.int64)


def _ids(values, n):
    if (not isinstance(values, (list, tuple)) or len(values) != n or
            any(not isinstance(v, str) or not v or len(v) > 256 for v in values) or
            len(set(values)) != n):
        raise ValueError('Unique global sample IDs required')
    return list(values)


def _semantics(known, unknown):
    for rows in (known, unknown):
        if not isinstance(rows, list) or not 0 < len(rows) <= 256:
            raise ValueError('Nonempty known/unknown semantic classes required')
        for r in rows:
            if (not isinstance(r, dict) or set(r) != {'id', 'name'} or
                    type(r['id']) is not int or not 0 < r['id'] <= 2147483647 or
                    not isinstance(r['name'], str) or not r['name'].strip()):
                raise ValueError('Positive named semantic IDs required; zero is unlabeled')
    rows = known + unknown
    if (len({r['id'] for r in rows}) != len(rows) or
            len({r['name'].strip().casefold() for r in rows}) != len(rows)):
        raise ValueError('Semantic classes must be disjoint and unique')


def _array_hash(a):
    return digest({'shape': list(a.shape), 'dtype': str(a.dtype),
                   'bytes': hashlib.sha256(a.tobytes()).hexdigest()})


class OpenSetScorer:
    """Immutable specification; hashes bind bytes, not truth of provenance."""
    def __init__(self, spec, expected_hash):
        if digest(spec) != _sha(expected_hash):
            raise ValueError('Scorer identity mismatch')
        keys = {'version', 'method', 'known', 'unknown', 'model_sha256',
                'representation', 'temperature', 'prototype', 'source_ids',
                'source_provenance', 'source_data_sha256'}
        if set(spec) != keys or spec['version'] != 1:
            raise ValueError('Unsupported scorer schema')
        _semantics(spec['known'], spec['unknown'])
        _sha(spec['model_sha256'])
        if not isinstance(spec['representation'], str) or not spec['representation']:
            raise ValueError('Frozen representation identity required')
        t = spec['temperature']
        if type(t) not in (int, float) or not math.isfinite(t) or t <= 0:
            raise ValueError('Finite positive temperature required')
        if spec['method'] == 'prototype_squared_distance':
            p = _matrix(spec['prototype'])
            if len(p) != len(spec['known']) or t != 1:
                raise ValueError('Prototype shape/temperature mismatch')
            _ids(spec['source_ids'], len(spec['source_ids']))
            if not spec['source_ids']:
                raise ValueError('Source support required')
            _sha(spec['source_provenance']); _sha(spec['source_data_sha256'])
        elif spec['method'] in ('msp', 'energy'):
            if any(spec[k] is not None for k in ('prototype', 'source_provenance',
                                                'source_data_sha256')) or spec['source_ids']:
                raise ValueError('Logit scorers cannot carry prototypes')
        else:
            raise ValueError('Unknown scoring method')
        self._json = json.dumps(spec, sort_keys=True, allow_nan=False)
        self.sha256 = expected_hash

    def as_dict(self):
        return json.loads(self._json)

    @classmethod
    def freeze(cls, method, known, unknown, *, model_sha256, representation,
               temperature=1.0, source_features=None, source_labels=None,
               source_ids=None, source_role=None, source_provenance=None):
        _semantics(known, unknown)
        spec = dict(version=1, method=method, known=copy.deepcopy(known),
                    unknown=copy.deepcopy(unknown), model_sha256=model_sha256,
                    representation=representation, temperature=temperature,
                    prototype=None, source_ids=[], source_provenance=None,
                    source_data_sha256=None)
        if method == 'prototype_squared_distance':
            if source_role != 'source_train':
                raise ValueError('Prototypes require source_train support')
            x = _matrix(source_features); y = _labels(source_labels, len(x))
            ids = _ids(source_ids, len(x)); known_ids = [r['id'] for r in known]
            if set(y.tolist()) != set(known_ids):
                raise ValueError('Every known class and only known classes must have support')
            with np.errstate(over='raise', invalid='raise'):
                # Divide before summing to avoid avoidable overflow in the mean.
                proto = [np.sum(x[y == k] / int((y == k).sum()), axis=0).tolist()
                         for k in known_ids]
            spec.update(prototype=proto, source_ids=ids,
                        source_provenance=_sha(source_provenance),
                        source_data_sha256=digest([_array_hash(x), _array_hash(y), ids]))
        elif any(v is not None for v in (source_features, source_labels, source_ids,
                                        source_role, source_provenance)):
            raise ValueError('Unexpected source support for logit scorer')
        return cls(spec, digest(spec))

    def score(self, values, *, input_kind):
        spec = self.as_dict(); x = _matrix(values)
        known = np.array([r['id'] for r in spec['known']], dtype=np.int64)
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            if spec['method'] == 'prototype_squared_distance':
                p = np.array(spec['prototype'], dtype=np.float64)
                if input_kind != 'features' or x.shape[1] != p.shape[1]:
                    raise ValueError('Frozen feature dimension required')
                # Bounded sample chunks; never allocate full samples x classes x features.
                score = np.empty(len(x)); pred = np.empty(len(x), dtype=np.int64)
                for lo in range(0, len(x), 128):
                    d = np.square(x[lo:lo+128, None, :] - p[None, :, :]).sum(axis=2)
                    ix = d.argmin(axis=1)
                    score[lo:lo+128] = d[np.arange(len(ix)), ix]
                    pred[lo:lo+128] = known[ix]
            else:
                if input_kind != 'raw_logits' or x.shape[1] != len(known):
                    raise ValueError('Original raw logits in frozen known-column order required')
                maximum = x.max(axis=1)
                shifted = (x - maximum[:, None]) / spec['temperature']
                normalizer = np.exp(shifted).sum(axis=1)
                pred = known[x.argmax(axis=1)]
                if spec['method'] == 'energy':
                    score = -maximum - spec['temperature'] * np.log(normalizer)
                else:
                    score = 1.0 - 1.0 / normalizer
        if not np.isfinite(score).all():
            raise ValueError('Nonfinite score')
        return score, pred


def calibrate(scorer, values, labels, sample_ids, *, input_kind, role,
              provenance_sha256, known_coverage):
    """Known-only empirical quantile; not a population coverage guarantee."""
    if role not in ('source_validation', 'independent_development'):
        raise ValueError('Target/test/query calibration forbidden')
    if type(known_coverage) not in (int, float) or not 0 < known_coverage <= 1:
        raise ValueError('Coverage must be in (0,1]')
    scores, _ = scorer.score(values, input_kind=input_kind)
    y = _labels(labels, len(scores)); ids = _ids(sample_ids, len(scores))
    spec = scorer.as_dict(); known = [r['id'] for r in spec['known']]
    if set(y.tolist()) != set(known):
        raise ValueError('Calibration requires all known classes and no unknown/unlabeled labels')
    if set(ids) & set(spec['source_ids']):
        raise ValueError('Source support/calibration overlap')
    threshold = float(np.sort(scores)[math.ceil(known_coverage * len(scores)) - 1])
    value = dict(version=1, scorer_sha256=scorer.sha256, role=role,
                 provenance_sha256=_sha(provenance_sha256), sample_ids=ids,
                 input_kind=input_kind, data_sha256=digest([_array_hash(_matrix(values)),
                     _array_hash(y), ids]), known_coverage=known_coverage,
                 threshold=threshold, comparator='score > threshold',
                 empirical_known_coverage=float(np.mean(scores <= threshold)))
    return value, digest(value)


def _calibration(scorer, value, expected_hash):
    if digest(value) != _sha(expected_hash) or value['scorer_sha256'] != scorer.sha256:
        raise ValueError('Calibration identity mismatch')
    if (value['version'] != 1 or value['role'] not in
            ('source_validation', 'independent_development') or
            value['comparator'] != 'score > threshold' or
            not math.isfinite(value['threshold'])):
        raise ValueError('Invalid calibration contract')
    _ids(value['sample_ids'], len(value['sample_ids']))


def detection_metrics(scores, is_unknown):
    """Unknown-positive ROC; FPR95 is diagnostic, never the deployed threshold."""
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(is_unknown)
    if s.ndim != 1 or y.shape != s.shape or y.dtype.kind != 'b' or not np.isfinite(s).all():
        raise ValueError('Finite scores and boolean unknown mask required')
    pos = int(y.sum()); neg = len(y) - pos
    if not pos or not neg:
        return dict(auroc=None, fpr95=None, unknown_count=pos, known_count=neg)
    order = np.argsort(s, kind='stable'); ss = s[order]; yy = y[order]
    starts = np.r_[0, np.flatnonzero(ss[1:] != ss[:-1]) + 1]
    ends = np.r_[starts[1:], len(ss)]
    previous_neg = 0; concordance = 0.0
    for a, b in zip(starts, ends):
        p = int(yy[a:b].sum()); n = int(b-a)-p
        concordance += p * (previous_neg + .5*n); previous_neg += n
    # Include entire tied score group at the first attainable >=95% unknown TPR.
    cutoff = np.sort(s[y])[pos - (95*pos + 99)//100]
    return dict(auroc=float(concordance / (pos*neg)),
                fpr95=float(np.mean(s[~y] >= cutoff)), unknown_count=pos, known_count=neg)


def evaluate(scorer, calibration, calibration_sha256, values, labels, sample_ids,
             *, input_kind):
    _calibration(scorer, calibration, calibration_sha256)
    if input_kind != calibration['input_kind']:
        raise ValueError('Calibration representation kind mismatch')
    scores, pred = scorer.score(values, input_kind=input_kind)
    y = _labels(labels, len(scores)); ids = _ids(sample_ids, len(scores))
    spec = scorer.as_dict(); known = [r['id'] for r in spec['known']]
    unknown = [r['id'] for r in spec['unknown']]
    if set(ids) & (set(spec['source_ids']) | set(calibration['sample_ids'])):
        raise ValueError('Evaluation/support/calibration overlap')
    if not set(y.tolist()) <= set(known + unknown + [0]):
        raise ValueError('Undeclared semantic label')
    labeled = y != 0; rejected = scores > calibration['threshold']
    # -1 is explicit prediction rejection; zero always remains unlabeled.
    prediction = np.where(rejected, -1, pred)
    order = known + [-1]; cm = np.zeros((len(order), len(order)), dtype=np.int64)
    index = {k:i for i,k in enumerate(order)}
    for truth, guess in zip(y[labeled], prediction[labeled]):
        cm[index[int(truth)] if truth in known else -1, index[int(guess)]] += 1
    per = []
    for i, k in enumerate(known):
        tp = int(cm[i, i]); support = int(cm[i].sum()); fp = int(cm[:, i].sum()) - tp
        per.append(dict(id=k, support=support, recall=tp/support if support else None,
                        f1=2*tp/(support+tp+fp) if support else None))
    complete = all(r['support'] for r in per)
    osstar = float(np.mean([r['recall'] for r in per])) if complete else None
    macrof1 = float(np.mean([r['f1'] for r in per])) if complete else None
    u = np.isin(y, unknown); un = int(u.sum())
    ur = float(np.mean(rejected[u])) if un else None
    hos = None if osstar is None or ur is None else (2*osstar*ur/(osstar+ur) if osstar+ur else 0.0)
    unknown_rows = [dict(id=k, support=int((y == k).sum()),
                        recall=float(np.mean(rejected[y == k])) if (y == k).any() else None)
                    for k in unknown]
    return dict(version=1, scorer_sha256=scorer.sha256, calibration_sha256=calibration_sha256,
                confusion=cm.tolist(), confusion_order=order, known_per_class=per,
                unknown_per_class=unknown_rows, known_macro_f1=macrof1,
                osstar_class_macro_recall=osstar, unknown_sample_recall=ur, hos=hos,
                detection=detection_metrics(scores[labeled], u[labeled]),
                unlabeled_excluded=int((~labeled).sum()), labeled_count=int(labeled.sum()),
                threshold=calibration['threshold'], score_sha256=_array_hash(scores),
                prediction_sha256=_array_hash(prediction), optimizer_updates=0,
                limits=['array provenance declared, not authenticated file release',
                        'no real closed-set stability or open-set benefit established',
                        'FPR95 is unknown-positive evaluation ROC, not calibration threshold'])
