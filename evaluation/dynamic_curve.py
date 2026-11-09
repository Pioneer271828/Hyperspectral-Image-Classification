"""Descriptive development curves, never a policy selector or label-release gate.

The caller authenticates prediction/runtime provenance and preregistration time.
Hashes bind supplied evidence; they do not establish that a benchmark was run.
"""
from copy import deepcopy
import math
import statistics
import numpy as np

from inference.gate_calibration import _metrics
from model.research_config import digest


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(x):
    return isinstance(x, str) and len(x) == 64 and all(c in '0123456789abcdef' for c in x)


def number(x):
    return type(x) in (int, float) and math.isfinite(x)


def quality(y, p, classes, bins, epsilon):
    out = _metrics(y, p.argmax(1), classes)
    if not len(y):
        return dict(out, nll=None, ece=None, calibration_bins=[])
    confidence = p.max(1).astype(np.float64)
    correct = p.argmax(1) == y - 1
    # Equal width [left,right), last bin includes confidence=1.
    index = np.minimum((confidence * bins).astype(np.int64), bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        rows.append(dict(bin=b, count=int(mask.sum()),
                         confidence=float(confidence[mask].mean()) if mask.any() else None,
                         accuracy=float(correct[mask].mean()) if mask.any() else None))
    ece = sum(r['count'] * abs(r['confidence'] - r['accuracy']) for r in rows if r['count']) / len(y)
    nll = -np.log(np.maximum(p[np.arange(len(y)), y - 1].astype(np.float64), epsilon)).mean()
    return dict(out, nll=float(nll), ece=float(ece), calibration_bins=rows)


def timing_summary(samples, *, repeats, blocks):
    require(isinstance(samples, list) and len(samples) == repeats and repeats >= 2, 'Complete timing repeats required')
    require(all(number(v) and v > 0 for v in samples), 'Finite positive measured milliseconds required')
    return dict(wall_ms=list(samples), p50_ms=statistics.median(samples),
                p95_ms=statistics.quantiles(samples, n=100, method='inclusive')[94],
                mean_ms=statistics.mean(samples), sample_sd_ms=statistics.stdev(samples),
                blocks_per_second=blocks * 1000 / statistics.median(samples))


def area(points, interval, field):
    """All points retained; no envelope, pruning, extrapolation or tied-cost picking."""
    pairs = sorted(((p['cost']['p50_ms'], p['quality'][field]) for p in points), key=lambda p: p[0])
    if len(pairs) < 2:
        return dict(status='insufficient_points', value=None, normalized=None)
    xs = [p[0] for p in pairs]
    if len(set(xs)) != len(xs):
        return dict(status='duplicate_cost', value=None, normalized=None)
    if any(p[1] is None for p in pairs):
        return dict(status='undefined_metric', value=None, normalized=None)
    lo, hi = interval
    if lo < xs[0] or hi > xs[-1]:
        return dict(status='uncovered_interval', value=None, normalized=None)
    total = 0.
    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        left, right = max(lo, x0), min(hi, x1)
        if left >= right:
            continue
        yl = y0 + (y1-y0) * (left-x0) / (x1-x0)
        yr = y0 + (y1-y0) * (right-x0) / (x1-x0)
        total += (yl+yr) * (right-left) / 2
    return dict(status='descriptive_only', value=total, normalized=total/(hi-lo))


class CurveProtocol:
    def __init__(self, gate_protocol, specification, *, expected_hash):
        s = specification
        require(isinstance(s, dict) and digest(s) == expected_hash, 'External curve protocol mismatch')
        require(set(s) == {'schema', 'gate_protocol', 'role', 'points', 'environment_hash', 'timing_scope',
                           'warmups', 'repeats', 'cost_interval_ms', 'area_definition', 'ece_bins', 'nll_epsilon'}, 'Exact curve fields required')
        require(s['schema'] == 'dy-development-curve-v1' and s['gate_protocol'] == gate_protocol.protocol_hash
                and s['role'] == 'confirmation', 'Frozen development confirmation required')
        gate_protocol._check()
        require(pin(s['environment_hash']) and s['timing_scope'] == 'resident_cpu_batch_public_api_validation_and_compute', 'Registered CPU measurement scope required')
        require(type(s['warmups']) is int and s['warmups'] >= 1 and type(s['repeats']) is int and s['repeats'] >= 4 and s['repeats'] % 4 == 0, 'Registered warmup and balanced repeat budget required')
        require(type(s['ece_bins']) is int and 1 <= s['ece_bins'] <= 100 and number(s['nll_epsilon']) and 0 < s['nll_epsilon'] < 1, 'Explicit calibration definitions required')
        interval = s['cost_interval_ms']
        require(isinstance(interval, list) and len(interval) == 2 and all(number(v) and v > 0 for v in interval) and interval[0] < interval[1], 'Fixed positive cost interval required')
        require(s['area_definition'] == 'all_points_cost_sorted_piecewise_linear_trapezoid_no_envelope', 'Fixed descriptive area definition required')
        require(isinstance(s['points'], list) and s['points'], 'Complete preregistered point list required')
        seen = set()
        for p in s['points']:
            require(isinstance(p, dict) and set(p) == {'id', 'method', 'policy_hash'} and isinstance(p['id'], str)
                    and p['id'] and p['id'] not in seen and pin(p['policy_hash']), 'Unique policy identities required')
            require(p['method'] in ('full', 'fast', 'adaptive', 'random', 'fixed'), 'Unknown control method')
            seen.add(p['id'])
        require({'full', 'fast'} <= {p['method'] for p in s['points']}, 'Both endpoints must remain in report')
        self.gate = gate_protocol
        self._spec = deepcopy(s)
        self.protocol_hash = expected_hash

    @property
    def specification(self):
        return deepcopy(self._spec)

    def report(self, records, results, *, expected_results_hash):
        require(digest(self._spec) == self.protocol_hash, 'Curve protocol changed')
        require(pin(expected_results_hash) and digest(results) == expected_results_hash, 'External result evidence mismatch')
        rows, data_hash = self.gate.records(records, 'confirmation')
        s, g = self._spec, self.gate.specification
        require(isinstance(results, list) and len(results) == len(s['points']), 'Every registered point must be reported')
        by_id = {p['id']: p for p in s['points']}
        seen = set()
        summaries = []
        for r in results:
            require(isinstance(r, dict) and set(r) == {'id', 'policy_hash', 'full_state', 'light_artifact', 'data_hash',
                      'environment_hash', 'timing_scope', 'warmups', 'routes', 'wall_ms', 'execution_receipt_sha256'}, 'Exact runtime evidence fields required')
            require(r['id'] in by_id and r['id'] not in seen, 'Unknown or repeated result')
            seen.add(r['id']); p = by_id[r['id']]
            require(r['policy_hash'] == p['policy_hash'] and r['full_state'] == g['full_state'] and r['light_artifact'] == g['light_artifact']
                    and r['data_hash'] == data_hash and pin(r['execution_receipt_sha256']), 'Model/data/policy/runtime identity mismatch')
            require(all(r[k] == s[k] for k in ('environment_hash', 'timing_scope', 'warmups')) and type(r['warmups']) is int, 'Incomparable timing environment/scope')
            routes = r['routes']
            require(isinstance(routes, list) and len(routes) == len(rows) and all(isinstance(v, dict) and set(v) == {'id', 'fast'} and type(v['fast']) is bool for v in routes), 'Explicit whole-block routes required')
            route_map = {v['id']: v['fast'] for v in routes}
            require(len(route_map) == len(rows) and set(route_map) == {v['block']['id'] for v in rows}, 'Complete unique block routes required')
            parts = []; groups = {}; accepted = 0
            for row in rows:
                accept = route_map[row['block']['id']]
                eligible = row['known'] and row['coverage'] >= g['min_coverage']
                require(not accept or eligible, 'Unknown/low coverage cannot use fast path')
                require(p['method'] != 'full' or not accept, 'Full endpoint cannot skip')
                require(p['method'] != 'fast' or accept == eligible, 'Fast endpoint must preserve coverage guard')
                selected = row['indices']
                part = (row['labels'], row['fast'][selected], row['full'][selected], np.full(len(selected), accept, dtype=bool))
                parts.append(part); accepted += accept
                groups.setdefault((row['block']['scene'], row['block']['condition']), []).append(part)
            def summarize(items):
                y, fast, full, use_fast = [np.concatenate([x[i] for x in items]) for i in range(4)]
                output = np.where(use_fast[:, None], fast, full)
                q = quality(y, output, g['classes'], s['ece_bins'], s['nll_epsilon'])
                base = quality(y, full, g['classes'], s['ece_bins'], s['nll_epsilon'])
                bad_fast = fast.argmax(1) != y-1; good_full = full.argmax(1) == y-1
                def rate(n, d): return float(n/d) if d else None
                ac = int(use_fast.sum()); fallback_bad = int((~use_fast & bad_fast).sum())
                drop = g['max_drop']
                safe = (min(q['support']) >= g['min_class_support'] and q['aa'] is not None and base['oa']-q['oa'] <= drop['oa']
                        and base['aa']-q['aa'] <= drop['aa'] and all(a-b <= drop['recall'] for a,b in zip(base['recall'],q['recall'])))
                return dict(quality=q, full=base, passes_empirical_protection=bool(safe), accepted_labeled_points=ac,
                    accepted_label_fraction=rate(ac,len(y)), wrong_accepted_points=int((use_fast & bad_fast).sum()),
                    wrong_given_accepted=rate(int((use_fast & bad_fast).sum()),ac),
                    fallback_fast_errors=fallback_bad, fallback_corrected=int((~use_fast & bad_fast & good_full).sum()),
                    correction_given_fallback_fast_error=rate(int((~use_fast & bad_fast & good_full).sum()),fallback_bad))
            summary = summarize(parts)
            grouped = [dict(scene=k[0], condition=k[1], **summarize(v)) for k,v in sorted(groups.items())]
            summaries.append(dict(p, **summary, groups=grouped, routes=deepcopy(routes),
                all_groups_protected=summary['passes_empirical_protection'] and all(v['passes_empirical_protection'] for v in grouped),
                accepted_tiles=accepted, total_tiles=len(rows), accepted_tile_fraction=accepted/len(rows),
                cost=timing_summary(r['wall_ms'], repeats=s['repeats'], blocks=len(rows)), execution_receipt_sha256=r['execution_receipt_sha256']))
        summaries.sort(key=lambda p: list(by_id).index(p['id']))
        curves = []
        for method in sorted({p['method'] for p in summaries}):
            points = [p for p in summaries if p['method'] == method]
            curves.append(dict(method=method, point_ids=[p['id'] for p in sorted(points,key=lambda p:p['cost']['p50_ms'])],
                               areas={f: area(points,s['cost_interval_ms'],f) for f in ('oa','aa','nll','ece')}))
        return dict(format='dy-development-curve-report-v1', protocol_hash=self.protocol_hash, evidence_hash=expected_results_hash,
                    data_hash=data_hash, points=summaries, curves=curves, cost_interval_ms=s['cost_interval_ms'][:],
                    selection_performed=False, deployment_interpolation=False, statistical_noninferiority=False,
                    provenance='Caller must authenticate runtime/prediction provenance and preregistration time; hashes alone do not prove execution.',
                    real_speed_claim=False, optimizer_updates=0, query=0)
