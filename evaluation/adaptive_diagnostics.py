"""Offline two-readout diagnostics; neither a label-release gate nor a router."""
import numpy as np

from .branch_diagnostics import branch_diagnostics
from .classification_metrics import class_order, class_indices, classification_metrics


def _summary(labels, fast, full, ids):
    branches = branch_diagnostics(labels, fast, full, full, ids)
    return dict(count=len(labels), four_way={name: branches['strata'][old]['count'] for name, old in zip(
                    ('both_correct', 'fast_only_correct', 'full_only_correct', 'both_wrong'),
                    ('both_correct', 'cnn_only_correct', 'graph_only_correct', 'both_wrong'))},
                fast=classification_metrics(labels, fast, ids), full=classification_metrics(labels, full, ids),
                full_corrects_fast=int(((fast != labels) & (full == labels)).sum()),
                full_harms_fast=int(((fast == labels) & (full != labels)).sum()),
                pixel_oracle_accuracy=float(((fast == labels) | (full == labels)).mean()) if len(labels) else None)


def _uncertainty(probabilities):
    order = np.sort(probabilities, axis=1)
    logs = np.zeros_like(probabilities); np.log(probabilities, out=logs, where=probabilities > 0)
    entropy = -(probabilities * logs).sum(1)
    return dict(max_probability=order[:, -1].tolist(), margin=(order[:, -1] - order[:, -2]).tolist(),
                entropy_nats=entropy.tolist())


def diagnose(records, class_ids, *, role, readout_kind):
    """Consume already-authorized aligned arrays; no threshold fitting or file access.

    IDs are unique within scene across blocks, and align both probability matrices.
    Masked-out entries are not scored or validated as probability/label values.
    A terminal auxiliary head is descriptive only, never evidence of skipped work.
    """
    if role not in ('train', 'validation'):
        raise ValueError('Only already-authorized train/validation diagnostics are allowed')
    if readout_kind not in ('terminal_auxiliary', 'independent_light', 'shared_prefix_exit'):
        raise ValueError('Explicit readout_kind required')
    classes = class_order(class_ids)
    if len(classes) < 2: raise ValueError('At least two classes required')
    seen = set(); blocks = []; all_y = []; all_f = []; all_d = []; all_b = []
    groups = {}
    for row in records:
        if set(row) != {'scene', 'condition', 'tile_id', 'ids', 'labels', 'mask', 'fast', 'full'}:
            raise ValueError('Exact aligned diagnostic record fields required')
        scene, condition, tile = row['scene'], row['condition'], row['tile_id']
        if (not isinstance(scene, str) or not scene or not isinstance(condition, str) or not condition
                or type(tile) is not int or tile < 0): raise ValueError('Invalid block identity')
        key = (scene, tile)
        if key in seen: raise ValueError('Repeated block')
        seen.add(key)
        loc = np.asarray(row['ids']); y = np.asarray(row['labels']); mask = np.asarray(row['mask'])
        if (loc.ndim != 1 or loc.dtype.kind not in 'iu' or len(np.unique(loc)) != len(loc)
                or (loc < 0).any() or (loc > np.iinfo(np.int64).max).any()): raise ValueError('Invalid IDs')
        if y.shape != loc.shape or mask.shape != loc.shape or mask.dtype != np.bool_: raise ValueError('Invalid alignment/mask')
        for i in loc:
            pkey = (scene, int(i), 'pixel')
            if pkey in seen: raise ValueError('Repeated scene coordinate')
            seen.add(pkey)
        # Validate only selected labels; all selected entries must be actual class IDs.
        truth = class_indices(y[mask], classes, 'Selected labels')
        truth = np.asarray(classes)[truth]
        probs = []
        for field in ('fast', 'full'):
            raw = np.asarray(row[field])
            if raw.shape != (len(loc), len(classes)) or raw.dtype.kind not in 'fiu': raise ValueError('Probability shape/type')
            p = raw[mask].astype(np.float64)
            if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
                    or not np.allclose(p.sum(1), 1, rtol=0, atol=1e-6)): raise ValueError('Invalid selected probabilities')
            probs.append(p)
        f, d = [np.asarray(classes)[p.argmax(1)] for p in probs]
        use_full = bool((d == truth).sum() > (f == truth).sum()) if len(truth) else None
        chosen = d if use_full else f
        block = dict(scene=scene, condition=condition, tile_id=tile, selected_ids=loc[mask].tolist(),
                     **_summary(truth, f, d, classes), oracle_use_full=use_full,
                     fast_uncertainty=_uncertainty(probs[0]), full_uncertainty=_uncertainty(probs[1]))
        blocks.append(block)
        all_y.extend(truth); all_f.extend(f); all_d.extend(d); all_b.extend(chosen)
        groups.setdefault((scene, condition), []).append((truth, f, d, chosen))
    y, f, d, chosen = [np.asarray(v, dtype=np.int64) for v in (all_y, all_f, all_d, all_b)]
    result = _summary(y, f, d, classes)
    result['block_oracle'] = classification_metrics(y, chosen, classes)
    result['groups'] = []
    for (scene, condition), parts in sorted(groups.items()):
        gy, gf, gd, gb = [np.concatenate([v[i] for v in parts]) for i in range(4)]
        result['groups'].append(dict(scene=scene, condition=condition, **_summary(gy, gf, gd, classes),
                                     block_oracle=classification_metrics(gy, gb, classes)))
    result.update(role=role, readout_kind=readout_kind, blocks=blocks, oracle_tie='fast',
                  unlabeled_blocks=sum(b['count'] == 0 for b in blocks),
                  oracle_scope='Allowed labeled points only; not unobserved pixels or deployable routing.',
                  deployment_speed_evidence=False, label_authorization='Caller must authenticate provenance; role string is not permission.')
    return result


def cost_report(costs, run_full, *, kind):
    """Account for supplied same-unit per-block costs, not an actual speed benchmark.

    Independent: fast + gate + reorder + run_full*full.
    Shared prefix: prefix + exit + gate + reorder + run_full*remaining.
    Empty records and sample SD with n<2 stay undefined. No label-derived routing.
    """
    if kind == 'terminal_auxiliary':
        if costs is not None or run_full is not None: raise ValueError('Terminal head has no independent fast-path cost')
        return dict(kind=kind, estimated_only=True, total=None, reason='No actual skipped path')
    required = {'fast', 'full', 'gate', 'reorder'} if kind == 'independent_light' else {'prefix', 'exit', 'remaining', 'gate', 'reorder'}
    if kind not in ('independent_light', 'shared_prefix_exit') or not isinstance(costs, dict) or set(costs) != required:
        raise ValueError('Explicit cost components required')
    flag = np.asarray(run_full)
    if flag.ndim != 1 or flag.dtype != np.bool_: raise ValueError('Boolean per-block decisions required')
    values = {}
    for name, raw in costs.items():
        v = np.asarray(raw)
        if v.shape != flag.shape or v.dtype.kind not in 'fiu' or not np.isfinite(v).all() or (v < 0).any():
            raise ValueError('Finite nonnegative aligned costs required')
        values[name] = v.astype(np.float64)
    v = values
    if kind == 'independent_light':
        total = v['fast'] + v['gate'] + v['reorder'] + flag * v['full']
        baseline = v['full']
    else:
        total = v['prefix'] + v['exit'] + v['gate'] + v['reorder'] + flag * v['remaining']
        baseline = v['prefix'] + v['remaining']
    if not np.isfinite(total).all() or not np.isfinite(baseline).all(): raise ValueError('Cost overflow')
    n = len(total)
    return dict(kind=kind, estimated_only=True, per_block=total.tolist(), count=n,
                mean=float(total.mean()) if n else None, sample_sd=float(total.std(ddof=1)) if n > 1 else None,
                p95=float(np.quantile(total, .95)) if n else None, full_baseline_mean=float(baseline.mean()) if n else None,
                run_full_fraction=float(flag.mean()) if n else None,
                limits='Input costs must share hardware/unit/conditions; estimated additive cost is not measured end-to-end latency.')
