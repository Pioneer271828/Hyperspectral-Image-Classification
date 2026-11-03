"""Read-only development-curve/exposure diagnosis; never selects new settings."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json


def require(ok, message):
    if not ok:
        raise ValueError(message)


def curve_summary(result):
    """Nested prefixes of one saved trajectory, not new budget experiments."""
    steps = result['steps']; trace = result['trace']; vals = result['validation']
    require(type(steps) is int and steps > 0, 'invalid budget')
    require([x['step'] for x in trace] == list(range(1, steps + 1)), 'incomplete trace')
    require(vals and [v['step'] for v in vals] == sorted(set(v['step'] for v in vals))
            and vals[-1]['step'] == steps and vals[0]['step'] > 0, 'invalid validation steps')
    require(all(math.isfinite(v['nll']) and v['nll'] >= 0 and 0 <= v['oa'] <= 1 for v in vals),
            'invalid validation metrics')
    require(all(math.isfinite(v) for x in trace for v in x['loss'].values()), 'invalid losses')
    best = min(vals, key=lambda v: (v['nll'], v['step']))
    require(result['selected_step'] == best['step'], 'selection differs from minimum NLL')
    window = min(100, steps)
    keys = list(trace[0]['loss'])
    prefixes = []
    for limit in sorted(set([min(n, steps) for n in (1000, 2000, 3000, 4000, steps)])):
        earlier = [v for v in vals if v['step'] <= limit]
        if earlier:
            picked = min(earlier, key=lambda v: (v['nll'], v['step']))
            prefixes.append({'budget_prefix': limit, 'selected_step': picked['step'],
                             'nll': picked['nll'], 'oa': picked['oa']})
    tail = vals[-min(10, len(vals)):]
    return {'selected_step': best['step'], 'selected_nll': best['nll'], 'selected_oa': best['oa'],
            'final_nll': vals[-1]['nll'], 'final_oa': vals[-1]['oa'],
            'best_oa_in_history': max(v['oa'] for v in vals),
            'last10_nll_range': [min(v['nll'] for v in tail), max(v['nll'] for v in tail)],
            'last10_oa_range': [min(v['oa'] for v in tail), max(v['oa'] for v in tail)],
            'prefixes': prefixes,
            'loss_first_window': {k: statistics.mean(x['loss'][k] for x in trace[:window]) for k in keys},
            'loss_last_window': {k: statistics.mean(x['loss'][k] for x in trace[-window:]) for k in keys}}


def exposure_summary(result):
    """Actual class participation, label visits, and balanced per-tile loss shares."""
    meta = result['sampler']; classes = meta['classes']; ids = meta['tile_ids']
    counts = meta['tile_class_counts']; probs = meta['tile_probabilities']
    require(len(ids) == len(counts) == len(probs) and len(set(ids)) == len(ids), 'sampler geometry')
    require(all(len(c) == len(classes) and all(type(n) is int and n >= 0 for n in c)
                and sum(c) > 0 for c in counts), 'invalid counts')
    require(all(math.isfinite(p) and p >= 0 for p in probs) and abs(sum(probs)-1) < 1e-10,
            'invalid sampling probabilities')
    by_id = dict(zip(ids, counts)); visits = dict.fromkeys(ids, 0)
    anchors = dict.fromkeys(classes, 0)
    for x in result['trace']:
        require(x['tile_id'] in by_id, 'trace outside released training tiles')
        visits[x['tile_id']] += 1
        if 'anchor_class' in x:
            require(x['anchor_class'] in anchors, 'unknown anchor class')
            anchors[x['anchor_class']] += 1
    rows = []
    for j, c in enumerate(classes):
        rows.append({'class': c, 'support_labels': sum(n[j] for n in counts),
                     'support_tiles': sum(n[j] > 0 for n in counts), 'anchor_draws': anchors[c],
                     'updates_present': sum(visits[i] for i, n in by_id.items() if n[j]),
                     'label_visits': sum(visits[i] * n[j] for i, n in by_id.items()),
                     'expected_balanced_loss_share': sum(p / sum(v > 0 for v in n)
                                                        for p, n in zip(probs, counts) if n[j])})
    return {'classes': rows, 'tile_visits_min_max': [min(visits.values()), max(visits.values())],
            'total_label_visits': sum(r['label_visits'] for r in rows),
            'unique_training_labels': sum(r['support_labels'] for r in rows)}


def seed_arm_decomposition(rows):
    """Descriptive balanced two-way SS, not a causal or significance analysis."""
    seeds = sorted(set(x['seed'] for x in rows)); modes = sorted(set(x['mode'] for x in rows))
    pairs = {(x['seed'], x['mode']): x['oa'] for x in rows}
    require(len(pairs) == len(rows) == len(seeds)*len(modes), 'incomplete/duplicate seed-arm cells')
    grand = statistics.mean(pairs.values())
    means = {s: statistics.mean(pairs[s, m] for m in modes) for s in seeds}
    arms = {m: statistics.mean(pairs[s, m] for s in seeds) for m in modes}
    total = sum((v-grand)**2 for v in pairs.values())
    seed_ss = len(modes)*sum((v-grand)**2 for v in means.values())
    arm_ss = len(seeds)*sum((v-grand)**2 for v in arms.values())
    return {'seed_means': means, 'arm_means': arms, 'total_ss': total,
            'seed_ss_fraction': seed_ss/total if total else 0.,
            'arm_ss_fraction': arm_ss/total if total else 0.,
            'scope': 'root changes initialization, head and sampler together; no isolated causal attribution'}


def legacy_summary(records):
    require(records and len({r['seed'] for r in records}) == len(records), 'missing/duplicate legacy records')
    names = records[0]['datasets']
    require(all(r['datasets'] == names and set(r['recorded_test_oa']) == set(names)
                for r in records), 'mixed legacy datasets')
    return {n: {'mean': statistics.mean(r['recorded_test_oa'][n] for r in records),
                'sample_sd': statistics.stdev(r['recorded_test_oa'][n] for r in records)
                if len(records) > 1 else None} for n in names}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--campaign-root', type=Path, required=True)
    p.add_argument('--m1-root', type=Path)
    p.add_argument('--legacy-dir', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args(); code = code_snapshot()
    require(sys.platform == 'linux' and (Path('/.dockerenv').exists() or
            Path('/run/systemd/container').exists()), 'remote container required')
    require(not sys.flags.optimize and not code['dirty'] and code['commit'] == a.expected_commit,
            'require clean exact SHA and enabled assertions')
    root = a.campaign_root.resolve(strict=True); out = external_directory(a.out_dir)
    require(out != root and root not in out.parents and out not in root.parents, 'separate diagnosis output')
    out.mkdir(parents=True, exist_ok=False); inputs = {}

    def read(path):
        path = Path(path).resolve(strict=True); inputs[str(path)] = hash_file(path)
        return json.loads(path.read_text(encoding='utf-8-sig'))

    try:
        plan = read(root/'plan.json'); summary = read(root/'paired-results.json')
        require(summary['status'] == 'completed' and summary['plan_hash'] == plan['plan_hash'], 'campaign incomplete')
        require(plan['plan_hash'] == digest({k: v for k, v in plan.items() if k != 'plan_hash'}), 'plan hash')
        require(all(code['files'].get(k) == v for k, v in plan['code']['files'].items()), 'production source changed')
        rows = []; cells = []; fingerprints = {}
        for seed in plan['seeds']:
            group = []
            for mode in plan['modes']:
                folder = root/'cases'/f'seed-{seed}'/mode
                receipt = read(folder/'case.json'); r = read(folder/'training/result.json')
                require(receipt['status'] == 'completed' and receipt['plan_hash'] == plan['plan_hash']
                        and receipt['seed'] == seed and receipt['mode'] == mode, 'case identity')
                require(receipt['receipt_hash'] == digest({k: v for k, v in receipt.items() if k != 'receipt_hash'}), 'receipt hash')
                for name, sha in receipt['artifacts'].items():
                    file = (folder/name).resolve(strict=True)
                    require(folder.resolve() in file.parents and hash_file(file) == sha, 'artifact hash')
                    inputs[str(file)] = sha
                require(not r['test_evaluated'] and r['development_hash'] == plan['inputs']['development_hash']
                        and r['steps'] == plan['updates'], 'budget/protocol mismatch')
                curves = curve_summary(r)
                rows.append({'seed': seed, 'mode': mode, 'curves': curves, 'exposure': exposure_summary(r),
                             'runtime': r['runtime_config']})
                cells.append({'seed': seed, 'mode': mode, 'oa': curves['selected_oa']})
                group.append(receipt)
            keys = ['common_initial_backbone_hash', 'initial_head_hash', 'sample_trace_hash',
                    'statistics_hash', 'evaluation_bank_hash', 'perturbation_sequence_hash']
            require(all(len({r[k] for r in group}) == 1 for k in keys), 'unpaired arms')
            fingerprints[seed] = {k: group[0][k] for k in keys}
        historical = []
        if a.m1_root:
            mp = read(a.m1_root/'plan.json')
            require(mp['inputs']['development_hash'] == plan['inputs']['development_hash'], 'different historical supports')
            for seed in plan['seeds']:
                r = read(a.m1_root/'cases'/f'seed-{seed}'/'full/training/result.json')
                require(r['development_hash'] == plan['inputs']['development_hash'] and not r['test_evaluated'], 'historical protocol')
                historical.append({'seed': seed, 'curves': curve_summary(r), 'exposure': exposure_summary(r),
                                   'initial_backbone_hash': r['initial_backbone_hash'], 'initial_head_hash': r['initial_head_hash']})
        legacy = []
        legacy_stats = None
        if a.legacy_dir:
            require(a.legacy_dir.is_dir(), 'legacy input directory missing')
            for file in sorted(a.legacy_dir.glob('*.json')):
                r = read(file)
                if 'test' in r and r.get('experiment') == 'vca_no_wave_attention_k64':
                    legacy.append({'seed': r['seed'], 'config': r['config'], 'datasets': r['datasets'],
                                   'recorded_test_oa': {k: v['oa'] for k, v in r['test'].items()}})
            legacy_stats = legacy_summary(legacy)
            for file in a.legacy_dir.glob('*.log'):
                inputs[str(file.resolve())] = hash_file(file)
        require_frozen_code(code)
        require(all(hash_file(k) == v for k, v in inputs.items()), 'input changed during diagnosis')
        write_json(out/'diagnosis.json', {'status': 'passed', 'code': code, 'rows': rows,
                   'descriptive_seed_arm_ss': seed_arm_decomposition(cells), 'fingerprints': fingerprints,
                   'historical_m1_full': historical, 'legacy_recorded_not_reproduced': legacy,
                   'legacy_recorded_summary': legacy_stats,
                   'inputs': inputs, 'new_training_updates': 0, 'new_test_evaluation': False})
        print('passed: saved curves/exposure, all case hashes; no training/test evaluation')
    except BaseException as error:
        write_json(out/'failure.json', {'error': repr(error), 'code': code, 'inputs': inputs})
        raise


if __name__ == '__main__':
    main()
