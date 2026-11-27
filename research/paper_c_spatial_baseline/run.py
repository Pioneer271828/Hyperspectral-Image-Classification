"""Normal C13 CLI: register and execute one frozen common-footprint contract.

Real execution is NOT authorized by this source. A future operator must obtain
the separately approved budget before invoking run with its exact registration.
"""
import argparse
import io
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from .protocol import METHODS, SEEDS, NATIVE, AUTHOR, OPTIMIZER, MODEL, LOSSES, admit, sha, digest, require
from .tasks import Task, clean, dependencies, parameters

ROOT = Path(__file__).resolve().parents[2]
# Reuse the exact existing C3 state machine; no checkpoint fork or monkeypatch.
sys.path.insert(0, str(ROOT/'research/paper_c_recovery'))
from transactions import execute, inspect, new_json, paired_plan, seed_all


def prepare(a, admitted, adapter, author):
    t = perf_counter(); cases = []; total_bytes = 0
    sweeps = 2 if a.synthetic else 40
    for scene, data in admitted['data'].items():
        n = len(data['roles']['train']['labels'])
        for seed in SEEDS:
            plan = paired_plan(list(range(n)), sweeps, seed)
            events = list(range(0, len(plan)+1, n if a.synthetic else n*5))
            for method in METHODS:
                seed_all(seed); task = Task(scene, data, method, adapter, author)
                p = parameters(task)
                # Conservative state ceiling includes Adam moments, selection
                # histories and serialization overhead. Measured values are
                # reported separately; this is not a runtime estimate.
                bound = 24*p + 4*1024**2 + len(events)*len(data['roles']['devval']['labels'])*256
                c = {'roster': list(range(n)), 'sweeps': sweeps, 'seed': seed, 'plan': plan,
                     'updates': len(plan), 'selection_steps': events, 'device': 'cpu',
                     'query_reads': 0, 'confirmation': False, 'optimizer': OPTIMIZER,
                     'protected': admitted['protected'], 'producer': a.expected_commit,
                     'dependency': NATIVE, 'author': AUTHOR, 'scene': scene, 'method': method,
                     'manifest_sha256': admitted['manifest_sha256'], 'model': MODEL, 'losses': LOSSES,
                     'patch_size': 7, 'graph_superpixel_scale': 10,
                     'trainable_parameters': p, 'state_bytes_upper_bound': bound,
                     'normalizer_sha256': data['normalizer_sha256'],
                     'heads': ['classification_logits'] if method == 'dsnet' else ['fused', 'graph', 'cnn'],
                     'selection': 'first minimum devval classification NLL; include step0',
                     'scope': 'synthetic engineering' if a.synthetic else 'new spatial-local development protocol; no historical S2 attribution'}
                cases.append({'id': f'{scene}-s{seed}-{method}', 'contract': c})
                total_bytes += bound*(len(plan)+1)
    updates = sum(x['contract']['updates'] for x in cases)
    require(updates <= a.max_updates and total_bytes <= a.max_state_bytes, 'Explicit update/storage budget insufficient')
    return {'schema': 'c13_registered_common_footprint_v1', 'producer': a.expected_commit,
            'hostname': socket.gethostname(), 'kind': admitted['kind'],
            'native_root': str(Path(a.native).resolve()), 'author_root': str(Path(a.author).resolve()),
            'manifest': str(Path(a.manifest).resolve()), 'manifest_sha256': admitted['manifest_sha256'],
            'ledger': admitted['ledger'], 'cases': cases, 'updates': updates,
            'labeled_center_exposures': updates,
            'spectral_pixel_exposures_with_repetition': updates*49,
            'selection_forward_calls': sum(len(c['contract']['selection_steps'])*len(admitted['data'][c['contract']['scene']]['roles']['devval']['labels']) for c in cases),
            'state_bytes_upper_bound': total_bytes, 'requested_update_ceiling': a.max_updates,
            'requested_state_ceiling': a.max_state_bytes, 'registration_seconds': perf_counter()-t,
            'approval': 'synthetic only now; future real budget and role release required',
            'class_guard': {'recall_floor_pp': -2., 'all_classes_each_seed': True,
                            'aggregate_gain': 'positive mean OA and at least2of3 paired seed gains',
                            'precision_limit': 'V5 per class has20pp recall increments; no independent-field uncertainty claim'},
            'uncertainty_units': 'model seeds are initialization repeats, not independent spatial/acquisition repeats',
            'confirmation_access': False, 'query_access': False,
            'confounds': ['native learned graph GroupNorm64 differs from S2 static graph DomainBN128',
                          'scale10 patch graph is a new local protocol, not original S2 graph topology',
                          'native fixed coordinates/wavelength metadata and method-specific losses differ from DSNet',
                          'same spectral footprint/exposure and label/update schedule do not equate capacity or FLOPs'],
            'costs_not_yet_measured': ['real data load and normalization', 'real end-to-end runtime', 'independent confirmation', 'backup and review']}


def audit(registration, out):
    """Read committed aggregates only; no model construction or scoring."""
    reg = json.loads(Path(registration).read_text()); out = Path(out)
    require(sha(out/'registration.json') == sha(registration), 'Run registration differs')
    require(sha(reg['manifest']) == reg['manifest_sha256'], 'Frozen role manifest changed')
    selected = {}; storage = 0; timings = {}; updates = 0
    manifest = json.loads(Path(reg['manifest']).read_text()); truth = {}
    for scene, s in manifest['scenes'].items():
        r = s['roles']['devval']; path = Path(reg['manifest']).parent/r['path']
        require(sha(path) == r['sha256'], 'Released label file changed')
        with np.load(path, allow_pickle=False) as z:
            truth[scene] = z['labels'].copy()  # No spectral arrays or neural calls.
    for case in reg['cases']:
        c = case['contract']; rows, done = inspect(out/case['id'], c)
        require(done, 'Comparison incomplete; no efficacy audit')
        events = [(r['step'], r['selection']) for r in rows if r['selection'] is not None]
        require(rows[-1]['selected_step'] == min(events, key=lambda x: x[1]['nll'])[0], 'First-min NLL selection differs')
        for _, metric in events:
            y = truth[c['scene']]; k = len(metric['support']); pred = np.asarray(metric['prediction'])
            require(pred.shape == y.shape and np.issubdtype(pred.dtype, np.integer)
                    and (pred >= 0).all() and (pred < k).all(), 'Prediction shape/classes differ')
            cm = np.bincount(y*k+pred, minlength=k*k).reshape(k, k)
            require(cm.tolist() == metric['confusion'] and cm.sum(1).tolist() == metric['support'], 'Independent confusion reconstruction differs')
            require(np.allclose(np.diag(cm)/cm.sum(1), metric['recall'], atol=1e-12, rtol=0), 'Independent class recall differs')
        chosen = next(r for r in rows if r['step'] == rows[-1]['selected_step'])
        selected[case['id']] = chosen['selection']
        require(chosen['selection']['points'] == sum(chosen['selection']['support']), 'Selected denominator differs')
        updates += sum(r['optimizer_updates'] for r in rows)
        for r in rows:
            require(r['checkpoint_bytes'] <= c['state_bytes_upper_bound'], 'Measured state exceeded registered bound')
            storage += r['checkpoint_bytes']
            for key, seconds in r['timings'].items():
                timings[key] = timings.get(key, 0.)+seconds
    require(updates == reg['updates'] and storage <= reg['state_bytes_upper_bound'], 'Cost budget differs')
    pairs = []
    scenes = sorted({c['contract']['scene'] for c in reg['cases']})
    for scene in scenes:
        for method in METHODS[1:]:
            seed_rows = []
            for seed in SEEDS:
                base = selected[f'{scene}-s{seed}-none']; other = selected[f'{scene}-s{seed}-{method}']
                require(base['support'] == other['support'], 'Paired class denominators differ')
                delta = [(b-a)*100 for a,b in zip(base['recall'], other['recall'])]
                seed_rows.append({'seed': seed, 'delta_oa_pp': (other['oa']-base['oa'])*100,
                                  'delta_recall_pp': delta, 'class_guard_pass': min(delta) >= -2.-1e-10})
            pairs.append({'scene': scene, 'method_minus_none': method, 'seeds': seed_rows,
                          'development_gate_pass': all(r['class_guard_pass'] for r in seed_rows)
                          and sum(r['delta_oa_pp'] for r in seed_rows) > 0
                          and sum(r['delta_oa_pp'] > 0 for r in seed_rows) >= 2})
    return {'schema': 'c13_committed_aggregate_audit_v1', 'kind': reg['kind'],
            'cases': len(reg['cases']), 'updates': updates, 'state_bytes': storage, 'timings_seconds': timings,
            'comparisons': pairs, 'real_science_conclusions': False if reg['kind'] == 'synthetic' else 'development only',
            'new_neural_calls': 0, 'query_reads': 0, 'confirmation_access': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['register', 'run', 'audit'])
    p.add_argument('--expected-commit', required=True); p.add_argument('--out', required=True)
    p.add_argument('--native'); p.add_argument('--author'); p.add_argument('--manifest')
    p.add_argument('--manifest-sha256'); p.add_argument('--synthetic', action='store_true')
    p.add_argument('--registration'); p.add_argument('--registration-sha256')
    p.add_argument('--execution-root')
    p.add_argument('--max-updates', type=int, default=0); p.add_argument('--max-state-bytes', type=int, default=0)
    p.add_argument('--budget-release-sha256'); p.add_argument('--resume', action='store_true')
    a = p.parse_args(); clean(ROOT, a.expected_commit); torch.set_num_threads(1)
    if a.command == 'audit':
        require(a.registration and a.execution_root, 'Explicit registration and execution root required')
        new_json(a.out, audit(a.registration, a.execution_root))
        return
    for name in ('native', 'author', 'manifest', 'manifest_sha256'):
        require(getattr(a, name) is not None, f'Missing {name}')
    require(sha(a.manifest) == a.manifest_sha256, 'Explicit manifest hash differs')
    admitted = admit(a.manifest, synthetic=a.synthetic)
    adapter, author = dependencies(ROOT, a.native, a.author)
    if a.command == 'register':
        require(a.max_updates > 0 and a.max_state_bytes > 0, 'Explicit budget required')
        value = prepare(a, admitted, adapter, author); new_json(a.out, value)
        print(json.dumps({'registration_sha256': sha(a.out), 'updates': value['updates'],
                          'state_bytes_upper_bound': value['state_bytes_upper_bound']})); return
    require(admitted['kind'] != 'synthetic_future_shape', 'Future-shape fixture is for admission/parameter planning only; no fitting')
    require(a.registration and sha(a.registration) == a.registration_sha256, 'Explicit frozen registration hash required')
    reg = json.loads(Path(a.registration).read_text())
    require(a.budget_release_sha256 == a.registration_sha256, 'Explicit frozen budget release required')
    require(reg['producer'] == a.expected_commit and reg['hostname'] == socket.gethostname()
            and reg['native_root'] == str(Path(a.native).resolve()) and reg['author_root'] == str(Path(a.author).resolve())
            and reg['manifest_sha256'] == admitted['manifest_sha256'] and reg['kind'] == admitted['kind'], 'Run binding differs')
    # Rebuild every contract from admitted inputs and the published registration.
    a.max_updates = reg['requested_update_ceiling']; a.max_state_bytes = reg['requested_state_ceiling']
    expected = prepare(a, admitted, adapter, author)
    for key in ('cases', 'updates', 'ledger', 'state_bytes_upper_bound', 'selection_forward_calls'):
        require(reg[key] == expected[key], f'Registered {key} differs')
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        require(a.resume and sha(out/'registration.json') == a.registration_sha256, 'Explicit unchanged resume required')
    else:
        require(not a.resume, 'Missing run cannot resume'); out.mkdir(mode=0o700)
        (out/'registration.json').write_bytes(Path(a.registration).read_bytes())
    already = sum(f.stat().st_size for f in out.rglob('state.pt'))
    require(shutil.disk_usage(out).free >= max(0, reg['state_bytes_upper_bound']-already), 'Insufficient state retention space')
    for case in reg['cases']:
        c = case['contract']; destination = out/case['id']
        def factory():
            return Task(c['scene'], admitted['data'][c['scene']], c['method'], adapter, author)
        def bound(phase, step):
            if phase == 'after_commit':
                require((destination/'stages'/f'{step:06d}'/'state.pt').stat().st_size <= c['state_bytes_upper_bound'], 'State retention bound exceeded; stop')
        result = execute(destination, c, factory, resume=destination.exists() and a.resume, fault=bound)
        print(json.dumps({'case': case['id'], 'result': result}), flush=True)
    value = audit(a.registration, out)
    if (out/'execution-audit.json').exists():
        require(json.loads((out/'execution-audit.json').read_text()) == value, 'Completed audit changed')
    else:
        new_json(out/'execution-audit.json', value)


if __name__ == '__main__':
    main()
