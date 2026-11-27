"""Rebuild a complete saved development result chain using restored bytes only."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import torch

from recover import BASE, read, require, sha, write


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def run(backup, expected):
    checkout = Path(__file__).resolve().parents[2]
    git = lambda *args: subprocess.check_output(['git', '-C', str(checkout), *args], text=True).strip()
    require(git('rev-parse', 'HEAD') == expected and not git('status', '--porcelain'), 'Clean published reconstruction source required')
    require(backup.resolve().is_relative_to(BASE / 'backups'), 'Authorized private restored backup required')
    layout = {k: Path(v) for k, v in read(backup / 'layout.json').items()}
    available = read(backup / 'available-private.json')
    index = {k: read(v / 'index.json') for k, v in layout.items()}
    audit = load_module('c11_original_numpy_audit', checkout / 'research/paper_c_recovery/audit_routes.py')
    banks = load_module('c11_released_bank_reader', checkout / 'research/paper_c_hsi_admission/bank_reader.py')
    case = 'paviaU-to-whu-honghu-s64'
    routes = layout['routes'] / 'restored' / case
    contract = read(routes / 'contract.json')
    setup = read(routes / 'setup.json')
    require(not contract['confirmation'] and not contract['calibration'] and contract['optimizer_updates'] == 0,
            'Only frozen development routing admitted')
    bank = banks.ReleasedBank(layout['roles'] / 'restored/whu-honghu', 'whu-honghu',
                              portable_index=layout['roles'] / 'archive-index.json')
    reg = bank.registration
    waves = np.asarray(reg['bands']['wavelengths'], dtype=np.float32)
    require(all(reg['bands']['valid']), 'Original valid-column scope differs')
    stats = reg['statistics']
    require(audit.token({k: v for k, v in stats.items() if k != 'statistics_hash'}) == setup['statistics_hash'],
            'Original frozen normalization statistics differ')
    roster, points, identities = [], [], []
    for row in bank.role_rows('devval'):
        if not bank.samples[row['id']]:
            continue
        tile = bank.read_tile('devval', row['id'])
        spectral = tile['spectral_normalized_hwc']
        y0, y1, x0, x1 = row['box']
        yy, xx = np.mgrid[y0:y1, x0:x1]
        coordinates = np.stack((yy / (reg['shape'][0]-1), xx / (reg['shape'][1]-1)), -1).astype(np.float32)
        where = np.asarray([(i//reg['shape'][1]-y0)*(x1-x0)+i%reg['shape'][1]-x0 for i, _ in tile['samples']])
        labels = np.asarray([y for _, y in tile['samples']], dtype=np.int64)
        for band, pos in [(f'stride{s}', np.arange(0,len(waves),s)) for s in (1,2,4)]+[('range_le1000nm', np.where(waves<=1000)[0])]:
            x = np.concatenate((spectral[:, :, pos], coordinates), -1)
            wave = waves[pos]
            coverage = float(np.float32((wave[-1]-wave[0])/(waves[-1]-waves[0])))
            key = f"{row['id']}-{band}"
            roster.append({'key': key, 'parent': str(row['id']), 'band': band, 'coverage': coverage,
                           'shape': list(x.shape[:2]), 'bands': len(pos), 'retained_fraction': len(pos)/len(waves),
                           'approximate_span': coverage, 'range_is_noop': band=='range_le1000nm' and len(pos)==len(waves),
                           'raw_x_sha256': __import__('hashlib').sha256(x.tobytes()).hexdigest(),
                           'raw_wave_sha256': __import__('hashlib').sha256(wave.tobytes()).hexdigest()})
            obs = audit.tensor(x)
            identities.append(audit.token([key,'whu-honghu',obs,str(row['id']),band,100,coverage,True,obs,audit.tensor(wave),None]))
            points.append((band, where, labels, spectral.shape[0]*spectral.shape[1]))
    require(roster == setup['requests'] and audit.pdigest(identities) == setup['input_hash'], 'Restored spectral, geometry, wavelength or role identity differs')
    rows = []
    for repeat in range(6):
        with np.load(routes / f'round-{repeat}-light.npz', allow_pickle=False) as light, np.load(routes / f'round-{repeat}-full.npz', allow_pickle=False) as full:
            plans, margins, cov, permuted = audit.plans(roster, light, contract)
            for arm in audit.ARMS:
                record = read(routes / f'round-{repeat}-{arm}.json')
                probfile = routes / f'round-{repeat}-{arm}.npz'
                require(sha(probfile) == record['probabilities_sha256'] and record['full_plan'] == plans[arm], 'Restored probabilities/route plan differ')
                matrices = {band: np.zeros((22,22), dtype=np.int64) for band, _, _, _ in points}
                with np.load(probfile, allow_pickle=False) as saved:
                    for i, (request, (band, where, labels, pixels)) in enumerate(zip(roster, points)):
                        p = saved[f'p{i}']
                        endpoint = full[f'p{i}'] if request['key'] in plans[arm] else light[f'p{i}']
                        require(p.shape == (pixels,22) and np.array_equal(p, endpoint) and np.isfinite(p).all()
                                and np.all(p>=0) and np.all(p<=1) and np.allclose(p.sum(1),1,atol=1e-6,rtol=0), 'Saved frozen endpoint differs')
                        matrices[band] += np.bincount((labels-1)*22+p[where].argmax(1), minlength=22**2).reshape(22,22)
                for band, cm in matrices.items():
                    m = record['metrics'][band]
                    support = cm.sum(1)
                    oa, aa = float(np.trace(cm)/cm.sum()), float(np.mean(np.diag(cm)/support))
                    require(np.array_equal(cm,m['confusion']) and np.array_equal(support,m['support'])
                            and oa==m['oa'] and aa==m['aa'] and np.allclose(np.diag(cm)/support,m['recall'],rtol=0,atol=1e-15), 'Rebuilt aggregate metrics differ')
                    rows.append({'case':case,'round':repeat,'arm':arm,'view':band,'points':int(cm.sum()),'oa':oa,'aa':aa,
                                 'full_calls':record['counts']['full_calls'],'pipeline_ms_recorded':record['pipeline_ms']})
    # Verify complete archived state envelopes, with neural execution disabled.
    state_count = 0
    for key in ('baseline','transfer'):
        for c in index[key]['recovery']:
            for role, pin in c['checkpoints'].items():
                path = layout[key] / 'restored' / pin['name']
                require(sha(path)==pin['sha256'], 'Restored state pointer differs')
                state = torch.load(path, map_location='cpu', weights_only=True)
                step = 0 if role=='initial' else c['selected_step'] if role=='selected' else c['updates']
                adam = state['optimizer']['state']
                require(state['step']==step and set(state['rng'])=={'python','numpy','torch_cpu'}
                        and state['module_modes'] and all(type(v) is bool for v in state['module_modes'].values())
                        and state['rng']['torch_cpu'].dtype==torch.uint8
                        and all(torch.isfinite(v).all() for v in state['model'].values())
                        and ((not adam) if not step else bool(adam) and all(float(v['step'])==step for v in adam.values())), 'Archived state envelope differs')
                if role=='current':
                    require(state['best']['step']==c['selected_step'], 'Current state selection pointer differs')
                state_count += 1
    target = next(c for c in index['transfer']['recovery'] if c['case']==case)
    target_contract = read(layout['transfer'] / 'restored/execution01' / case / 'contract.json')
    transfer = target_contract['transfer']
    def state_by_hash(checksum):
        path = Path(available[checksum]['path'])
        require(sha(path)==checksum, 'Restored lineage dependency differs')
        return torch.load(path,map_location='cpu',weights_only=True)
    initial = state_by_hash(target['checkpoints']['initial']['sha256'])
    source = state_by_hash(transfer['source_selected_sha256'])
    scratch = state_by_hash(transfer['target_initial_sha256'])
    require(all(torch.equal(v,source['model'][k] if k.startswith('backbone.') else scratch['model'][k])
                for k,v in initial['model'].items()) and equal(initial['optimizer'],scratch['optimizer'])
            and equal(initial['rng'],scratch['rng']), 'Source inheritance or fresh target state differs')
    selected = read(layout['transfer'] / 'restored/execution01' / case / 'stages' / f"{target['selected_step']:06d}" / 'record.json')['selection']
    full = next(r for r in rows if r['round']==0 and r['arm']=='full' and r['view']=='stride1')
    require(full['oa']==selected['oa'] and full['aa']==selected['aa'], 'Selected model readout chain differs')
    path = backup / 'review/rebuilt-routing.csv'
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    receipt = {'status':'PASSED','verifier':expected,'case':case,'restored_states_inspected':state_count,
               'restored_role_blocks_read':bank.payload_reads,'distinct_original_development_points':110,
               'route_requests':len(roster),'stage_reconstructions':36,'view_metric_rows':len(rows),
               'aggregate_differences':0,'source_backbone_fresh_target_heads_RNG_optimizer_exact':True,
               'restored_input_geometry_wavelength_normalization_hashes_exact':True,'selected_checkpoint_readout_exact':True,
               'table_sha256':sha(path),'new_neural_calls':0,'optimizer_updates':0,'query_arrays_decoded':0,
               'scope':'Saved native-source3/target3 engineering; 3 stride views plus approximate <=1000nm cut share the same 110 points. Six rounds are timing repeats, not independent samples.',
               'cross_host_optimizer_continuation_exercised':False}
    write(backup/'review/rebuild-receipt.json',receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup',type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    args=parser.parse_args()
    with patch.object(torch.nn.Module,'__call__',side_effect=AssertionError('Neural execution forbidden during recovery')):
        run(args.backup,args.expected_commit)
