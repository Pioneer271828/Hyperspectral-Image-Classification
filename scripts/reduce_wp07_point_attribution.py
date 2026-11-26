"""Read-only point decomposition and graph-input proxy analysis; no model inference."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict, Counter
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transfer.wp07_point_attribution import require, identity, decompose, concentration


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    p = Path(path).resolve(strict=True)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def checked(p):
    require(pin(p['path'])['sha256'] == p['sha256'], 'Changed input pin')
    return read(p['path'])


def table(path, rows):
    require(rows, 'Empty analysis table')
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def mean(rows, key):
    return math.fsum(r[key] for r in rows)/len(rows)


def key(r):
    return r['tile'], r['released_ordinal'], r['head'], r['band']


def proxy(entry, cell):
    q, a, b = (entry[k].to_dense().double() for k in ('q', 'near', 'wide'))
    r = q.T/q.sum(0).clamp_min(1e-6)[:, None]
    near, wide = ((q[cell]@m)@m@r for m in (a, b))
    require(bool((near >= 0).all() and (wide >= 0).all()), 'Negative proxy')
    require(abs(float(near.sum())-1) < 1e-5 and abs(float(wide.sum())-1) < 1e-5, 'Proxy mass differs')
    require(not bool(((near > 0) & ~(wide > 0)).any()), 'Near support outside wide')
    return near, wide


def tensor_digest(x, support=False):
    x = x.coalesce()
    if support:
        return hashlib.sha256(x.indices()[:, x.values()>0].numpy().tobytes()).hexdigest()
    return hashlib.sha256(x.indices().numpy().tobytes()+x.values().numpy().tobytes()).hexdigest()


def main():
    require(not sys.flags.optimize, 'Refuse optimized Python')
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('core', 'capture', 'out'): p.add_argument('--'+n, type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(1); start = time.perf_counter()
    require(not args.out.exists(), 'Use a fresh output'); args.out.mkdir()
    capture = read(args.capture/'receipt.json')
    require(capture['status']=='passed' and capture['optimizer_updates']==0 and len(capture['cases'])==6, 'Capture incomplete')
    inputs = [pin(args.capture/'receipt.json')]
    point_rows, proxy_rows, view_rows, parameter_rows, gradient_rows, band_graph_rows = [], [], [], [], [], []
    private_map, proxy_cache = {}, {}
    for scene in ('paviaU', 'xuzhou'):
        base = args.core/scene
        plan = read(base/'pair/plan.json'); inputs.append(pin(base/'pair/plan.json'))
        cache_pin = plan['cache']; require(pin(cache_pin['path'])['sha256']==cache_pin['sha256'], 'Cache changed')
        inputs.append(cache_pin); entries = torch.load(cache_pin['path'], map_location='cpu', weights_only=False)['entries']
        bank = plan['preflight']['sha256']
        cases = {(c['seed'], c['arm']): c for c in plan['cases']}
        scene_captures = [checked(r) for r in capture['cases'] if checked(r)['scene']==scene]
        inputs.extend(r for r in capture['cases'] if checked(r)['scene']==scene)
        for captured in scene_captures:
            seed = captured['seed']; maps, metrics = {}, {}
            for arm in ('content', 'mean'):
                path = Path(cases[seed, arm]['destination'])/'independent.json'
                inputs.append(pin(path)); data = read(path)['endpoints']['selected']
                maps[arm] = {key(r): r for r in data['original_point_predictions']}; metrics[arm] = data['metrics']
                require(len(maps[arm])==270, 'Duplicate or missing endpoint point')
            captured_map = {key(r): r for r in captured['points']}
            require(set(captured_map)==set(maps['content'])==set(maps['mean']), 'Point identities differ')
            cfpath = base/'counterfactual-gpu3-output'/f'{scene}-{seed}-content.json'
            inputs.append(pin(cfpath)); cf = read(cfpath)['comparisons']['selected']['content']
            validation = cf['validation']['rows']; responses = cf['response']
            require(len(validation)==len(responses), 'Response/view pairing differs')
            contexts = {}
            for v, response in zip(validation, responses, strict=True):
                weights = v['graph']['weights']['graph']['weights']
                require(len(weights)==2 and all(w>0 for w in weights) and abs(sum(weights)-1)<1e-6, 'Nonpositive soft gate')
                contexts[v['tile'], v['band']] = (weights, response)
            for k, a in maps['content'].items():
                b, c = captured_map[k], maps['mean'][k]
                require(a['class_id']==b['class_id']==c['class_id'], 'Labels differ')
                pid = identity(scene, bank, a['tile'], a['released_ordinal'])
                vid = identity(scene, bank, a['tile'], band=a['band'])
                weights, response = contexts[a['tile'], a['band']]
                decomposition = decompose(a['probabilities'], b['probabilities'], c['probabilities'], a['class_id'])
                if a['head']=='cnn': require(decomposition['coefficient_delta']==0, 'CNN coefficient effect is nonzero')
                point_rows.append({'scene': scene, 'seed': seed, 'head': a['head'], 'band': a['band'],
                    'point_id': pid, 'view_id': vid, 'class_id': a['class_id'], **decomposition,
                    'near_weight': weights[0], 'wide_weight': weights[1],
                    'view_actual_branch_rms_delta': response['rms_delta'],
                    'view_actual_branch_relative_l2_delta': response['relative_l2_delta']})
                mapping = {'scene': scene, 'tile': a['tile'], 'ordinal': a['released_ordinal'], 'native_index': b['native_index']}
                require(pid not in private_map or private_map[pid]==mapping, 'Point map changed across bands/seeds/heads')
                private_map[pid] = mapping
                proxykey = scene, a['tile'], a['band'], b['native_index']
                if proxykey not in proxy_cache:
                    near, wide = proxy(entries[f'devval/{a["tile"]}/{a["band"]}'], b['native_index'])
                    proxy_cache[proxykey] = near, wide
                near, wide = proxy_cache[proxykey]
                mixed = weights[0]*near+weights[1]*wide; average = .5*near+.5*wide
                require(torch.equal(mixed>0, average>0), 'Positive mixtures have different structural support')
                if a['head']=='fused':
                    proxy_rows.append({'scene': scene, 'seed': seed, 'band': a['band'], 'point_id': pid, 'view_id': vid,
                        'class_id': a['class_id'], 'near_weight': weights[0], 'wide_weight': weights[1],
                        'native_cells': len(near), 'near_positive_support': int((near>0).sum()),
                        'wide_positive_support': int((wide>0).sum()), 'mixed_positive_support': int((mixed>0).sum()),
                        'near_inverse_HHI': concentration(near.tolist()), 'wide_inverse_HHI': concentration(wide.tolist()),
                        'mean_inverse_HHI': concentration(average.tolist()), 'content_inverse_HHI': concentration(mixed.tolist()),
                        'near_self_weight': float(near[b['native_index']]), 'wide_self_weight': float(wide[b['native_index']]),
                        'content_self_weight': float(mixed[b['native_index']]), 'mean_self_weight': float(average[b['native_index']]),
                        'content_minus_mean_proxy_l1': float((mixed-average).abs().sum()),
                        'same_positive_union_support': True})
            casepoints = [r for r in point_rows if r['scene']==scene and r['seed']==seed]
            for head in ('fused', 'graph', 'cnn'):
                for band in ('full', 'even_retained_positions'):
                    rows = [r for r in casepoints if r['head']==head and r['band']==band]
                    require(len(rows)==45 and Counter(r['class_id'] for r in rows)==dict.fromkeys(range(1,10),5), 'Point/class denominator differs')
                    for label, field in [('content','nll_content'), ('mean','nll_trained_mean')]:
                        require(abs(mean(rows, field)-metrics[label][head][band]['nll'])<1e-12, 'Original independent NLL differs')
                    require(abs(mean(rows,'nll_content_weights_mean_coeff')-captured['metrics'][head][band]['nll'])<1e-12, 'Captured NLL differs')
            parameter_rows.extend({'scene':scene,'seed':seed,**r} for r in captured['parameter_changes'])
            for band in ('full','even_retained_positions'):
                hist = [r for r in captured['gate_history'] if r['band']==band]
                gradient_rows.append({'scene':scene,'seed':seed,'band':band,'steps':len(hist),
                    'positive_gradient_steps':sum(r['gradient_l1']>0 for r in hist), 'changed_steps':sum(r['changed'] for r in hist),
                    'gradient_l1_mean':mean(hist,'gradient_l1'),'gradient_l1_max':max(r['gradient_l1'] for r in hist)})
        for tile in sorted({r['tile'] for r in private_map.values() if r['scene']==scene}):
            full, retained = (entries[f'devval/{tile}/{band}'] for band in ('full','even_retained_positions'))
            row = {'scene':scene,'view_pair_id':identity(scene,bank,tile)}
            for name in ('q','near','wide'):
                x,y = full[name].coalesce(),retained[name].coalesce()
                require(x.shape==y.shape, 'Across-band graph shapes differ')
                row[name+'_same_positive_support'] = tensor_digest(x,True)==tensor_digest(y,True)
                row[name+'_same_values'] = tensor_digest(x)==tensor_digest(y)
                row[name+'_max_abs_weight_delta'] = float((x.to_dense()-y.to_dense()).abs().max())
            band_graph_rows.append(row)
    require(len(point_rows)==1620 and len(proxy_rows)==540 and len(private_map)==90, 'Final denominators differ')
    aggregate, classes = [], []
    groups = defaultdict(list)
    for r in point_rows: groups[r['scene'],r['seed'],r['head'],r['band']].append(r)
    fields = ['nll_content','nll_content_weights_mean_coeff','nll_trained_mean','total_delta','coefficient_delta','training_residual_delta']
    for k, rows in groups.items():
        common = dict(zip(('scene','seed','head','band'),k))
        aggregate.append({**common,'points':len(rows),**{f:mean(rows,f) for f in fields},
            **{t:sum(r['transition']==t for r in rows) for t in ('correct->correct','wrong->wrong','wrong->correct','correct->wrong')}})
        for cls in range(1,10):
            subset = [r for r in rows if r['class_id']==cls]
            classes.append({**common,'class_id':cls,'points':len(subset),**{f:mean(subset,f) for f in fields},
                'content_correct':sum(r['pred_content']==cls for r in subset),
                'trained_mean_correct':sum(r['pred_trained_mean']==cls for r in subset),
                'wrong_to_correct':sum(r['transition']=='wrong->correct' for r in subset),
                'correct_to_wrong':sum(r['transition']=='correct->wrong' for r in subset)})
        for view in sorted({r['view_id'] for r in rows}):
            subset = [r for r in rows if r['view_id']==view]; first=subset[0]
            view_rows.append({**common,'view_id':view,'points':len(subset),**{f:mean(subset,f) for f in fields},
                'near_weight':first['near_weight'],'wide_weight':first['wide_weight'],
                'branch_rms_delta':first['view_actual_branch_rms_delta'],
                'branch_relative_l2_delta':first['view_actual_branch_relative_l2_delta']})
    pairs = defaultdict(dict)
    for r in point_rows: pairs[r['scene'],r['seed'],r['head'],r['point_id']][r['band']]=r
    band_pairs=[]
    for k, rows in pairs.items():
        full,ret = (rows[b] for b in ('full','even_retained_positions'))
        require(full['class_id']==ret['class_id'], 'Across-band class differs')
        band_pairs.append({**dict(zip(('scene','seed','head','point_id'),k)),'class_id':full['class_id'],
            **{f+'_retained_minus_full':ret[f]-full[f] for f in fields},
            'content_prediction_same':full['pred_content']==ret['pred_content'],
            'trained_mean_prediction_same':full['pred_trained_mean']==ret['pred_trained_mean'],
            'wide_weight_retained_minus_full':ret['wide_weight']-full['wide_weight']})
    outputs={'points.csv':point_rows,'class-decomposition.csv':classes,'endpoint-decomposition.csv':aggregate,
        'point-graph-proxy.csv':proxy_rows,'view-decomposition.csv':view_rows,'paired-band-points.csv':band_pairs,
        'paired-band-graphs.csv':band_graph_rows,'parameter-changes.csv':parameter_rows,'gate-gradients.csv':gradient_rows}
    for name, rows in outputs.items(): table(args.out/name,rows)
    private = args.out/'_private';private.mkdir();(private/'point-map.json').write_text(json.dumps(private_map,indent=2)+'\n')
    descriptors=read(args.capture/'band-descriptors.json');inputs.append(pin(args.capture/'band-descriptors.json'))
    bands=[]
    for key_, descriptor in descriptors.items():
        scene,band=key_.split('/');m=descriptor['metadata'];w=m['wavelengths']
        bands.append({'scene':scene,'band':band,'actual_band_count':descriptor['actual_band_count'],
            'declared_wavelength_min':min(w),'declared_wavelength_max':max(w),'declared_wavelength_span':max(w)-min(w),
            'wavelength_unit':m['wavelength_unit'],'wavelength_quality':m['wavelength_quality'],
            'FWHM_present':m['fwhm'] is not None,'original_sensor_mapping_authenticated':False})
    table(args.out/'band-metadata.csv',bands)
    summary={'status':'passed','program':pin(__file__),'inputs':inputs,'tables':{n:len(r) for n,r in outputs.items()},
        'unique_validation_points':90,'points_per_scene_band':45,'per_class_per_scene_band':5,
        'model_seeds_share_one_spatial_release':True,'optimizer_updates':0,'new_classifier_evaluations':0,
        'public_original_tile_or_native_indices':False,'all_point_decompositions_exact_within':1e-12,
        'independent_and_captured_NLL_reconstructed_within':1e-12,
        'interpretation':'Reference-order-dependent decomposition; training residual is not a unique causal effect. Positive support and inverse HHI differ. Graph-input scalar proxy is not nonlinear classifier or full raw-image receptive field. No significance inference.',
        'elapsed_seconds':time.perf_counter()-start}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
