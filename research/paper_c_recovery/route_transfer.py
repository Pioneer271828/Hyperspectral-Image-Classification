"""Frozen six-arm deployment readout for an independently audited new transfer."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from transactions import sha,new_json
from tasks import NATIVE,MODEL,prior_module
from run import clean,verify_files


def previous_policy(root):
    p=root/'research/paper_c_routing/policy.py';s=importlib.util.spec_from_file_location('paper_c_fixed_routing_policy',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def prepare_requests(bank):
    from inference.native_deployment import Request
    from research_runtime import tensor_hash
    requests=[];points=[];labels=[];roster=[]
    for row in bank.role_rows('devval'):
        if not row['samples']:continue
        _,_,view,context=bank.materialize(row,{'mode':'none'});obs=view.observation
        value=torch.tensor(np.concatenate((obs.spectral,obs.coordinates),-1));waves=torch.tensor(np.asarray(view.bands.wavelengths,dtype=np.float32))
        n=len(waves);y0,_,x0,_=row['box'];width=bank.release.shape[1]
        local=[(int(i)//width-y0)*value.shape[1]+int(i)%width-x0 for i,_ in row['samples']]
        ys=[int(y) for _,y in row['samples']]
        views=[(f'stride{s}',torch.arange(0,n,s)) for s in (1,2,4)]+[('range_le1000nm',torch.where(waves<=1000)[0])]
        for name,pos in views:
            if len(pos)<4:raise ValueError('Synthetic view retains fewer than four admitted columns')
            x=torch.cat((value[:,:,pos],value[:,:,n:]),-1);wave=waves[pos]
            coverage=float((wave[-1]-wave[0])/(waves[-1]-waves[0]));key=f"{row['id']}-{name}"
            r=Request(key,bank.release.scene,tensor_hash(x),str(row['id']),name,x,wave,None,None,
                      50 if bank.release.scene=='paviaU' else 100,coverage,True)
            requests.append(r);points.append(local);labels.append(ys)
            roster.append({'key':key,'parent':r.parent,'band':name,'coverage':coverage,'shape':list(x.shape[:2]),
                'bands':len(pos),'retained_fraction':len(pos)/n,'approximate_span':coverage,
                'range_is_noop':name=='range_le1000nm' and len(pos)==n,
                'raw_x_sha256':__import__('hashlib').sha256(x.numpy().tobytes()).hexdigest(),
                'raw_wave_sha256':__import__('hashlib').sha256(wave.numpy().tobytes()).hexdigest()})
    return requests,points,labels,roster


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('expected-commit','training-run','lineage-audit','case','out'):p.add_argument('--'+n,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];clean(root,a.expected_commit)
    if sys.platform!='linux' or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Authorized Linux CPU only')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);policy=previous_policy(root)
    training=Path(a.training_run);reg=json.loads((training/'registration.json').read_text());lineage=json.loads(Path(a.lineage_audit).read_text())
    if lineage['status']!='PASSED' or lineage['producer']!=reg['producer'] or lineage['new_target_optimizer_updates']!=2007:
        raise ValueError('Complete independent transfer lineage audit required')
    case=next(x for x in reg['cases'] if x['id']==a.case);c=case['contract'];proved=next(x for x in lineage['cases'] if x['case']==a.case)
    if not all(proved[k] for k in ('source_backbone_exact','fresh_target_heads_exact','target_initial_rng_optimizer_exact')):
        raise ValueError('Source-only backbone and matched fresh target controls not verified')
    checkpoint=training/a.case/'stages'/f'{proved["selected_step"]:06d}'/'state.pt'
    selected_row=json.loads((checkpoint.parent/'record.json').read_text())
    if sha(checkpoint)!=selected_row['state_sha256']:raise ValueError('Selected target state differs')
    clean(reg['dependency_root'],NATIVE)
    protected={**c['protected'],str(checkpoint):sha(checkpoint),str(Path(a.lineage_audit).resolve()):sha(a.lineage_audit),
               str(training/'registration.json'):sha(training/'registration.json'),str(root/'research/paper_c_routing/policy.py'):sha(root/'research/paper_c_routing/policy.py')}
    verify_files(protected)
    contract={'schema':'paper_c_new_transfer_routing_v1','code':a.expected_commit,'training_producer':reg['producer'],'case':a.case,
        'qualification':{'source':c['transfer']['source'],'target':c['scene'],'scope':'native3/target3 engineering development only',
            'source_inherited_updates':c['transfer']['source_inherited_executed_updates'],'target_updates':c['updates'],
            'source_selected_step':c['transfer']['source_selected_step'],'target_selected_step':proved['selected_step'],
            'target_K':5,'target_V':5,'light':'exact frozen trained target CNN branch copy; no light fitting'},
        'seed':64,'full_fraction':.5,'rounds':6,'descriptor':'approximate_span','range_view':True,
        'calibration':False,'confirmation':False,'optimizer_updates':0,'protected':protected,
        'sensor_limit':'Approximate center/stride views, not measured SRF or unseen-family confirmation; <=1000nm can be a no-op for VNIR products.'}
    out=Path(a.out).resolve();runs=Path(reg['runs_root']).resolve()
    if out==runs or not out.is_relative_to(runs):raise ValueError('External run path required')
    pin=policy.digest(contract)
    if out.exists():
        if not a.resume or json.loads((out/'contract.json').read_text())!=contract:raise ValueError('Explicit unchanged resume required')
        if (out/'summary.json').exists():
            policy.verify_completed(out);print(json.dumps({'completed_reentry':True,'new_neural_calls':0,'new_optimizer_updates':0}));return
    else:out.mkdir(mode=0o700);new_json(out/'contract.json',contract)
    sys.path.insert(0,reg['dependency_root'])
    from inference.light_cnn import LightCNN
    from inference.native_deployment import Runtime
    from model.VCASAHGCN import MultiDatasetVCASAHGCN
    from transfer.g3_training import state_hash
    start=perf_counter();old=prior_module(root,'pilot');bank,receipt=old.load_bank(SimpleNamespace(runs_root=reg['runs_root']),c['scene'])
    pilot=json.loads(Path(reg['pilot']).read_text())['scenes'][c['scene']]
    if receipt!=pilot['receipt'] or bank.pin!=pilot['pin']:raise ValueError('Existing target bank admission differs')
    requests,points,labels,roster=prepare_requests(bank);classes=bank.release.classes
    full=MultiDatasetVCASAHGCN({c['scene']:classes},**MODEL)
    full.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu')['model'],strict=True);full.eval();full.requires_grad_(False)
    full.backbone.adapter.cache_attention=False;full.backbone.adapter.clear_attention_cache();light=LightCNN(full)
    before={'full':state_hash(full),'light':state_hash(light)};runtime=Runtime(device='cpu',full=full,light=light,identity=before,graph_policy='rebuild')
    input_hash=policy.digest([r.identity for r in requests]);loading_ms=(perf_counter()-start)*1000
    if (out/'setup.json').exists():
        setup=json.loads((out/'setup.json').read_text())
        if setup['input_hash']!=input_hash or setup['models']!=before:raise ValueError('Resumed input/model drift')
        reference=setup['reference']
    else:
        ref=policy.execute(runtime,requests,'content_coverage');reference=ref['full_plan']
        setup={'input_hash':input_hash,'models':before,'requests':roster,'reference':reference,'reference_counts':ref['counts'],
            'reference_ms':ref['pipeline_ms'],'loading_ms':loading_ms,'protected':protected,'bank_pin':bank.pin,
            'distinct_development_points':sum(len(y) for i,y in enumerate(labels) if requests[i].band=='stride1'),
            'class_order':list(range(1,classes+1)),'statistics_hash':bank.release.statistics['statistics_hash'],
            'bank_normalization_is_training_contract':True,'warmup_limit':'one paid content+coverage setup reference, not one warmup per arm'}
        new_json(out/'setup.json',setup)
    rows=[];completed=0
    for repeat in range(6):
        order=policy.ARMS[repeat:]+policy.ARMS[:repeat]
        for arm in order:
            name=f'round-{repeat}-{arm}';rowfile=out/(name+'.json');probfile=out/(name+'.npz')
            if rowfile.exists():
                row=json.loads(rowfile.read_text())
                if row['contract']!=pin or row['probabilities_sha256']!=sha(probfile):raise ValueError('Committed route stage differs')
            else:
                if probfile.exists():raise ValueError('Uncommitted probability artifact retained; no overwrite')
                actual=policy.execute(runtime,requests,arm,reference=reference)
                arrays={f'p{i}':v[0].cpu().numpy() for i,v in enumerate(actual['predictions'])}
                if arm=='full':arrays.update({f'c{i}':v[2].cpu().numpy() for i,v in enumerate(actual['predictions'])})
                with probfile.open('xb') as f:np.savez(f,**arrays)
                groups={}
                for i,req in enumerate(requests):
                    g=groups.setdefault(req.band,{'y':[],'p':[]});g['y'].extend(labels[i]);g['p'].extend(arrays[f'p{i}'][points[i]])
                row={k:v for k,v in actual.items() if k!='predictions'}
                row.update(contract=pin,round=repeat,input_hash=input_hash,
                    metrics={k:policy.metrics(g['y'],np.asarray(g['p']),classes) for k,g in groups.items()},probabilities_sha256=sha(probfile))
                new_json(rowfile,row);completed+=1
            rows.append(row)
            if a.stop_after is not None and completed>=a.stop_after:
                print(json.dumps({'partial':True,'new_stages':completed,'new_optimizer_updates':0}));return
    if before!={'full':state_hash(full),'light':state_hash(light)}:raise ValueError('Frozen deployment weights changed')
    verify_files(protected)
    for arm in policy.ARMS:
        same=[r for r in rows if r['arm']==arm]
        if any(r['full_plan']!=same[0]['full_plan'] or r['metrics']!=same[0]['metrics'] for r in same):raise ValueError('Frozen repeat drift')
    files={f.name:sha(f) for f in out.iterdir() if f.is_file()}
    new_json(out/'summary.json',{'status':'PASSED_NEW_ENGINEERING_TRANSFER_DEVELOPMENT','files':files,'results':rows,
        'optimizer_updates':0,'query_reads':0,'confirmation':False,'protected_inputs_unchanged':True,'models_unchanged':True})
    print(json.dumps({'status':'PASSED','stages':len(rows),'new_optimizer_updates':0,'requests':len(requests)}))


if __name__=='__main__':main()
