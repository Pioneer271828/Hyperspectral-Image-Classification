"""Independent NumPy input/route/endpoint audit; no producer model or policy imports."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

ARMS=('light','full','content','content_coverage','permuted_coverage','matched_random')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def token(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def pdigest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def tensor(a):
    a=np.ascontiguousarray(a,dtype=np.float32);h=hashlib.sha256(canonical({'dtype':'torch.float32','shape':list(a.shape)}).encode());h.update(a.tobytes());return h.hexdigest()


def inputs(root,reg,c,setup):
    target=c['qualification']['target'];runs=Path(reg['runs_root']);pilot=json.loads(Path(reg['pilot']).read_text())['scenes'][target]
    receipt_path=Path(pilot['receipt']['path'])
    if sha(receipt_path)!=pilot['receipt']['sha256']:raise ValueError('Original bank receipt differs')
    receipt=json.loads(receipt_path.read_text())
    from transfer import development_bank as banks,development_bank_custody as custody,paviau_development_bank as pavia
    if target=='paviaU':bank,_,_=pavia.open_publication(runs,pilot['receipt'])
    else:bank=banks.Bank(receipt,runs,custody.PROVIDERS[target](runs).release)
    if bank.pin!=setup['bank_pin'] or bank.reads!=0:raise ValueError('Original metadata bank admission differs')
    release=bank.release;stats=release.statistics
    if token({k:v for k,v in stats.items() if k!='statistics_hash'})!=stats['statistics_hash'] or stats['statistics_hash']!=setup['statistics_hash']:
        raise ValueError('Frozen training normalization differs')
    low=np.asarray(stats['minimum'],dtype=np.float32);high=np.asarray(stats['maximum'],dtype=np.float32)
    waves=np.asarray(release.bands.wavelengths,dtype=np.float32)
    if not all(release.bands.valid):raise ValueError('New invalid-column mapping not admitted by this audit')
    rosters=[];points=[];identities=[];members=bank.members
    for row in bank.role_rows('devval'):
        if not row['samples']:continue
        member=members[row['id']];file=bank.root/member['member']
        if sha(file)!=member['content']['npy_sha256']:raise ValueError('Released raw bank array bytes differ')
        raw=np.load(file,allow_pickle=False)
        if raw.dtype!=np.float32 or raw.ndim!=3:raise ValueError('Published bank dtype/shape differs')
        spectral=(raw-low)/np.where(high>low,high-low,1) if row['representation']=='raw_fp32' else raw.copy()
        y0,y1,x0,x1=row['box'];yy,xx=np.mgrid[y0:y1,x0:x1]
        coordinates=np.stack((yy/max(release.shape[0]-1,1),xx/max(release.shape[1]-1,1)),-1).astype(np.float32)
        width=release.shape[1];where=np.asarray([(int(i)//width-y0)*(x1-x0)+int(i)%width-x0 for i,_ in row['samples']],dtype=np.int64)
        labels=np.asarray([int(y) for _,y in row['samples']],dtype=np.int64)
        views=[(f'stride{s}',np.arange(0,len(waves),s)) for s in (1,2,4)]+[('range_le1000nm',np.where(waves<=1000)[0])]
        for band,pos in views:
            x=np.concatenate((spectral[:,:,pos],coordinates),-1);wave=waves[pos]
            coverage=float(np.float32((wave[-1]-wave[0])/(waves[-1]-waves[0])));key=f"{row['id']}-{band}"
            item={'key':key,'parent':str(row['id']),'band':band,'coverage':coverage,'shape':list(x.shape[:2]),'bands':len(pos),
                  'retained_fraction':len(pos)/len(waves),'approximate_span':coverage,'range_is_noop':band=='range_le1000nm' and len(pos)==len(waves),
                  'raw_x_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'raw_wave_sha256':hashlib.sha256(wave.tobytes()).hexdigest()}
            rosters.append(item);points.append((band,where,labels,raw.shape[0]*raw.shape[1]));obs=tensor(x)
            identities.append(token([key,target,obs,str(row['id']),band,50 if target=='paviaU' else 100,coverage,True,obs,tensor(wave),None]))
    if rosters!=setup['requests'] or pdigest(identities)!=setup['input_hash']:raise ValueError('Independent normalized request bytes/coordinates/waves differ')
    return points,release.classes


def plans(roster,light,c):
    ids=[r['key'] for r in roster];margins={}
    for i,key in enumerate(ids):
        top=np.sort(light[f'p{i}'],axis=1)[:,-2:];margins[key]=float(np.min(top[:,1]-top[:,0]))
    coverage={r['key']:r['coverage'] for r in roster};permuted=dict(coverage)
    for parent in sorted({r['parent'] for r in roster}):
        keys=[r['key'] for r in roster if r['parent']==parent]
        ordered=sorted(keys,key=lambda k:(pdigest([c['seed'],k]),k));values=[coverage[k] for k in ordered]
        permuted.update(zip(ordered,values[1:]+values[:1]))
    out={'light':[],'full':sorted(ids)};n=int(np.ceil(len(ids)*c['full_fraction']))
    for arm,cov in (('content',None),('content_coverage',coverage),('permuted_coverage',permuted)):
        score={k:1-margins[k]+(0 if cov is None else 1-cov[k]) for k in ids}
        out[arm]=sorted(sorted(ids,key=lambda k:(-score[k],k))[:n])
    # Preserve the exact actual WP11 cohort: band, shape, coverage, known.
    strata={};cheap=set(ids)-set(out['content_coverage']);rng=np.random.default_rng(c['seed']);accepted=[]
    for row in roster:strata.setdefault((row['band'],tuple(row['shape']),row['coverage'],True),[]).append(row['key'])
    for keys in strata.values():accepted.extend(rng.permutation(keys)[:len(cheap.intersection(keys))].tolist())
    out['matched_random']=sorted(set(ids)-set(accepted))
    return out,margins,coverage,permuted


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--training-run',required=True)
    p.add_argument('--dependency',required=True);p.add_argument('--lineage-audit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.run);c=json.loads((root/'contract.json').read_text());setup=json.loads((root/'setup.json').read_text());summary=json.loads((root/'summary.json').read_text())
    reg=json.loads((Path(a.training_run)/'registration.json').read_text());proved={r['case']:r for r in json.loads(Path(a.lineage_audit).read_text())['cases']}[c['case']]
    if c['confirmation'] or c['calibration'] or summary['optimizer_updates']!=0:raise ValueError('Development-only zero-update route scope differs')
    for name,h in summary['files'].items():
        f=(root/name).resolve()
        if not f.is_relative_to(root.resolve()) or sha(f)!=h:raise ValueError('Original route artifact differs')
    for name,h in setup['protected'].items():
        if sha(name)!=h:raise ValueError('Original protected input differs')
    git=lambda *args:subprocess.check_output(['git','-C',a.dependency,*args],text=True).strip()
    if git('rev-parse','HEAD')!='395caef57fba8ab6a09223538f748179dab552f2' or git('status','--porcelain'):raise ValueError('Pinned data-access dependency differs')
    sys.path.insert(0,a.dependency)
    # Only custodian-approved metadata/data readers are reused. No producer
    # policy, model, task, trainer or forward is imported/called by this audit.
    import torch
    with patch.object(torch.nn.Module,'__call__',side_effect=AssertionError('Independent audit attempted a neural forward')):
        points,classes=inputs(root,reg,c,setup)
    rosters=setup['requests'];keys=[r['key'] for r in rosters];audited=[];endpoint_elements=0
    for repeat in range(6):
        with np.load(root/f'round-{repeat}-light.npz',allow_pickle=False) as light,np.load(root/f'round-{repeat}-full.npz',allow_pickle=False) as full:
            expected,margins,coverage,permuted=plans(rosters,light,c)
            if setup['reference']!=expected['content_coverage']:raise ValueError('Paid setup reference plan differs')
            for i,key in enumerate(keys):
                if not np.array_equal(light[f'p{i}'],full[f'c{i}']):raise ValueError('Exact copied-CNN endpoint differs from full CNN branch')
            for arm in ARMS:
                row=json.loads((root/f'round-{repeat}-{arm}.json').read_text());selected=set(expected[arm]);matrices={b:np.zeros((classes,classes),dtype=np.int64) for b,_,_,_ in points}
                if row['full_plan']!=expected[arm] or row['contract']!=pdigest(c):raise ValueError('Independent route plan/contract differs')
                if arm in ('content','content_coverage','permuted_coverage') and (row['margin']!=margins or row['descriptor_values']!=(permuted if arm=='permuted_coverage' else coverage)):
                    raise ValueError('Whole-tile confidence or descriptor permutation differs')
                count=row['counts']
                if (count['full_calls']!=len(selected) or count['graph_builds']!=len(selected) or count['graph_reuses']
                        or count['light_calls']!=(0 if arm=='full' else len(keys))
                        or count['gate_calls']!=(len(keys) if arm in ('content','content_coverage','permuted_coverage') else 0)):
                    raise ValueError('Actual neural/graph/gate budget differs')
                with np.load(root/f'round-{repeat}-{arm}.npz',allow_pickle=False) as saved:
                    for i,(key,(band,where,y,pixels)) in enumerate(zip(keys,points)):
                        p=saved[f'p{i}'];want=full[f'p{i}'] if key in selected else light[f'p{i}']
                        if (p.shape!=(pixels,classes) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1)
                                or not np.allclose(p.sum(1),1,atol=1e-6,rtol=0) or not np.array_equal(p,want)):
                            raise ValueError('Frozen endpoint probabilities/normalization differ')
                        endpoint_elements+=p.size;matrices[band]+=np.bincount((y-1)*classes+p[where].argmax(1),minlength=classes**2).reshape(classes,classes)
                for band,m in matrices.items():
                    support=m.sum(1);r=row['metrics'][band]
                    if (not np.array_equal(m,r['confusion']) or not np.array_equal(support,r['support'])
                            or np.any(support==0) or not np.allclose(np.diag(m)/support,r['recall'],rtol=0,atol=1e-15)
                            or float(np.trace(m)/m.sum())!=r['oa'] or float(np.mean(np.diag(m)/support))!=r['aa']):
                        raise ValueError('Independent confusion, recall, OA or AA differs')
                    if arm=='full' and band=='stride1' and r['confusion']!=proved['selected_metrics']['confusion']:
                        raise ValueError('Native full selected-checkpoint readout differs from training selection')
                audited.append({'round':repeat,'arm':arm,'pipeline_ms':row['pipeline_ms'],'counts':count,'metrics':row['metrics']})
    result={'schema':'paper_c_new_direction_route_audit_v1','status':'PASSED','case':c['case'],'source':c['qualification']['source'],'target':c['qualification']['target'],
        'stages':len(audited),'requests':len(points),'endpoint_probability_elements_checked':endpoint_elements,
        'input_bytes_coordinates_wavelengths_exact':True,'copied_CNN_and_full_CNN_endpoints_exact':True,'native_selected_readout_matches_training':True,
        'coverage_matched_random_plans_identical':expected['content_coverage']==expected['matched_random'],
        'range_noop_requests':sum(r['range_is_noop'] for r in rosters),'results':audited,
        'audit_neural_forwards':0,'audit_optimizer_updates':0,'new_GT_or_query_reads':0,'confirmation':False}
    with Path(a.out).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','case','stages','requests','coverage_matched_random_plans_identical','range_noop_requests')}))


if __name__=='__main__':main()
