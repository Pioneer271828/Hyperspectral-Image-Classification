"""Fit only a registered source or target support stage; never decode development arrays."""
import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from c8_bridge import clean,fresh,write,read,digest,canonical,sha,protocol,require_learning_enabled
from phase_data import PhaseData
from features import REPRESENTATIONS
from models import fit,centroid


def main():
    require_learning_enabled()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--stage',choices=('source','target'),required=True);p.add_argument('--direction',nargs=2,required=True);p.add_argument('--c7-root',required=True);p.add_argument('--inputs',required=True);p.add_argument('--plan',required=True);p.add_argument('--source-stage');p.add_argument('--out',required=True);a=p.parse_args();out=fresh(a.expected_commit,a.out);out.mkdir();began=perf_counter();cfg=protocol();config_hash=sha(canonical(cfg))
    if tuple(a.direction) not in (('xuzhou','whu-honghu'),('whu-honghu','xuzhou')):raise ValueError('Unregistered direction')
    if (a.stage=='target')!=(a.source_stage is not None):raise ValueError('Target stage requires exact sealed source stage only')
    source_manifest=None;source_root=None
    if a.source_stage:
        source_root=Path(a.source_stage);source_manifest=read(source_root/'manifest.json')
        if source_manifest['stage']!='source' or source_manifest['direction']!=a.direction or source_manifest['protocol_sha256']!=config_hash or source_manifest['source']!=a.expected_commit or not source_manifest['complete']:raise ValueError('Source fitting ancestry differs')
    phase='source_fit' if a.stage=='source' else 'target_adapt';data=PhaseData(expected_commit=a.expected_commit,c7_root=a.c7_root,inputs=a.inputs,plan=a.plan,direction=a.direction,allowed={phase});X,y,frame=data.load(phase,out/'features');endpoints=[];updates=0
    expected_scene=a.direction[0] if a.stage=='source' else a.direction[1]
    if frame['scene']!=expected_scene or frame['original_role']!='devtrain':raise ValueError('Fitting cannot use another scene or development-validation role')
    for representation in REPRESENTATIONS:
        features=X[representation]
        for seed in cfg['seeds']:
            arms=('source_fit',) if a.stage=='source' else ('target_adapt','target_scratch64','target_scratch128')
            for arm in arms:
                steps=128 if arm=='target_scratch128' else 64;start=None;ancestor=None
                if arm=='target_adapt':
                    entry=next(m for m in source_manifest['models'] if m['representation']==representation and m['seed']==seed and m['arm']=='source_fit');path=source_root/entry['file']
                    if digest(path)!=entry['sha256']:raise ValueError('Sealed source endpoint changed')
                    parent=read(path);start=parent['theta'];ancestor={'file':entry['file'],'sha256':entry['sha256'],'source_updates':64}
                tag=f'{representation}-s{seed}-{arm}';checkpoints=out/(tag+'-checkpoints');checkpoints.mkdir();model=fit(features,y,seed=seed,steps=steps,start=start,checkpoint_dir=checkpoints);updates+=model['optimizer_updates']
                model.update(source=a.expected_commit,protocol_sha256=config_hash,direction=a.direction,representation=representation,arm=arm,fit_scene=frame['scene'],fit_role='devtrain',fit_phase=phase,fit_points=10,fit_class_counts=[5,5],feature_cache_sha256=frame['cache_sha256'],parent_source=ancestor,source_updates=64 if arm in ('source_fit','target_adapt') else 0,target_updates=0 if arm=='source_fit' else steps,selection='fixed_terminal_no_development_selection')
                path=out/(tag+'.json');write(path,model);endpoints.append({'file':path.name,'sha256':digest(path),'representation':representation,'seed':seed,'arm':arm,'kind':'logistic','optimizer_updates':steps})
        if a.stage=='source':
            model=centroid(features,y);model.update(source=a.expected_commit,protocol_sha256=config_hash,direction=a.direction,representation=representation,arm='centroid',fit_scene=frame['scene'],fit_role='devtrain',fit_points=10,feature_cache_sha256=frame['cache_sha256'],selection='closed_form_no_development_selection')
            path=out/(representation+'-centroid.json');write(path,model);endpoints.append({'file':path.name,'sha256':digest(path),'representation':representation,'seed':None,'arm':'centroid','kind':'centroid','optimizer_updates':0})
    expected_updates=384 if a.stage=='source' else 1536;expected_models=8 if a.stage=='source' else 18
    if updates!=expected_updates or len(endpoints)!=expected_models or data.denied or 'torch' in sys.modules:raise ValueError('Stage budget/input/checkpoint scope differs')
    if source_root:
        for entry in source_manifest['models']:
            if digest(source_root/entry['file'])!=entry['sha256']:raise ValueError('Source endpoint mutated during target fitting')
    clean(a.expected_commit);result={'schema':'paper_c_C8_fitting_stage_v1','source':a.expected_commit,'protocol_sha256':config_hash,'stage':a.stage,'direction':a.direction,'frame':frame,'models':endpoints,'optimizer_updates':updates,'complete':True,'source_checkpoints_sealed_before_target_stage':source_root is not None,'spectral_NPY_opens':len(data.opens),'spectral_NPY_paths':data.opens,'denied_opens':data.denied,'no_development_array_phase_available':True,'no_historical_weights_loaded':True,'wall_seconds':perf_counter()-began,'new_label_memberships':0,'query_reads':0};write(out/'manifest.json',result);print(__import__('json').dumps({key:result[key] for key in ('stage','direction','optimizer_updates','complete','spectral_NPY_opens','wall_seconds')}))


if __name__=='__main__':main()
