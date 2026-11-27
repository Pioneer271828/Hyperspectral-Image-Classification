"""Read development arrays only after every registered fitting endpoint is sealed."""
import argparse
import hashlib
import hmac
import itertools
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from c8_bridge import clean,fresh,write,read,digest,canonical,sha,protocol
from phase_data import PhaseData,load_cache
from features import REPRESENTATIONS
from models import predict
from metrics import measures,sensitivity,paired,cluster_weights,percentile


def verified_stages(experiment,producer):
    cfg=protocol();ph=sha(canonical(cfg));result=[];updates=0;models=0
    for source,target in cfg['directions']:
        for stage in ('source','target'):
            folder=experiment/f'{stage}-{source}-to-{target}';m=read(folder/'manifest.json')
            if not m['complete'] or m['source']!=producer or m['protocol_sha256']!=ph or m['direction']!=[source,target] or m['stage']!=stage or not m['no_development_array_phase_available'] or m['denied_opens']:raise ValueError('All registered fitting stages must be sealed before evaluation')
            for entry in m['models']:
                if digest(folder/entry['file'])!=entry['sha256']:raise ValueError('Fitting endpoint changed')
            updates+=m['optimizer_updates'];models+=len(m['models']);result.append((folder,m))
    if updates!=3840 or models!=52:raise ValueError('Full frozen fitting budget not completed')
    return result


def comparison_rows(rows):
    result=[]
    for source,target in protocol()['directions']:
        for rep in REPRESENTATIONS:
            for seed in protocol()['seeds']:
                group={r['method']:r for r in rows if r['direction']==[source,target] and r['representation']==rep and r['seed']==seed and r['scene']==target}
                for a,b in [('source_fit_transfer','target_scratch64'),('target_adapt','target_scratch64'),('target_adapt','target_scratch128')]:result.append(paired(group[a],group[b]))
        for seed in protocol()['seeds']:
            for method in ('source_fit_transfer','target_adapt','target_scratch64','target_scratch128'):
                group={r['representation']:r for r in rows if r['direction']==[source,target] and r['seed']==seed and r['method']==method};result.append(paired(group['raw_distribution'],group['legacy_scene_distribution']))
    for scene in ('xuzhou','whu-honghu'):
        for rep in REPRESENTATIONS:
            transfer=next(r for r in rows if r['scene']==scene and r['method']=='centroid_transfer' and r['representation']==rep)
            native=next(r for r in rows if r['scene']==scene and r['method']=='centroid_same_device' and r['representation']==rep);result.append(paired(transfer,native))
    return result


def direction_rows(rows):
    result=[]
    for rep in REPRESENTATIONS:
        for method in ('source_fit_transfer','target_adapt','target_scratch64','target_scratch128'):
            groups=[[r for r in rows if r['direction']==direction and r['method']==method and r['representation']==rep] for direction in protocol()['directions']]
            if any(len(g)!=3 for g in groups):raise ValueError('All fixed seeds required for directional summary')
            correct=[np.mean([np.asarray(r['predictions'])==r['truth'] for r in g],axis=0) for g in groups];draws=[cluster_weights(g[0]['parent_ids']) for g in groups]
            differences=[float(wa@correct[0]/wa.sum()-wb@correct[1]/wb.sum()) for wa,wb in itertools.product(*draws)]
            per_seed=[groups[0][i]['metrics']['OA']-groups[1][i]['metrics']['OA'] for i in range(3)]
            result.append({'representation':rep,'method':method,'direction_A':'xuzhou_to_whu-honghu','direction_B':'whu-honghu_to_xuzhou','OA_A_minus_B':float(correct[0].mean()-correct[1].mean()),'per_seed_differences':per_seed,'observed_parent_cross_product_draws':len(differences),'descriptive_range':percentile(differences),'point_pairing':False,'causal_sensor_interpretation':False})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--producer-commit');p.add_argument('--experiment',required=True);p.add_argument('--c7-root',required=True);p.add_argument('--inputs',required=True);p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=fresh(a.expected_commit,a.out);out.mkdir();began=perf_counter();producer=a.producer_commit or a.expected_commit;stages=verified_stages(Path(a.experiment),producer)
    key=os.urandom(32);keyfile=out/'pseudonymization.key';keyfile.write_bytes(key);keyfile.chmod(0o600)
    alias=lambda kind,scene,value:hmac.new(key,canonical([kind,scene,int(value)]),hashlib.sha256).hexdigest()[:20]
    loader=PhaseData(expected_commit=a.expected_commit,c7_root=a.c7_root,inputs=a.inputs,plan=a.plan,direction=('xuzhou','whu-honghu'),allowed={'source_development','target_development'});frames={}
    for scene,phase in [('xuzhou','source_development'),('whu-honghu','target_development')]:
        loader.load(phase,out/f'features-{scene}');frames[scene]=load_cache(out/f'features-{scene}')[0]
    rows=[]
    for folder,manifest in stages:
        source,target=manifest['direction']
        for entry in manifest['models']:
            model=read(folder/entry['file']);rep=entry['representation'];evaluations=[(target,entry['arm'])] if manifest['stage']=='target' else [(source,'source_same_device' if entry['kind']=='logistic' else 'centroid_same_device'),(target,'source_fit_transfer' if entry['kind']=='logistic' else 'centroid_transfer')]
            for scene,method in evaluations:
                data=frames[scene];z,predp,pred=predict(model,data[rep]);y=data['y'];identity=f'{source}_to_{target}/{rep}/{entry["seed"]}/{method}'
                row={'id':identity,'direction':[source,target],'scene':scene,'method':method,'representation':rep,'seed':entry['seed'],'model_kind':entry['kind'],'fit_scene':model['fit_scene'],'fit_labeled_points':10,'model_file':str(folder/entry['file']),'model_sha256':entry['sha256'],'point_ids':data['point_ids'].tolist(),'parent_ids':data['parent_ids'].tolist(),'point_aliases':[alias('point',scene,i) for i in data['point_ids']],'parent_aliases':[alias('parent',scene,i) for i in data['parent_ids']],'truth':y.tolist(),'logits':z.tolist(),'probabilities':predp.tolist(),'predictions':pred.tolist(),'metrics':measures(y,z,predp,pred),'target_scene_statistics_in_numeric_features':rep=='legacy_scene_distribution' and scene==target,'target_support_used_to_fit_relative_to_recorded_direction':manifest['stage']=='target','full_pipeline_G2_claim':False};row['parent_sensitivity']=sensitivity(row);rows.append(row)
    if len(rows)!=68 or sum(len(r['truth']) for r in rows)!=680:raise ValueError('Registered endpoint evaluation coverage differs')
    identities=[]
    for source,target in protocol()['directions']:
        source_folder=Path(a.experiment)/f'source-{source}-to-{target}';opposite_target=Path(a.experiment)/f'target-{target}-to-{source}'
        for rep in REPRESENTATIONS:
            for seed in protocol()['seeds']:
                left=read(source_folder/f'{rep}-s{seed}-source_fit.json');right=read(opposite_target/f'{rep}-s{seed}-target_scratch64.json')
                if not np.array_equal(left['theta'],right['theta']):raise ValueError('Matched same-device/source-versus-scratch identity failed')
                identities.append({'scene':source,'representation':rep,'seed':seed,'identical':True})
    verified_stages(Path(a.experiment),producer);clean(a.expected_commit);result={'schema':'paper_c_C8_private_development_evaluation_v1','evidence_kind':'real_development','evaluator':a.expected_commit,'producer':producer,'protocol_sha256':sha(canonical(protocol())),'all_fits_sealed_before_development_arrays':True,'rows':rows,'paired_comparisons':comparison_rows(rows),'direction_differences':direction_rows(rows),'matched_same_device_identity_checks':identities,'unique_development_points':20,'prediction_presentations':680,'linear_endpoint_count':48,'centroid_endpoint_count':4,'producer_optimizer_updates':3840,'development_NPY_opens':len(loader.opens),'wall_seconds':perf_counter()-began,'selection_used_target_development':False,'new_label_memberships':0,'query_reads':0};write(out/'evaluation.json',result);print(__import__('json').dumps({'evaluations':len(rows),'unique_development_points':20,'prediction_presentations':680,'matched_identity_checks':len(identities),'producer_optimizer_updates':3840,'target_validation_model_selection':False}))


if __name__=='__main__':main()
