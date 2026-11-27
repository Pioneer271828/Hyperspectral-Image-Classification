"""Metadata-only allocation support and admission; no model/array dependencies."""
import argparse
from collections import defaultdict,Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess

ARMS=('light','full','content','content_coverage','permuted_coverage','matched_random')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def token(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def alias_support(multiplicities,k):
    # Coefficient of product(1+x+...+x**m): aliases do not create new inputs.
    counts=[1]+[0]*k
    for m in multiplicities:
        counts=[sum(counts[j-a] for a in range(min(m,j)+1)) for j in range(k+1)]
    return counts[k]


def support(requests,family):
    groups=defaultdict(list)
    for r in requests:
        if family=='legacy':key=(r['view'],tuple(r['shape'] or []))
        elif family=='shape_only':key=tuple(r['shape'] or [])
        elif family=='same_context_count_span':key=(r['context'],r['K'],r['minimum_column'],r['maximum_column'])
        else:raise ValueError('Unknown prespecified null family')
        groups[key].append(r)
    rows=[];total=distinct=1;variation={'count':False,'span':False,'internal_mask':False}
    for key,rs in sorted(groups.items(),key=lambda x:str(x[0])):
        n=len(rs);k=sum(r['full_selected']['content_coverage'] for r in rs);ways=math.comb(n,k)
        aliases=Counter((r['context'],r['mask_sha256']) for r in rs)
        unique=alias_support(aliases.values(),k);total*=ways;distinct*=unique
        if 0<k<n:
            for feature,field in (('count','K'),('span','endpoint_span_columns'),('internal_mask','mask_sha256')):
                variation[feature]|=len({r[field] for r in rs})>1
        rows.append({'cohort_id':token([family,key])[:20],'n':n,'k':k,'assignments':str(ways),
                     'alias_quotient_assignments':str(unique),'distinct_inputs':len(aliases)})
    return {'family':family,'cohorts':rows,'assignments':str(total),'alias_quotient_assignments':str(distinct),
            'workload_shape_verified':all(r['shape'] is not None for r in requests),
            'missing_shape_limit':'For old shape_only counts this is an unstratified numerical benchmark, not verified shape matching. Legacy saturation survives every shape refinement.' if any(r['shape'] is None for r in requests) else None,
            'multiple_assignments':total>1,'feature_varies_within_randomized_strata':variation,
            'limit':'Uniform legal allocation is a hypothetical allocation benchmark, not a randomized acquisition or causal device experiment.'}


def admit(case):
    legacy=case['nulls']['legacy'];internal=case['nulls']['same_context_count_span'];reasons=[]
    if not case['native_retained_column_map_verified']:reasons.append('NATIVE_COLUMN_MAPPING_UNVERIFIED')
    if case['wavelength_quality']!='calibrated':reasons.append('CENTERS_APPROXIMATE_NO_SRF_BINDING')
    if not legacy['multiple_assignments']:reasons.append('LEGACY_RANDOMIZATION_FORCED')
    if not internal['feature_varies_within_randomized_strata']['internal_mask']:reasons.append('NO_DISTINCT_SAME_COUNT_SPAN_INTERNAL_MASKS')
    if not case['router_reads_internal_geometry']:reasons.append('OLD_ROUTER_DESCRIPTOR_ONLY_SPAN')
    return {'admitted':not reasons,'reasons':reasons,'stage':'BLOCKED_BEFORE_MODEL_ARRAY_OR_LABEL_READ' if reasons else 'METADATA_QUALIFIED_REQUIRES_SEPARATE_FROZEN_EVALUATION_REGISTRATION',
            'new_neural_calls':0,'new_optimizer_updates':0,'independent_confirmation':False}


def case_metadata(folder,old=False):
    folder=Path(folder);summary=json.loads((folder/'summary.json').read_text())
    setup=json.loads((folder/'setup.json').read_text());contract=json.loads((folder/'contract.json').read_text())
    for name in ('setup.json','contract.json'):
        if summary['files'][name]!=sha(folder/name):raise ValueError('Committed metadata differs')
    q=contract['qualification'];target=q['target'];source=q['source'];raw=setup['requests']
    parents=sorted({r['parent'] for r in raw});anonymous={p:f'context-{i:04d}' for i,p in enumerate(parents)}
    plans={}
    for arm in ARMS:
        f=folder/('round-0-'+arm+'.json')
        if sha(f)!=summary['files'][f.name]:raise ValueError('Committed route decision differs')
        row=json.loads(f.read_text());plans[arm]=set(row['full_plan'])
        if len(plans[arm])!=row['counts']['full_calls']:raise ValueError('Actual heavy count differs')
    n=max(r['bands'] for r in raw);requests=[]
    for r in raw:
        view=r['band'];k=r['bands']
        if view.startswith('stride'):mask=list(range(0,n,int(view[6:])))
        elif view=='range_le1000nm':mask=list(range(k))
        else:raise ValueError('Unreviewed mask recipe')
        if len(mask)!=k:raise ValueError('Pinned column recipe disagrees with recorded count')
        context=anonymous[r['parent']];gaps=[b-a for a,b in zip(mask,mask[1:])]
        requests.append({'request_id':token([target,context,view])[:20],'context':context,'view':view,'shape':r.get('shape'),
            'K':k,'native_delivered_columns':n,'minimum_column':mask[0],'maximum_column':mask[-1],
            'endpoint_span_columns':mask[-1]-mask[0],'maximum_internal_gap_columns':max(gaps,default=0),
            'mask_indices':mask,'mask_sha256':token(mask),'recorded_approximate_span':r['coverage'],
            'mask_quality':'reconstructed pinned delivered-column recipe; INDEX_STRESS_NOT_NATIVE_SENSOR',
            'full_selected':{arm:r['key'] in chosen for arm,chosen in plans.items()}})
    if any(not chosen<=set(r['key'] for r in raw) for chosen in plans.values()):raise ValueError('Unlisted request in plan')
    case={'case':folder.name,'source':source,'target':target,'profile':'old_G3_source10000_target1000' if old else 'native_source3_target3',
        'model_seed':contract.get('frozen_model_seed') if old else int(folder.name.rsplit('-s',1)[1]),
        'router_seed':contract['seed'],'producer':contract['code'],'contract_sha256':sha(folder/'contract.json'),
        'setup_sha256':sha(folder/'setup.json'),'model_state_pins':setup['models'],
        'role':'previously_exposed_devval','calibration_released':False,'confirmation_released':False,
        'shared_semantic_map_active':False,'native_retained_column_map_verified':False,'wavelength_quality':'approximate',
        'device_scene_confounding':'one acquisition per admitted product; model seeds/context tiles are not independent devices',
        'router_reads_internal_geometry':False,'requests':requests,
        'old_shape_limit':'Original setup lacks shape; view-saturated groups remain saturated under every shape refinement.' if old else None}
    case['nulls']={family:support(requests,family) for family in ('legacy','shape_only','same_context_count_span')}
    if old and case['nulls']['legacy']['multiple_assignments']:raise ValueError('Cannot infer shape-refined support from missing old shape metadata')
    case['admission']=admit(case);return case


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('expected-commit','routes','old-routes','out'):p.add_argument('--'+n,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=a.expected_commit or subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():raise ValueError('Clean full source SHA required')
    cases=[case_metadata(f) for f in sorted(Path(a.routes).iterdir()) if f.is_dir() and (f/'contract.json').exists()]
    cases += [case_metadata(f,True) for f in sorted(Path(a.old_routes).glob('g3-original-*-source_pretrained_full_finetune-span05'))]
    if len(cases)!=18:raise ValueError('Exactly fifteen new and three old cases required')
    value={'schema':'paper_c_unlabeled_identifiability_v1','producer':a.expected_commit,'cases':cases,
        'admitted_native_evaluation_cases':sum(c['admission']['admitted'] for c in cases),
        'no_model_weight_observation_coordinate_label_or_probability_arrays_opened':True,
        'read_scope':'Committed setup and routing JSON; only roster descriptors and actual route bits retained. Confidence/metrics are not used or exported.',
        'new_neural_calls':0,'new_optimizer_updates':0,'new_query_reads':0,'confirmation_directions':0}
    out=Path(a.out).resolve()
    if out.is_relative_to(repo):raise ValueError('External artifact path required')
    with out.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps({'cases':len(cases),'native_admitted':value['admitted_native_evaluation_cases'],
        'legacy_support_one':sum(not c['nulls']['legacy']['multiple_assignments'] for c in cases),'new_neural_calls':0}))


if __name__=='__main__':main()
