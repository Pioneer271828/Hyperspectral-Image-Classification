"""Version new engineering eligibility without rewriting the frozen old catalog."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--table',required=True);p.add_argument('--collection',required=True)
    p.add_argument('--lineage',required=True);p.add_argument('--role-bindings',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    with Path(a.table).open(encoding='utf-8-sig') as f:original=list(csv.DictReader(f))
    collection=json.loads(Path(a.collection).read_text());lineage=json.loads(Path(a.lineage).read_text());roles=json.loads(Path(a.role_bindings).read_text())
    if len(original)!=30 or collection['cases']!=15 or collection['stages']!=540:raise ValueError('Complete original roster and new audited matrix required')
    profiles={};roster={(r['source'],r['target']) for r in lineage['cases']}
    for source,target in sorted(roster):
        rows=[r for r in lineage['cases'] if (r['source'],r['target'])==(source,target)]
        if sorted(r['seed'] for r in rows)!=[64,512,2048]:raise ValueError('Three fixed model seeds required per new direction')
        if any(not all(r[k] for k in ('source_backbone_exact','fresh_target_heads_exact','target_initial_rng_optimizer_exact')) for r in rows):raise ValueError('Lineage proof missing')
        profiles[source,target]={'profile':'NEW_NATIVE_SOURCE3_TARGET3_ENGINEERING','training_producer':lineage['producer'],
            'cases':[{'id':r['case'],'seed':r['seed'],'selected_step':r['selected_step'],'source_selected_step':r['source_selected_step'],
                      'source_inherited_updates':r['source_inherited_executed_updates'],'target_updates':r['committed_optimizer_updates'],
                      'contract_sha256':r['contract_sha256']} for r in rows],
            'target_K':5,'target_V':5,'light':'exact copied trained target CNN branch; no separate fitting',
            'model':'native learned local graph64/32 GroupNorm, original three-head/graph losses',
            'same_target_scratch_controls_verified':True,'input_bytes_and_light_full_endpoints_verified':True,
            'effectiveness_status':'NOT_ESTABLISHED_SHORT_ENGINEERING_BUDGET_AND_EXPOSED_DEVELOPMENT_ROLES'}
    out=[]
    for old in original:
        pair=old['source'],old['target'];ready=old['status']=='READY_DEVELOPMENT_ONLY';profile=profiles.get(pair)
        if pair==('paviaU','xuzhou'):
            profile={'profile':'PRIOR_FROZEN_ORIGINAL_G3_P2','producer':'afe4eaed2b51e70a37ce49a48a869227ef598014',
                'source_updates':10000,'target_updates':1000,'target_K':10,'original_target_V':0,
                'scoring_role':'historically released V5 recontextualization; development only',
                'comparison_limit':'Different budget/normalizer/model lineage from the five new engineering directions; do not pool macro scores.'}
        out.append({'source':pair[0],'target':pair[1],'original_data_gate':old['status'],
            'development_pair_available':ready and profile is not None,'profile':profile,
            'independent_calibration':False,'independent_confirmation':False,'query_access_unlocked':False,
            'remaining_data_device_role_gap':old['minimum_gap'],
            'semantic_scope':'Original target numeric class order with fresh target heads; no source-target class-ID equivalence or silent Xuzhou class rename.'})
    value={'schema':'paper_c_versioned_30_direction_engineering_eligibility_v3','directions':out,
        'new_engineering_directions':len(profiles),'development_pairs_across_noncomparable_profiles':sum(r['development_pair_available'] for r in out),
        'new_roles_or_query_authorized':0,'formal_confirmation_directions':0,
        'role_bindings':roles['scenes'],
        'device_limit':'Published sensor families are evidence for acquisitions, not retained-band/SRF/serial calibration. All current wave metadata remain approximate.',
        'permission_limit':'Eligibility stays within original three-scene released K5/V5 banks; no new labels, full GT or confirmation access.'}
    with Path(a.out).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps({k:value[k] for k in ('new_engineering_directions','development_pairs_across_noncomparable_profiles','formal_confirmation_directions')}))


if __name__=='__main__':main()
