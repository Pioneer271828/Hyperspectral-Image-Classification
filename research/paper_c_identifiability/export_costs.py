"""Export existing profile/binding scalar costs to D; zero new profiling."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('profile','binding','out'):p.add_argument('--'+n,required=True)
    a=p.parse_args();profile=json.loads(Path(a.profile).read_text());binding=json.loads(Path(a.binding).read_text())
    rows=[r for r in profile['rows'] if r['scene']=='paviaU']
    if len(rows)!=2 or {r['method'] for r in rows}!={'native','dsnet'}:raise ValueError('Original Pavia paired profile required')
    value={'schema':'paper_c_existing_pavia_cost_export_v1','profile_producer':profile['code'],
        'profile_sha256':sha(a.profile),'binding_sha256':sha(a.binding),'scope':'existing synthetic28x28, one point/class, CPU1; not real convergence or deployment throughput',
        'rows':[{k:r[k] for k in ('method','bands','classes','synthetic_shape','checkpoint_max_bytes','checkpoint_median_bytes',
                  'component_median_seconds','proposed_full40_updates_all_three_seeds','proposed_full40_checkpoint_bytes_estimate')} for r in rows],
        'existing_pavia_role_cost_binding':next(r['costs'] for r in binding['scenes'] if r['scene']=='paviaU'),
        'selection_sweeps':{'engineering3':[0,1,2,3],'proposed40':[0,5,10,15,20,25,30,35,40]},
        'updates_per_sweep':30,'three_seed_two_method_full40_updates':7200,
        'prefix_adoption':'NOT_PROVEN_AND_NOT_AUTHORIZED: contracts/selection history/modes differ; no cold splice or budget deduction.',
        'selection_model_buffer_guard':True,'selection_RNG_preservation_guard':False,
        'new_profiles':0,'new_model_constructions':0,'new_neural_calls':0,'new_optimizer_updates':0,'new_label_or_query_reads':0}
    with Path(a.out).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps({'status':'EXPORTED_EXISTING_SCALAR_COSTS_ONLY','profile_sha256':value['profile_sha256'],'new_profiles':0}))


if __name__=='__main__':main()
