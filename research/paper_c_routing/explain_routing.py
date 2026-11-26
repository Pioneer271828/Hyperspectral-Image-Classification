"""Post-hoc score-scale diagnosis from saved metadata; no labels or NN."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',required=True)
    p.add_argument('--cases',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args();suite=Path(a.suite)
    records=[]
    for case in a.cases:
        root=suite/case;setup=json.loads((root/'setup.json').read_text())
        row=json.loads((root/'round-0-content_coverage.json').read_text())
        roster=setup['requests'];coverage={r['key']:r['coverage'] for r in roster};m=row['margin']
        selected=set(row['full_plan']);others=set(coverage)-selected
        if not all(coverage[i]<coverage[j] for i in selected for j in others):
            records.append({'case':case,'certificate_applies':False,
                            'reason':'Selected requests are not strictly separated by coverage alone; no complete-cohort separation claim is made',
                            'margin_range':[min(m.values()),max(m.values())]})
            continue
        boundary=max([0.]+[(m[i]-m[j])/(coverage[j]-coverage[i]) for i in selected for j in others])
        original_scores={k:1-m[k]+1-coverage[k] for k in coverage}
        minimum_gap=min(original_scores[i]-original_scores[j] for i in selected for j in others)
        if boundary>=1 or minimum_gap<=0:raise ValueError('Original unit coverage weight does not strictly separate cohorts')
        by_view={v:[r['key'] for r in roster if r['band']==v] for v in {r['band'] for r in roster}}
        records.append({'case':case,'certificate_applies':True,'descriptor':json.loads((root/'contract.json').read_text())['descriptor'],
                        'margin_range':[min(m.values()),max(m.values())],
                        'within_view_descriptor_constant':all(len({coverage[k] for k in keys})==1 for keys in by_view.values()),
                        'full_selected_views':sorted(v for v,keys in by_view.items() if set(keys)<=selected),
                        'partial_selected_views':sorted(v for v,keys in by_view.items() if 0<len(set(keys)&selected)<len(keys)),
                        'coverage_weight_strictly_above_this_locks_observed_cohorts':boundary,
                        'original_coverage_weight':1.,'smallest_selected_minus_unselected_score':minimum_gap,
                        'scope':'Sufficient separation for observed saved scores at quota10/20 only. Not a recommended operating weight or out-of-sample theorem.'})
    value={'schema':'paper_c_score_scale_diagnosis_v1','records':records,'label_values_read':0,
           'new_neural_forwards':0,'optimizer_updates':0,'post_hoc':True,
           'selection':'none; original six-arm policy and every result retained; no replacement weight is chosen',
           'conclusion':'For cases with a positive certificate, unit coverage weight forces complete observed-cohort selection; conditional matched random has no alternate eligible plan. Nonseparable cases are left explicit and require their own route/paired-output analysis.'}
    with Path(a.out).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps({'status':'EXPLAINED_OBSERVED_SATURATION','cases':len(records),'labels_read':0,'optimizer_updates':0}))


if __name__=='__main__':main()
