"""Saved-output paired error analysis; no NN, significance test or new selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyse(audit_path,run,development):
    audit=json.loads(audit_path.read_text());setup=json.loads((run/'setup.json').read_text())
    if audit['status']!='PASSED':raise ValueError('Successful independent audit required')
    committed=json.loads((run/'summary.json').read_text())
    for name,h in committed['files'].items():
        if sha(run/name)!=h:raise ValueError('Original committed evidence differs')
    dev=json.loads(development.read_text());rec=dev['record'];width=dev['plan']['shape'][1]
    if setup['protected'].get(str(development))!=sha(development):raise ValueError('Released labels differ')
    lookup=dict(zip(rec['validation_support']['indices'],rec['validation_support']['labels']))
    by_parent={str(t['id']):t for t in rec['roles'] if t['role']=='devval'}
    saved={arm:np.load(run/f'round-0-{arm}.npz',allow_pickle=False) for arm in audit['compact_results']}
    trials={};contingencies=[];cohorts={}
    for i,row in enumerate(setup['requests']):
        key,view=row['key'],row['band'];tile=by_parent[row['parent']]
        y0,y1,x0,x1=tile['box']
        for y in range(y0,y1):
            for x in range(x0,x1):
                if y*width+x not in lookup:continue
                label=lookup[y*width+x];local=(y-y0)*(x1-x0)+x-x0
                for arm,arrays in saved.items():
                    trials.setdefault((view,label,arm),[]).append(int(arrays[f'p{i}'][local].argmax())==label-1)
        cohorts.setdefault(view,[]).append(key)
    for (view,label,arm),outcome in trials.items():
        for reference in ('light','full','content'):
            if reference==arm:continue
            a=np.asarray(outcome);b=np.asarray(trials[(view,label,reference)])
            counts=[int(np.sum(a&b)),int(np.sum(a&~b)),int(np.sum(~a&b)),int(np.sum(~a&~b))]
            contingencies.append({'view':view,'class_id':label,'candidate':arm,'reference':reference,
                                  'support':len(a),'both_correct':counts[0],'candidate_only_correct':counts[1],
                                  'reference_only_correct':counts[2],'both_wrong':counts[3],
                                  'recall_delta_pp':100*(counts[1]-counts[2])/len(a)})
    coverage=set(audit['compact_results']['content_coverage']['full_plan'])
    random=set(audit['compact_results']['matched_random']['full_plan'])
    saturation={view:{'requests':len(keys),'full_calls':len(set(keys)&coverage),
                      'all_or_none':len(set(keys)&coverage) in (0,len(keys))} for view,keys in cohorts.items()}
    value={'audit_sha256':sha(audit_path),'producer':audit['producer'],'scope':audit['scope'],
           'frozen_model_seed':audit['contract'].get('frozen_model_seed',2036),
           'original_method':audit['contract'].get('original_method',
                  'source_pretrained_full_finetune' if audit['contract']['profile']=='g3_pavia_xuzhou' else 'WP11 scratch different budget'),
           'descriptor':audit['contract']['descriptor'],'released_distinct_labels':len(lookup),
           'same_labels_reused_across_views':True,'independent_confirmation':False,
           'matched_random_identical_plan':coverage==random,'random_strata':saturation,
           'all_random_strata_saturated':all(x['all_or_none'] for x in saturation.values()),
           'paired_error_contingencies':contingencies,
           'metrics':{arm:row['metrics'] for arm,row in audit['compact_results'].items()},
           'timing':{k:audit[k] for k in ('latency_p50_ms','latency_p95_ms','graph_p50_ms','planning_p50_ms','loading_ms','setup_reference_ms')},
           'prohibition':'No new seed/arm/threshold is selected. Four views and six timing rounds do not multiply label support or independent repeats.'}
    for arrays in saved.values():arrays.close()
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',required=True)
    p.add_argument('--cases',nargs='+',required=True);p.add_argument('--development',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();suite=Path(a.suite);cases={}
    for case in a.cases:
        cases[case]=analyse(suite/(case+'-audit02.json'),suite/case,Path(a.development))
    result={'schema':'paper_c_saved_output_paired_analysis_v1','cases':cases,'new_neural_forwards':0,'new_optimizer_updates':0,
            'semantic_class_names':'Numeric1..9 only: original publisher table contradicts historical project names; no semantic remapping applied.',
            'statistical_tests':'none; fixed support/split and historically exposed context; no IID-pixel/seed significance claim'}
    methods=('target_scratch_full','source_pretrained_head_only','source_pretrained_full_finetune')
    index={(x['original_method'],x['frozen_model_seed']):x for x in cases.values() if x['descriptor']=='approximate_span'}
    if all((m,s) in index for m in methods for s in (2036,2046,2056)):
        paired=[]
        for candidate,reference in ((methods[1],methods[0]),(methods[2],methods[0]),(methods[2],methods[1])):
            for arm in index[(methods[0],2036)]['metrics']:
                for view in index[(methods[0],2036)]['metrics'][arm]:
                    differences=[];recall=[]
                    for seed in (2036,2046,2056):
                        x=index[(candidate,seed)]['metrics'][arm][view];y=index[(reference,seed)]['metrics'][arm][view]
                        differences.append(100*(x['oa']-y['oa']))
                        recall.append([100*(i-j) for i,j in zip(x['recall'],y['recall'])])
                    paired.append({'candidate':candidate,'reference':reference,'arm':arm,'view':view,
                                   'paired_OA_delta_pp':differences,'mean_delta_pp':float(np.mean(differences)),
                                   'sd_ddof1_pp':float(np.std(differences,ddof=1)),
                                   'paired_class_recall_delta_pp':recall,'distinct_target_labels':45,
                                   'source_cost':'Scratch0 source updates; both adapted methods inherit source10000; target1000/K10/V0 equal'})
        result['matched_original_method_comparisons']=paired
    target=Path(a.out)
    with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    rows=[{'case':case,**row} for case,value in cases.items() for row in value['paired_error_contingencies']]
    with target.with_suffix('.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'status':'PASSED','cases':len(cases),'paired_class_rows':len(rows),'neural_forwards':0,'optimizer_updates':0}))


if __name__=='__main__':main()
