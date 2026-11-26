"""Collect independently audited new-direction development and saturation evidence."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--routes',required=True);p.add_argument('--audits',required=True)
    p.add_argument('--lineage-audit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    routes=Path(a.routes);audits=Path(a.audits);lineage=json.loads(Path(a.lineage_audit).read_text());rows=[];certificates=[];counts=defaultdict(int)
    for r in lineage['cases']:
        case=r['case'];root=routes/case;independent=json.loads((audits/(case+'.json')).read_text())
        if independent['status']!='PASSED' or independent['stages']!=36 or independent['case']!=case:raise ValueError('Complete independent case audit required')
        setup=json.loads((root/'setup.json').read_text());summary=json.loads((root/'summary.json').read_text())
        for key,value in setup['reference_counts'].items():counts[key]+=value
        for stage in summary['results']:
            for key,value in stage['counts'].items():counts[key]+=value
        for arm in ('light','full','content','content_coverage','permuted_coverage','matched_random'):
            stages=[s for s in summary['results'] if s['arm']==arm]
            if len(stages)!=6 or any(s['metrics']!=stages[0]['metrics'] for s in stages):raise ValueError('Frozen model-repeat metrics differ')
            row={'case':case,'source':r['source'],'target':r['target'],'seed':r['seed'],'arm':arm,
                'pipeline_p50_ms':float(np.median([s['pipeline_ms'] for s in stages])),
                'pipeline_p90_ms':float(np.quantile([s['pipeline_ms'] for s in stages],.9)),
                'graph_ms_median':float(np.median([s['graph_ms'] for s in stages])),
                'full_calls':stages[0]['counts']['full_calls'],'light_calls':stages[0]['counts']['light_calls'],
                'metrics':stages[0]['metrics'],'target_scratch_metrics':r['target_control_selected_metrics'],
                'full_transfer_minus_scratch_OA_pp':r['transfer_minus_scratch_OA_pp'],
                'source_updates_inherited':r['source_inherited_executed_updates'],'target_updates':r['committed_optimizer_updates'],
                'source_selected_step':r['source_selected_step'],'target_selected_step':r['selected_step']}
            rows.append(row)
        # Certificate uses confidence/metadata only, never labels or accuracy.
        cov=json.loads((root/'round-0-content_coverage.json').read_text());selected=set(cov['full_plan']);margins=cov['margin'];roster=setup['requests']
        chosen=[x['coverage'] for x in roster if x['key'] in selected];other=[x['coverage'] for x in roster if x['key'] not in selected]
        gap=min(other)-max(chosen);margin_range=max(margins.values())-min(margins.values());cohorts=defaultdict(list)
        for x in roster:cohorts[x['band'],tuple(x['shape']),x['coverage']].append(x['key'])
        saturated=all(len(set(keys)&selected) in (0,len(keys)) for keys in cohorts.values())
        certificates.append({'case':case,'chosen_max_coverage':max(chosen),'unchosen_min_coverage':min(other),'coverage_gap':gap,
            'whole_tile_margin_range':margin_range,'sufficient_coverage_dominance':gap>margin_range,
            'all_matched_cohorts_zero_or_all_full':saturated,
            'random_plan_forced_for_every_random_seed':saturated,
            'observed_random_equals_coverage':independent['coverage_matched_random_plans_identical'],
            'range_noop_requests':independent['range_noop_requests'],'label_values_used_by_certificate':0})
    if len(rows)!=90 or len(certificates)!=15:raise ValueError('Exactly fifteen complete six-arm cases required')
    value={'schema':'paper_c_five_direction_development_collection_v1','cases':15,'stages':540,'rows':rows,'certificates':certificates,
        'runtime_calls_including_paid_setup':dict(counts),'new_routing_optimizer_updates':0,'query_reads':0,'confirmation':False,
        'all_random_controls_structurally_forced':all(c['random_plan_forced_for_every_random_seed'] for c in certificates),
        'limits':['Original G3 results are separate from this native3/target3 engineering protocol.',
            'Three training seeds are not independent roles/geographies/sensors; six timing rounds reuse each frozen model and point set.',
            'No-op <=1000nm cuts remain explicitly labeled for VNIR products; they are not real camera/SRF interventions.',
            'Resident shared-CPU batch timings exclude cold setup, data/code hash validation, training, audit and backup; no isolated/GPU speed claim.']}
    out=Path(a.out)
    with out.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    fields=['case','source','target','seed','arm','view','OA','AA','pipeline_p50_ms','pipeline_p90_ms','full_calls','light_calls']
    with out.with_suffix('.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            for view,m in row['metrics'].items():
                w.writerow({**{k:row[k] for k in fields if k in row},'view':view,'OA':m['oa'],'AA':m['aa']})
    print(json.dumps({'cases':15,'stages':540,'runtime_calls_including_setup':dict(counts),
        'all_random_controls_structurally_forced':value['all_random_controls_structurally_forced']}))


if __name__=='__main__':main()
