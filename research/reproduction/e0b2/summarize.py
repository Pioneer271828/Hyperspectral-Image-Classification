"""Read-only compact E0-B2 reporting; run on the training container."""
import json
from pathlib import Path
from statistics import mean, stdev

base=Path('/home/lbx/hsi-pioneer271828/runs/e0b2-issue131-analysis02')
r=json.loads((base/'results.json').read_text())
q=json.loads((base/'independent.json').read_text())
assert q['status']=='passed' and q['metric_cases']==1134 and q['branch_cases']==378
assert len(r['contrasts'])==648 and len(r['class_changes'])==240
assert set(r['rows'])=={f'{m}-{s}' for m in ('none','capacity_control','x_residual') for s in (64,512,2048)}
def stats(v):
    if any(x is None for x in v):return dict(values=[None if x is None else 100*x for x in v],mean=None,sample_sd=None)
    v=[100*x for x in v]
    return dict(values=v,mean=mean(v),sample_sd=stdev(v))
summary={'schema_version':1,'issue':131,'audit':q,'fused':[],'branches':[],
         'contrasts':r['contrasts'],'class_changes':r['class_changes'],
         'historical_head_checks':r['historical_head_checks'],'training_files_unchanged':r['training_files_unchanged'],
         'scope':r['scope']}
for n in ('paviaU','xuzhou','whu-honghu'):
 for policy in ('within_split_only','whole_annotation_posthoc_only'):
  for w in ('1','2','3'):
   for region in ('boundary','observed_nonboundary'):
    for m in ('none','capacity_control','x_residual'):
     rows=[r['rows'][f'{m}-{s}'][n][policy]['widths'][w][region] for s in (64,512,2048)]
     fused=rows[0]['heads']['fused']
     summary['fused'].append(dict(dataset=n,policy=policy,width=int(w),region=region,mode=m,count=fused['count'],
                                  classes=fused['macro_class_ids'],**{k:stats([a['heads']['fused'][k] for a in rows]) for k in ('oa','aa','macro_f1')}))
     if policy=='within_split_only':
      b=[a['branches'] for a in rows]
      summary['branches'].append(dict(dataset=n,width=int(w),region=region,mode=m,
         **{h:stats([a['heads'][h]['oa'] for a in rows]) for h in ('cnn','graph','fused')},
         both_wrong=stats([a['strata']['both_wrong']['count']/a['count'] if a['count'] else None for a in b]),
         correction=stats([a['correction']['rate'] for a in b])))
with (base/'summary.json').open('x') as f:json.dump(summary,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
for row in summary['fused']:
 if row['policy']=='within_split_only' and row['region']=='boundary':
  print(row['dataset'],row['width'],row['mode'],'n=',row['count'], {k:(row[k]['mean'],row[k]['sample_sd']) for k in ('oa','aa','macro_f1')})
for row in summary['branches']:
 if row['dataset']=='whu-honghu' and row['width']==1:print('BRANCH',row)
for row in r['class_changes']:
 if row['small_by_train_support_le25'] and row['head']=='fused' and row['reference']=='none':print('SMALL',row)
print('QA',q)
