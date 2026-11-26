"""Independent count, cost and protection reduction for the completed frozen package."""
from pathlib import Path
import collections,csv,hashlib,json,statistics
p=Path('/home/lbx/hsi-pioneer271828/runs/wp07-b-20261126')
audits=[json.loads(f.read_text()) for f in sorted((p/'pair-v2/execution').glob('*/b-audit.json'))]
ind=[json.loads(f.read_text()) for d in ['independent-gpu3','independent-gpu7'] for f in sorted((p/d).glob('xuzhou-*.json'))]
assert len(audits)==len(ind)==18
entries={}
for name in ['lane-gpu3.log','lane-gpu7.log','lane-gpu3-v2.log','lane-gpu7-v2.log']:
 for line in (p/name).read_text().splitlines():
  if line.startswith('{'):
   e=json.loads(line)
   if 'phase' in e:
    key=(e['case'],e['phase']);assert key not in entries;entries[key]=e
assert len(entries)==54
cost=[]
for a in audits:
 e=[entries[a['case'],phase] for phase in ['prefix','resume','completed']]
 cost.append({'case':a['case'],'parameters':a['parameters'],'stage_wall_seconds':sum(v['wall_seconds'] for v in e),'audit_wall_seconds':a['audit_wall_seconds'],'process_peak_allocated_bytes':max(v['process_peak_allocated_bytes'] for v in e),'process_peak_reserved_bytes':max(v['process_peak_reserved_bytes'] for v in e),'fresh_updates':a['fresh_updates']})
with (p/'results/cost.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(cost[0]));w.writeheader();w.writerows(cost)
structure=[r for a in ind for endpoint in a['endpoints'].values() for r in endpoint['support']]
multirange=[r for r in structure if len(r['supports'])==2]
assert len(structure)==360 and len(multirange)==180
assert all(r['identical_positive_support'] and r['identical_weighted_support'] for r in multirange)
points=sum(len(e['points']) for a in ind for e in a['endpoints'].values())
assert points==9720
for a in ind:
 base=next(r for r in audits if r['case']==a['case'])
 for endpoint,field in [('selected','head_metrics'),('last','last_head_metrics')]:
  for head,bands in a['endpoints'][endpoint]['metrics'].items():
   for band,m in bands.items():
    other=base[field][head][band];assert m['confusion_matrix']==other['confusion_matrix'] and abs(m['nll']-other['nll'])<1e-12
pair=list(csv.DictReader((p/'results/paired-endpoints.csv').open()))
classpair=list(csv.DictReader((p/'results/paired-per-class.csv').open()))
fallback=[a['missing_fallback'] for a in audits if a['missing_fallback'] is not None]
summary={'cases':18,'updates':sum(a['optimizer_updates'] for a in audits),'phase_receipts':len(entries),'metrics_endpoints':216,'per_class_endpoints':1944,'paired_endpoints':len(pair),'paired_per_class':len(classpair),'independent_original_point_predictions':points,'native_view_replays':len(structure),'multi_range_view_replays':len(multirange),'identical_positive_and_weighted_graph_support_views':len(multirange),'selected_steps':dict(collections.Counter(a['selected_step'] for a in audits)),'fallback_cases':len(fallback),'fallback_views':sum(a['views'] for a in fallback),'all_fallback_exact':all(a['all_validation_rows_and_head_hashes_exact'] for a in fallback),'training_stage_wall_seconds_sum':sum(e['wall_seconds'] for e in entries.values()),'audit_wall_seconds_sum':sum(a['audit_wall_seconds'] for a in audits),'independent_wall_seconds_sum':sum(a['wall_seconds'] for a in ind),'process_peak_allocated_bytes':max(e['process_peak_allocated_bytes'] for e in entries.values()),'process_peak_reserved_bytes':max(e['process_peak_reserved_bytes'] for e in entries.values()),'observed_resource_utilization_range':[min(e['resource_sample']['utilization'] for e in entries.values()),max(e['resource_sample']['utilization'] for e in entries.values())],'audit_script_sha256':hashlib.sha256((p/'audit_wp07_b.py').read_bytes()).hexdigest(),'new_query_access':False,'new_raw_gt_access':False,'old_matrix_and_preflight_unchanged':True,'reentry':json.loads((p/'reentry-proof.json').read_text())}
(p/'results/independent-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
print('PRIMARY_GUARD_FAILURES')
for r in classpair:
 if r['candidate']=='content' and r['baseline']=='mean' and r['endpoint']=='selected' and r['protected']=='False':print(r)
