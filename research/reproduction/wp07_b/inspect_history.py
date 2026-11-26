"""Read existing published WP07 evidence; perform no model execution."""
import csv,json,hashlib
from pathlib import Path
r=Path('/home/lbx/hsi-pioneer271828/runs');o=r/'wp07-b-20261126'
h=r/'wp07-paviau-real-results-20261118.TC3eXtC5/aggregate/results'
j=json.loads((h/'comparison-audit.json').read_text())
assert hashlib.sha256((h/'comparison-audit.json').read_bytes()).hexdigest()=='728076ffe4751662cea2580e66c3b915d3fc5b30cecc1308ff1d761a3b8cc1bb'
assert j['case_count']==39 and j['optimizer_updates']==2340 and not j['query_accessed'] and not j['raw_gt_loaded']
rows=list(csv.DictReader((h/'endpoints.csv').open()));assert len(rows)==468
pairs=[]
idx={(row['seed'],row['factor'],row['arm'],row['endpoint'],row['head'],row['band']):row for row in rows}
for key,a in idx.items():
 s,f,arm,e,head,band=key
 if arm not in ['mean','content','metadata','joint']:continue
 base={'mean':'off','content':'mean','metadata':'mean','joint':'content'}[arm];b=idx[s,f,base,e,head,band]
 pairs.append(dict(seed=s,factor=f,candidate=arm,baseline=base,endpoint=e,head=head,band=band,**{k+'_delta':float(a[k])-float(b[k]) for k in ['oa','aa','nll']}))
with (o/'historical-primary-pairs.csv').open('x',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
matrix={'source_sha256':hashlib.sha256((h/'comparison-audit.json').read_bytes()).hexdigest(),'worker_commit':j['worker_commit'],'real_updates_reused':2340,'new_real_updates':0,'query_accessed':False,'raw_gt_loaded':False,'matrix':[{'scene':'paviaU','seeds':[64,512,2048],'factor':1,'arms':['off'],'steps':60,'events':[0,30,60],'train_points_per_class':5,'validation_points_per_class':5,'bands':[103,52]},{'scene':'paviaU','seeds':[64,512,2048],'factors':[2,4],'arms':['off','fixed','mean','content','metadata_missing','joint_missing'],'steps':60,'events':[0,30,60],'train_points_per_class':5,'validation_points_per_class':5,'bands':[103,52]},{'scene':'xuzhou','seeds':[64],'historical_protocol':'issue411 class-first 256 steps per full trajectory, also 7+249 recovery','arms':['off','cnn_fixed1','cnn_fixed2','cnn_fixed3','cnn_average','cnn_content','graph_average','both_content'],'real_updates_reused':4096,'same_protocol_as_B':False},{'scene':'paviaU/xuzhou/honghu','package':'band-scale-20261119','scope':'232 frozen forwards and 108 synthetic training cells','new_real_updates':0}]}
(o/'existing-matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')
print(json.dumps({'historical_endpoint_rows':len(rows),'derived_paired_rows':len(pairs),'new_optimizer_updates':0}))
