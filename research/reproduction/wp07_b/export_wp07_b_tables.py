"""Export exact integer class counts from independently replayed confusion matrices."""
import csv,json
from pathlib import Path
p=Path('/home/lbx/hsi-pioneer271828/runs/wp07-b-20261126');endpoints=[];classes=[]
for d in ['independent-gpu3','independent-gpu7']:
 for f in sorted((p/d).glob('xuzhou-*.json')):
  a=json.loads(f.read_text())
  for endpoint,data in a['endpoints'].items():
   for head,bands in data['metrics'].items():
    for band,m in bands.items():
     key={'case':a['case'],'seed':int(a['case'].split('/')[1]),'endpoint':endpoint,'step':62,'head':head,'band':band}
     endpoints.append({**key,**{k:m[k] for k in ['count','oa','aa','nll','kappa','macro_f1']}})
     for i,pc in enumerate(m['per_class']):
      correct=m['confusion_matrix'][i][i];support=pc['support'];assert support==5 and pc['recall']==correct/support
      classes.append({**key,'class_id':pc['class_id'],'correct':correct,'support':support,'recall':pc['recall']})
assert len(endpoints)==216 and len(classes)==1944
for name,rows in [('independent-endpoints.csv',endpoints),('per-class-correct.csv',classes)]:
 with (p/'results'/name).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'endpoints':len(endpoints),'integer_per_class_counts':len(classes),'new_optimizer_updates':0}))
