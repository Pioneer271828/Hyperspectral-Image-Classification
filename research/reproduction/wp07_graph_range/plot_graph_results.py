"""Plot raw paired seeds and label-free graph-path geometry; no training or selection."""
import csv,importlib.util,json,sys
from pathlib import Path
if sys.flags.optimize:raise RuntimeError('Plot verification refuses optimized Python')
p=Path(sys.argv[1])
if importlib.util.find_spec('matplotlib') is None:
 (p/'results/plots-not-produced.json').write_text(json.dumps({'reason':'matplotlib unavailable; complete numeric tables supplied'})+'\n');raise SystemExit(0)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
rows=list(csv.DictReader((p/'results/paired-endpoints.csv').open()));r=[v for v in rows if v['candidate']=='content' and v['baseline']=='mean' and v['head']=='fused' and v['endpoint']=='selected']
if len(r)!=6:raise ValueError('Expected six fused paired seed/band endpoints')
colors=['#3178a5','#d88730','#4b9475'];fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
for j,(band,title) in enumerate([('full','All 103 bands'),('even_retained_positions','Retained 52 bands')]):
 data=sorted([v for v in r if v['band']==band],key=lambda v:int(v['seed']))
 for i,(field,label) in enumerate([('nll_delta','NLL difference'),('oa_delta','OA difference (percentage points)')]):
  values=[float(v[field])*(100 if field=='oa_delta' else 1) for v in data];ax=axes[i,j];ax.bar([str(v['seed']) for v in data],values,color=colors,width=.55);ax.axhline(0,color='#333333',linewidth=.8);ax.set_title(title if i==0 else '');ax.set_ylabel(label);ax.set_xlabel('Initialization/order seed');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  if field=='nll_delta':ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,-3))
fig.suptitle('PaviaU scale50: content - mean, separately trained paired controls',fontsize=13)
fig.text(.5,-.025,'Same K5/V5 release; 45 validation points per band. Lower NLL is better. Three seeds share one spatial/support release.',ha='center',fontsize=9)
fig.savefig(p/'results/primary-pairs.png',dpi=180,bbox_inches='tight');plt.close(fig)
reach=json.loads((p/'reach-diagnostic-output/summary.json').read_text())['protocols'];names=['paviaU-factor1-scale50','xuzhou-factor1-scale100','paviaU-factor1-scale16-diagnostic','xuzhou-factor1-scale16-diagnostic'];labels=['PaviaU / 50','Xuzhou / 100','PaviaU / fine16','Xuzhou / fine16'];region=[100*reach[n]['views_with_changed_region_twohop_support']/reach[n]['views'] for n in names];pixel=[100*reach[n]['views_with_changed_Q_twohop_native_footprint']/reach[n]['views'] for n in names]
fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');x=np.arange(4);ax.bar(x-.18,region,width=.36,label='Two-hop region support differs',color='#3178a5');ax.bar(x+.18,pixel,width=.36,label='Q + two hops + broadcast coverage differs',color='#d88730');ax.set_xticks(x,labels);ax.set_ylim(0,115);ax.set_ylabel('Views with different structural coverage (%)');ax.legend(loc='upper center',bbox_to_anchor=(.5,1.18),ncol=2,fontsize=9);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
for i,n in enumerate(names):ax.text(i,104,f"n={reach[n]['views']}",ha='center',fontsize=9)
fig.suptitle('Label-free graph-path support: operator difference is not pixel coverage expansion',fontsize=12)
fig.text(.5,-.04,'Positive-weight structural upper bound relative to graph-input pixel features; excludes upstream normalization/CNN and nonlinear gates.\nFine16 is an initial-geometry pilot, with no new training or score-based choice.',ha='center',fontsize=8.5)
fig.savefig(p/'results/structural-coverage.png',dpi=180,bbox_inches='tight');plt.close(fig)
print(json.dumps({'plots':['primary-pairs.png','structural-coverage.png'],'training_updates':0}))
