"""Label-free two-hop/Q reach and linear mixing proxy; no classifier or learned GNN weights."""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['WP07_GRAPH_SOURCE'])
import torch
from research_runtime import code_snapshot,require_frozen_code
from transfer.native_scale_campaign import read,packet
from transfer import registered_development as dev,wp07_graph_range as g
from transfer.wp07_graph_training import BankProvider
from transfer.native_scale_observation import from_bank
from transfer.development_block_probe import VIEWS
from transfer.g3_training import state_hash

def measure(entry):
 q=entry['q'].to_dense().double();a=entry['near'].to_dense().double();b=entry['wide'].to_dense().double()
 ra=a@a;rb=b@b;pool=q.T/q.sum(0).clamp_min(1e-6)[:,None]
 x=(q@ra)@pool;y=(q@rb)@pool
 g.require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()),'Nonfinite mixing proxy')
 mx=x>0;my=y>0
 return {'nodes':q.shape[1],'native_cells':q.shape[0],
  'region_twohop_support_changed_rows':int(((ra>0)!=(rb>0)).any(1).sum()),
  'Q_twohop_native_footprint_changed_cells':int((mx!=my).any(1).sum()),
  'near_native_coverage_min':int(mx.sum(1).min()),'near_native_coverage_max':int(mx.sum(1).max()),
  'wide_native_coverage_min':int(my.sum(1).min()),'wide_native_coverage_max':int(my.sum(1).max()),
  'linear_proxy_max_abs_weight_delta':float((x-y).abs().max()),
  'linear_proxy_mean_row_l1_delta':float((x-y).abs().sum(1).mean())}

def main():
 g.require_checks();p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 code=code_snapshot();require_frozen_code(code);torch.set_num_threads(1);root=a.root.resolve(strict=True);geometry=read(dev._pin(a.geometry),root);rows=[];started=time.perf_counter()
 for name in geometry['protocol_order']:
  proto=read(geometry['protocols'][name],root);cache=packet(proto['cache'],root)
  for k,e in cache['entries'].items():
   role,tile,band=k.split('/');rows.append({'protocol':name,'role':role,'tile':int(tile),'band':band,**measure(e)})
  print(json.dumps({'done':name,'views':len(cache['entries'])}),flush=True)
 # Fine-grid rows are supplementary initial geometry only, never a new training release or a score-based choice.
 for scene,factor in [('paviaU',1),('xuzhou',1),('paviaU',2),('xuzhou',2)]:
  name=f'{scene}-factor{factor}-scale16-diagnostic'
  original=read(geometry['protocols'][f'{scene}-factor1-scale'+('50' if scene=='paviaU' else '100')],root)
  receipt=read(original['preflight'],root);provider=BankProvider(root,receipt,64);model=g.initialize(provider.baseline(64),7385);before=state_hash(model);count=0
  for role in ['devtrain','devval']:
   for row in provider._bank.role_rows(role):
    for band,spec in VIEWS.items():
     actual,_=from_bank(provider._bank,role,row['id'],spec,factor,code['commit']);entry=g.capture_geometry(model,actual,code['commit'],role,16)
     rows.append({'protocol':name,'role':role,'tile':row['id'],'band':band,**measure(entry)});count+=1
  g.require(state_hash(model)==before,'Geometry changed initial model');print(json.dumps({'done':name,'views':count}),flush=True)
 a.out.mkdir()
 with (a.out/'reach.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary={}
 for protocol in dict.fromkeys(r['protocol'] for r in rows):
  v=[r for r in rows if r['protocol']==protocol]
  summary[protocol]={'views':len(v),'views_with_changed_region_twohop_support':sum(r['region_twohop_support_changed_rows']>0 for r in v),'views_with_changed_Q_twohop_native_footprint':sum(r['Q_twohop_native_footprint_changed_cells']>0 for r in v),'max_linear_proxy_weight_delta':max(r['linear_proxy_max_abs_weight_delta'] for r in v)}
 result={'status':'passed','worker_commit':code['commit'],'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'protocols':summary,'elapsed_seconds':time.perf_counter()-started,'optimizer_updates':0,'labels_scored':0,'classifier_heads_evaluated':0,'scope':'positive structural reach upper bound and Q*A^2*column_normalized_QT linear proxy; excludes GNN channel weights, normalization and nonlinear gates; scale16 diagnostic is not a training release'}
 (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
