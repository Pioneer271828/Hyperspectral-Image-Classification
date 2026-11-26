"""Independent bitset Q -> A^2 -> Q.T support and float64 strength diagnostics."""
import argparse,csv,hashlib,json,math,os,sys,time
from pathlib import Path
sys.path.insert(0,os.environ['WP07_GRAPH_SOURCE'])
import torch
from transfer.native_scale_campaign import read,packet
from transfer import registered_development as dev
from transfer.wp07_graph_range import require,require_checks,sparse_digest

THRESHOLDS=(0.,1e-12,1e-9,1e-6,1e-4,1e-3)

def bits(indices,values,n,m):
 rows=[0]*n
 for a,b,v in zip(indices[0],indices[1],values):
  if v>0:rows[a]|=1<<b
 return rows

def members(value):
 while value:
  b=value&-value;yield b.bit_length()-1;value-=b

def compose(rows,other):
 return [union(other[j] for j in members(r)) for r in rows]

def union(items):
 result=0
 for value in items:result|=value
 return result

def analyze(entry):
 q,a,b=(entry[k].coalesce() for k in ('q','near','wide'))
 require(all(x.dtype==torch.float32 for x in [q,a,b]),'Expected source float32 cached operands')
 n,m=q.shape
 qr=bits(q.indices().tolist(),q.values().tolist(),n,m)
 qt=bits(q.indices().flip(0).tolist(),q.values().tolist(),m,n)
 ar=bits(a.indices().tolist(),a.values().tolist(),m,m);br=bits(b.indices().tolist(),b.values().tolist(),m,m)
 aa=compose(ar,ar);bb=compose(br,br)
 left=compose(compose(qr,aa),qt);right=compose(compose(qr,bb),qt)
 require(all(mask==(1<<n)-1 for mask in left+right),'Selected scope no longer has complete structural graph-path coverage')
 Q=q.to_dense().double();A=a.to_dense().double();B=b.to_dense().double();R=Q.T/Q.sum(0).clamp_min(1e-6)[:,None]
 X=(Q@(A@A))@R;Y=(Q@(B@B))@R
 require(bool((X>0).all()) and bool((Y>0).all()),'Bitset and float64 positive supports differ')
 require(float((X.sum(1)-1).abs().max())<1e-5 and float((Y.sum(1)-1).abs().max())<1e-5,'Linear proxy row masses differ')
 rows=[]
 for t in THRESHOLDS:
  x=X>t;y=Y>t
  rows.append({'threshold':t,'near_min_count':int(x.sum(1).min()),'near_max_count':int(x.sum(1).max()),'wide_min_count':int(y.sum(1).min()),'wide_max_count':int(y.sum(1).max()),'changed_output_cells':int((x!=y).any(1).sum())})
 data={'native_cells':n,'regions':m,'region_twohop_changed_rows':sum(x!=y for x,y in zip(aa,bb)),
       'pixel_feature_graph_path_changed_rows':sum(x!=y for x,y in zip(left,right)),
       'near_all_native_cells_reachable':True,'wide_all_native_cells_reachable':True,
       'min_positive_Q':float(q.values()[q.values()>0].min()),'min_positive_near_A':float(a.values()[a.values()>0].min()),'min_positive_wide_A':float(b.values()[b.values()>0].min()),
       'min_near_proxy_weight':float(X.min()),'min_wide_proxy_weight':float(Y.min()),
       'max_abs_proxy_weight_delta':float((X-Y).abs().max()),'mean_row_l1_delta':float((X-Y).abs().sum(1).mean()),
       'q_indices_sha256':sparse_digest(q)['indices'],'q_values_sha256':sparse_digest(q)['values'],
       'near_indices_sha256':sparse_digest(a)['indices'],'near_values_sha256':sparse_digest(a)['values'],
       'wide_indices_sha256':sparse_digest(b)['indices'],'wide_values_sha256':sparse_digest(b)['values']}
 return data,rows

def self_test():
 q=torch.eye(3).to_sparse().coalesce();a=torch.tensor([[.5,.5,0.],[1/3,1/3,1/3],[0.,.5,.5]]).to_sparse().coalesce();b=(torch.ones(3,3)/3).to_sparse().coalesce()
 d,_=analyze({'q':q,'near':a,'wide':b});require(d['pixel_feature_graph_path_changed_rows']==0 and d['max_abs_proxy_weight_delta']>1e-5,'Support/strength separation fixture failed')

def main():
 require_checks();torch.set_num_threads(1);self_test();p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();root=args.root.resolve(strict=True);decision=read(dev._pin(args.geometry),root);selected=read(decision['protocols'][decision['selected_protocol']],root);cache=packet(selected['cache'],root);rows=[];threshold=[];start=time.perf_counter()
 for k,entry in cache['entries'].items():
  role,tile,band=k.split('/');common={'role':role,'tile':int(tile),'band':band};data,sweep=analyze(entry);rows.append({**common,**data});threshold.extend({**common,**r} for r in sweep)
 require(len(rows)==336,'Expected all selected PaviaU views');args.out.mkdir()
 for name,data in [('views.csv',rows),('threshold-sweep.csv',threshold)]:
  with (args.out/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 result={'status':'passed','views':336,'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'geometry':dev._pin(args.geometry),'cache':selected['cache'],'all_bitset_and_float64_supports_complete':True,'positive_threshold':0.0,'positive_test':'strict > 0; no epsilon edges inserted','source_operand_dtype':'float32','analysis_dtype':'float64','float64_normal_min':torch.finfo(torch.float64).tiny,'minimum_measured_proxy_weight':min(min(r['min_near_proxy_weight'],r['min_wide_proxy_weight']) for r in rows),'max_abs_weight_delta':max(r['max_abs_proxy_weight_delta'] for r in rows),'threshold_sweep':THRESHOLDS,'threshold_sweep_use':'fixed response-strength slices only, never structural admission or model/setting selection','elapsed_seconds':time.perf_counter()-start,'optimizer_updates':0,'labels_scored':0,'classifier_heads_evaluated':0,'scope':'graph path relative to its input pixel features; Q A^2 R with R=column-normalized Q.T. Excludes upstream CNN/spatial normalization dependencies and learned GNN channel weights, per-node LayerNorm and ReLU; not an entire-network raw-input receptive-field claim.'}
 (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
