"""Read-only all-case FP32 repeat/FP64 reference audit; no test scoring."""
import argparse,json,os,sys,time
from pathlib import Path
AUDIT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(AUDIT))
from numerical_audit import compare,verify_exact_double_cast
PRODUCER=Path('/home/lbx/hsi-pioneer271828/checkouts/m4s2-issue135');BASE=PRODUCER.parents[1]
sys.path.insert(0,str(PRODUCER));os.chdir(PRODUCER)
import torch
from research_runtime import code_snapshot,write_json
from joint_training import import_preparation,load_plan
from joint_supervision import SupervisionRuntime
from loadData.dataset_registry import file_hash


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True)
 p.add_argument('--slot',type=int,choices=[0,1],required=True);a=p.parse_args()
 consumer=code_snapshot(AUDIT);source=code_snapshot(PRODUCER)
 assert consumer['commit']==a.expected_commit and not consumer['dirty']
 batch=BASE/'runs/m4s2-issue135-full01';launch=json.loads((batch/'launch.json').read_text())
 assert source==launch['code'] and not source['dirty']
 assert all(consumer['files'].get(n)==h for n,h in source['files'].items())
 assert json.loads((batch/'status.json').read_text())['status']=='training_completed'
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 context={'torch':torch.__version__,'deterministic':True,'cudnn_deterministic':True,
          'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
          'cublas_workspace':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'device':torch.cuda.get_device_name(0)}
 out=BASE/'runs/num1-issue141-audit01';out.mkdir(exist_ok=True)
 receipt=out/f'slot-{a.slot}.json';assert not receipt.exists()
 protected={str(p):file_hash(p) for c in launch['cases'] for p in Path(c['out']).iterdir() if p.is_file()}
 assert all(file_hash(p)==h for p,h in launch['inputs'].items())
 _,items,_=import_preparation(launch['prepared'],launch['data_root'],load_plan('configs/experiments/base97_training.json'),source)
 rows=[];started=time.monotonic()
 for case in launch['cases'][a.slot::2]:
  folder=Path(case['out']);r=json.loads((folder/'result.json').read_text());rt=SupervisionRuntime(case['mode'],case['supervision'])
  model=rt.load_selected(folder/'primary.pt',items,folder/f'raw-validation-{r["selected_step"]:06d}.pt',device='cuda')
  state={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
  saved=torch.load(folder/'selected-probabilities.pt',weights_only=True,map_location='cpu')
  moved={n:{k:v.to('cuda') if torch.is_tensor(v) else v for k,v in item.items()} for n,item in items.items()}
  _,first=rt.validation_snapshot(model,moved);_,second=rt.validation_snapshot(model,moved)
  assert all(torch.equal(state[n],v.cpu()) for n,v in model.state_dict().items())
  model.double();verify_exact_double_cast(state,model.state_dict())
  double_items={n:{k:v.double() if torch.is_tensor(v) and v.is_floating_point() else v for k,v in item.items()} for n,item in moved.items()}
  _,reference=rt.validation_snapshot(model,double_items)
  verify_exact_double_cast(state,model.state_dict())
  for name in items:
   for phase in ('train','val'):
    assert set(saved[name][phase])==set(first[name][phase])==set(reference[name][phase])
    for head in saved[name][phase]:
     for label,left,right in [('saved_vs_fp32',saved,first),('fp32_repeat',first,second),
                              ('saved_vs_fp64',saved,reference),('fp32_vs_fp64',first,reference)]:
      rows.append({'case':case['name'],'dataset':name,'phase':phase,'head':head,'comparison':label,
                   **compare(left[name][phase][head],right[name][phase][head])})
  write_json(out/f'slot-{a.slot}-progress.json',{'rows':rows,'test_scoring':False,'optimizer_updates':0})
  print('AUDITED',case['name'],flush=True)
  del model,moved,double_items,saved,first,second,reference,state;torch.cuda.empty_cache()
 assert code_snapshot(PRODUCER)==source and code_snapshot(AUDIT)==consumer
 assert all(file_hash(p)==h for p,h in protected.items()) and all(file_hash(p)==h for p,h in launch['inputs'].items())
 write_json(receipt,{'status':'completed_descriptive_audit','consumer':consumer,'producer':source,'context':context,
    'rows':rows,'script_sha256':file_hash(__file__),'protected_files_unchanged':len(protected),
    'inputs_unchanged':len(launch['inputs']),'optimizer_updates':0,'test_scoring':False,'elapsed_seconds':time.monotonic()-started})
 print('AUDIT_SLOT_COMPLETE',a.slot,len(rows),flush=True)


if __name__=='__main__':main()
