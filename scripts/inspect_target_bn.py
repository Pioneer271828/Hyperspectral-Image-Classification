"""Calibrate a pinned single-target initializer on a pinned label-free graph."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from joint_baseline import NAMES,_item_fingerprint
from target_initialization import initialize_target,state_id
from target_bn_calibration import calibrate_target,verify_target_binding,OBS_KEYS,POLICY,CONTEXT,GRAPH_POLICY
from model.checkpoint_io import save_model_checkpoint,load_model_checkpoint
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot,require_frozen_code,external_directory,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('expected-commit','out-dir','source-path','source-weight-sha256','source-sidecar-sha256',
              'observation','observation-record','observation-record-sha256'):p.add_argument('--'+k,required=True)
    p.add_argument('--target',choices=NAMES,required=True);p.add_argument('--mode',choices=('none',POLICY),default='none')
    p.add_argument('--model-seed',type=int,default=64);p.add_argument('--head-seed',type=int,default=1064)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    a=p.parse_args();source=code_snapshot()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:raise RuntimeError('Remote clean expected SHA required')
    if file_hash(a.observation_record)!=a.observation_record_sha256:raise ValueError('Graph record changed')
    graph=json.loads(Path(a.observation_record).read_text())
    if (graph['policy']!=GRAPH_POLICY or graph['target']!=a.target or graph['graph_fit_labels']!=0
            or file_hash(a.observation)!=graph['observation_sha256']):raise ValueError('Label-free target graph identity mismatch')
    item=torch.load(a.observation,map_location='cpu',weights_only=True)
    if _item_fingerprint(item)!=graph['prepared_fields'] or any(k.startswith(('train_','val_','test_')) for k in item):
        raise ValueError('Target observation fields/label access changed')
    observation={k:item[k] for k in OBS_KEYS}
    model,initial=initialize_target(a.target,'source_pretrained_head_only',model_seed=a.model_seed,head_seed=a.head_seed,
        source_path=a.source_path,source_weight_sha256=a.source_weight_sha256,source_sidecar_sha256=a.source_sidecar_sha256,device=a.device)
    model.eval();before=state_id(model)
    if a.device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    calibrated,receipt=calibrate_target(model,observation,mode=a.mode,context=CONTEXT,graph_policy=GRAPH_POLICY,source_state_sha256=before)
    if a.device=='cuda':torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    if state_id(model)!=before:raise ValueError('Original initializer mutated')
    if a.mode!= 'none':verify_target_binding(calibrated,observation,context=CONTEXT,graph_policy=GRAPH_POLICY,source_state_sha256=before)
    out=external_directory(a.out_dir);out.mkdir(exist_ok=False);save_model_checkpoint(calibrated,out/'calibrated.pt')
    loaded=load_model_checkpoint(out/'calibrated.pt',device=a.device)
    if state_id(loaded)!=state_id(calibrated):raise ValueError('Calibrated state roundtrip differs')
    if a.mode!='none':verify_target_binding(loaded,observation,context=CONTEXT,graph_policy=GRAPH_POLICY,source_state_sha256=before)
    require_frozen_code(source)
    pins={a.source_path:a.source_weight_sha256,a.source_path+'.json':a.source_sidecar_sha256,
        a.observation:graph['observation_sha256'],a.observation_record:a.observation_record_sha256}
    if any(file_hash(p)!=h for p,h in pins.items()):raise ValueError('Protected source/observation input changed')
    write_json(out/'receipt.json',{'status':'passed','source':source,'target':a.target,'mode':a.mode,
        'initialization':initial,'calibration':receipt,'elapsed_seconds':elapsed,
        'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
        'original_initializer_unchanged':True,'protected_files':pins,
        'artifacts':{p.name:file_hash(p) for p in out.iterdir() if p.is_file()},
        'optimizer_updates':0,'model_test_scoring':False})
    print('TARGET_BN_CALIBRATION_PASSED',a.target,a.mode,flush=True)


if __name__=='__main__':main()
