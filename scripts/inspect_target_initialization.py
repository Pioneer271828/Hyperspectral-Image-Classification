"""Inspect pinned source-to-target initialization; optional unlabeled GPU chain."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from target_initialization import initialize_target,METHODS
from joint_baseline import NAMES,forward
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot,require_frozen_code,external_directory,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('expected-commit','out-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--target',choices=NAMES,required=True);p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--model-seed',type=int,required=True);p.add_argument('--head-seed',type=int,required=True)
    for k in ('source-path','source-weight-sha256','source-sidecar-sha256','observation','observation-sha256'):p.add_argument('--'+k)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    a=p.parse_args();source=code_snapshot()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:raise RuntimeError('Remote clean expected SHA required')
    if bool(a.observation)!=bool(a.observation_sha256):raise ValueError('Observation and pin must be provided together')
    if a.observation and a.method=='source_pretrained_head_only':
        raise ValueError('Head-only target BN must be calibrated before training/inference validation')
    model,record=initialize_target(a.target,a.method,model_seed=a.model_seed,head_seed=a.head_seed,
        source_path=a.source_path,source_weight_sha256=a.source_weight_sha256,
        source_sidecar_sha256=a.source_sidecar_sha256,device=a.device)
    check={'performed':False}
    if a.observation:
        if file_hash(a.observation)!=a.observation_sha256:raise ValueError('Observation artifact changed')
        item=torch.load(a.observation,map_location='cpu',weights_only=True)
        if item['name']!=a.target or any(k.startswith(('train_','val_','test_')) for k in item):raise ValueError('Observation target/label scope mismatch')
        moved={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in item.items()}
        probability=forward(model,a.target,moved)[0]
        loss=-probability[:,0].clamp_min(1e-8).log().mean();loss.backward()
        valid=any(p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
            for p in model.backbone.HGNN_Branch.parameters())
        if not torch.isfinite(loss) or not valid:raise ValueError('Target initialized graph chain is nonfinite/inactive')
        if file_hash(a.observation)!=a.observation_sha256:raise ValueError('Observation changed during chain check')
        check={'performed':True,'objective':'artificial_class0_not_gt','finite_graph_gradient':True,
            'observation_sha256':a.observation_sha256,'optimizer_updates':0,'model_test_scoring':False}
    require_frozen_code(source);out=external_directory(a.out_dir);out.mkdir(exist_ok=False)
    write_json(out/'receipt.json',{'status':'passed','source':source,'initialization':record,'chain_check':check})
    print('TARGET_INITIALIZATION_PASSED',a.target,a.method,flush=True)


if __name__=='__main__':main()
