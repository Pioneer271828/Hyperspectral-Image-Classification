"""Twelve-update G4 target support loop; no query labels or full-training profile."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_training import run_training
from joint_baseline import NAMES
from target_initialization import METHODS
from target_data_context import load_target_context
from target_training import target_settings,TargetRuntime
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot,require_frozen_code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('expected-commit','out-dir','observation','observation-record','observation-record-sha256',
              'label-family','label-family-sha256'):p.add_argument('--'+k,required=True)
    for k in ('source-path','source-weight-sha256','source-sidecar-sha256'):p.add_argument('--'+k)
    p.add_argument('--target',choices=NAMES,required=True);p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--model-seed',type=int,default=64);p.add_argument('--support-seed',type=int,default=64)
    p.add_argument('--support-per-class',type=int,choices=(1,5,10),default=10)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int)
    a=p.parse_args();source=code_snapshot()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:
        raise RuntimeError('Clean remote expected SHA required')
    settings=target_settings(a.target,a.method,seed=a.model_seed,support_seed=a.support_seed,
        per_class=a.support_per_class,device=a.device)
    items,provenance,guard=load_target_context(a.target,a.observation,a.observation_record,a.observation_record_sha256,
        a.label_family,a.label_family_sha256,per_class=a.support_per_class,support_seed=a.support_seed)
    source_spec={}
    if a.method!=METHODS[0]:
        source_spec={'source_path':a.source_path,'source_weight_sha256':a.source_weight_sha256,
            'source_sidecar_sha256':a.source_sidecar_sha256}
        if any(v is None for v in source_spec.values()):raise ValueError('Pinned source checkpoint pair required')
        provenance['protected_files'].update({a.source_path:a.source_weight_sha256,a.source_path+'.json':a.source_sidecar_sha256})
    elif any(v is not None for v in (a.source_path,a.source_weight_sha256,a.source_sidecar_sha256)):
        raise ValueError('Scratch must not receive source inputs')
    provenance['source_spec']=source_spec
    def finish():
        guard();require_frozen_code(source)
        if any(file_hash(p)!=h for p,h in provenance['protected_files'].items()):raise ValueError('Protected source input changed')
    finish()
    runtime=TargetRuntime(settings,items,a.out_dir,source_spec=source_spec)
    result=run_training(settings,items,a.out_dir,provenance,source,resume=a.resume,stop_after=a.stop_after,
        device=a.device,finish_guard=finish,runtime_adapter=runtime)
    print('TARGET_SHORT',a.target,a.method,result['operation'],result['updates_this_call'],flush=True)


if __name__=='__main__':main()
