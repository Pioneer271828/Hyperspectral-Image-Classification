"""Prepare target image-only observations/graphs; no target labels or model."""
import argparse,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from joint_baseline import NAMES
from joint_training import load_plan
from target_observation import load_target_image,build_target_observation
from research_runtime import code_snapshot,require_frozen_code,external_directory,write_json
from loadData.dataset_registry import file_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('expected-commit','prepared-dir','data-root','out-dir'):p.add_argument('--'+k,required=True)
    p.add_argument('--target',choices=NAMES,required=True)
    a=p.parse_args();source=code_snapshot()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:raise RuntimeError('Remote clean expected SHA required')
    out=external_directory(a.out_dir);out.mkdir(exist_ok=False);start=time.monotonic()
    plan=load_plan('configs/experiments/base97_training.json')
    image,identity=load_target_image(a.prepared_dir,a.data_root,a.target,manifest_sha256=plan['prepared_manifest_sha256'])
    item,record=build_target_observation(image,a.target,64+NAMES.index(a.target))
    if any(k.startswith(('train_','val_','test_')) for k in item):raise ValueError('Label fields in observation artifact')
    if file_hash(identity['image_path'])!=identity['image_sha256']:raise ValueError('Target image changed during preparation')
    require_frozen_code(source);torch.save(item,out/'observation.pt')
    write_json(out/'record.json',{'status':'passed','source':source,**identity,**record,
        'observation_sha256':file_hash(out/'observation.pt'),'elapsed_seconds':time.monotonic()-start})
    print('TARGET_OBSERVATION_PASSED',a.target,record['regions'],flush=True)


if __name__=='__main__':main()
