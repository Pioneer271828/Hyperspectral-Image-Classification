"""Two-update real-data engineering check; no accuracy experiment or test evaluation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from research_runtime import code_snapshot,external_directory,hash_file,require_frozen_code,write_json
from loadData.observation_sampling import BandMetadata
from transfer.development_training import prepare_scene,adapt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit',required=True)
    for name in ('development-root','protocol-root','data-root','out-dir'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();code=code_snapshot()
    if sys.platform!='linux' or not Path('/.dockerenv').exists() or code['dirty'] or code['commit']!=args.expected_commit or sys.flags.optimize or not torch.cuda.is_available():
        p.error('Require container, CUDA, assertions and clean exact SHA')
    out=external_directory(args.out_dir)
    for path in (args.development_root.resolve(),args.protocol_root.resolve(),args.data_root.resolve()):
        if out==path or out in path.parents or path in out.parents:p.error('Output overlaps inputs')
    paths=[args.development_root/'manifest.json',args.development_root/'development.json',args.development_root/'parent-target-plan.json',args.protocol_root/'task.json']
    hashes={str(x):hash_file(x) for x in paths}
    manifest=json.loads(paths[0].read_text());record=json.loads(paths[1].read_text());plan=json.loads(paths[2].read_text());task=json.loads(paths[3].read_text())
    assert manifest['status']=='passed' and manifest['validation_hash']==record['validation_hash']
    assert manifest['parent_task_hash']==task['task_hash'] and task['target_plan_hash']==plan['plan_hash']
    identity=task['target_identity'];root=args.data_root.resolve(strict=True);image=(root/identity['image_path']).resolve(strict=True)
    assert image.is_relative_to(root) and hash_file(image)==identity['image_sha256']
    from scipy.io import loadmat
    from loadData.dataset_registry import DATASETS
    from loadData.data_read import readWavelengths
    cube=loadmat(image,variable_names=[DATASETS['xuzhou'][1]])[DATASETS['xuzhou'][1]]
    wave,_=readWavelengths('xuzhou',cube.shape[-1])
    bands=BandMetadata(identity['image_sha256'],tuple(range(len(wave))),tuple(wave),'nm','approximate',(True,)*len(wave))
    scene=prepare_scene(cube,bands,plan,record)
    base=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/screen_m6_none.json').read_text())
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False)
    receipt={'status':'running','code':code,'scope':'real_data_two_update_engineering_only','command':[sys.executable,*sys.argv]}
    write_json(out/'manifest.json',receipt)
    try:
        result=adapt(scene,record,base,out/'scratch',device='cuda')
        require_frozen_code(code)
        assert all(hash_file(x)==v for x,v in hashes.items()) and hash_file(image)==identity['image_sha256']
        receipt.update(status='passed',optimizer_steps=result['steps'],validation_labels=45,test_evaluated=False,
                       raw_label_file_loaded=False,development_hash=record['validation_hash'])
    except BaseException as error:
        receipt.update(status='failed',error=repr(error));raise
    finally:write_json(out/'manifest.json',receipt)


if __name__=='__main__':main()
