"""Frozen paired target-budget experiment; validation selection, zero external test evaluation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from research_runtime import code_snapshot,external_directory,hash_file,require_frozen_code,write_json
from loadData.observation_sampling import BandMetadata
from transfer.development_training import prepare_scene,adapt
from model.checkpoint_io import load_model_checkpoint
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit',required=True)
    for name in ('development-root','protocol-root','data-root','out-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-run',type=Path,required=True)
    args=p.parse_args();code=code_snapshot()
    if sys.platform!='linux' or not Path('/.dockerenv').exists() or code['dirty'] or code['commit']!=args.expected_commit or sys.flags.optimize or not torch.cuda.is_available():
        p.error('Require container, CUDA, assertions and clean exact SHA')
    out=external_directory(args.out_dir)
    for path in (args.development_root.resolve(),args.protocol_root.resolve(),args.data_root.resolve(),args.source_run.resolve()):
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
    receipt={'status':'running','code':code,'scope':'single_seed_paired_budget_validation_only','command':[sys.executable,*sys.argv]}
    write_json(out/'manifest.json',receipt)
    try:
        source_result=json.loads((args.source_run/'source/result.json').read_text())
        source_manifest=json.loads((args.source_run/'manifest.json').read_text())
        assert source_manifest['status']=='passed' and source_result['source_test_evaluated'] is False
        assert source_manifest['code']['commit']=='8fdab9b031727b1058e408cb8e2c3e5feee0a727'
        assert source_manifest['task_hash']=='22345436b815a9a85fa75b421bb8a6796767e159261417b61ddf958d7321969a'
        for name,key in [('model.pt','checkpoint_sha256'),('model.pt.json','sidecar_sha256')]:
            source_path=args.source_run/'source'/name
            assert hash_file(source_path)==source_result[key]
            hashes[str(source_path)]=source_result[key]
        source=load_model_checkpoint(args.source_run/'source/model.pt','cuda')
        receipt['source_checkpoint_sha256']=source_result['checkpoint_sha256']
        receipt['source_extra_updates']=10000
        runs=[]
        write_json(out/'preregistered-settings.json',{'seed':2036,'budgets':[1000,5000],'val_every':100,'methods':['target_scratch_full','source_pretrained_full_finetune'],'development_hash':record['validation_hash'],'test_evaluation':False})
        for method in ('target_scratch_full','source_pretrained_full_finetune'):
            for budget in (1000,5000):
                started=time.monotonic()
                result=adapt(scene,record,base,out/f'{method}-{budget}',method=method,source=source,steps=budget,val_every=100,device='cuda',progress=lambda event:write_json(out/'progress.json',event))
                best=min(result['validation'],key=lambda row:row['nll'])
                runs.append({'method':method,'budget':budget,'selected':best,'wall_seconds':time.monotonic()-started,'initial_backbone_hash':result['initial_backbone_hash'],'initial_head_hash':result['initial_head_hash'],'sampled_tiles':[x['tile_id'] for x in result['trace']]})
                write_json(out/'runs.json',runs)
        assert len({r['initial_head_hash'] for r in runs})==1
        for index in (0,2):
            assert runs[index]['sampled_tiles']==runs[index+1]['sampled_tiles'][:1000]
            assert runs[index]['initial_backbone_hash']==runs[index+1]['initial_backbone_hash']
        assert runs[2]['initial_backbone_hash']==source_result['backbone_hash']
        receipt['runs']=[{k:v for k,v in r.items() if k!='sampled_tiles'} for r in runs]
        require_frozen_code(code)
        assert all(hash_file(x)==v for x,v in hashes.items()) and hash_file(image)==identity['image_sha256']
        receipt.update(status='passed',optimizer_steps=12000,validation_labels=45,test_evaluated=False,
                       raw_label_file_loaded=False,development_hash=record['validation_hash'])
    except BaseException as error:
        receipt.update(status='failed',error=repr(error));raise
    finally:write_json(out/'manifest.json',receipt)


if __name__=='__main__':main()
