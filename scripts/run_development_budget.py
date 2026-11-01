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
from model.research_config import digest
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit',required=True)
    for name in ('development-root','protocol-root','data-root','out-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=(2036,2046,2056),default=2036)
    p.add_argument('--low-backbone-lr',action='store_true',help='Single full-finetune5000 run at1e-5; requires matching cached baseline')
    p.add_argument('--baseline-run',type=Path)
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
    if args.low_backbone_lr:
        if args.baseline_run is None:p.error('Cached baseline required')
        from train_spatial import SpatialConfig
        original=SpatialConfig(base)
        assert original.research.training.lr==5e-5 and original.research.training.head_lr==5e-4
        base['training']['lr']=1e-5
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False)
    receipt={'status':'running','code':code,'scope':'single_seed_paired_budget_validation_only','command':[sys.executable,*sys.argv]}
    write_json(out/'manifest.json',receipt)
    try:
        source_result=json.loads((args.source_run/'source/result.json').read_text())
        source_manifest=json.loads((args.source_run/'manifest.json').read_text())
        assert source_manifest['status']=='passed' and source_result['source_test_evaluated'] is False
        assert source_manifest['code']['commit']=='8fdab9b031727b1058e408cb8e2c3e5feee0a727'
        source_task_path=args.source_run/'frozen-task.json'
        source_task=json.loads(source_task_path.read_text())
        hashes[str(source_task_path)]=hash_file(source_task_path)
        assert source_task['task_hash']==digest({k:v for k,v in source_task.items() if k!='task_hash'})
        assert source_manifest['task_hash']==source_task['task_hash']
        assert source_task['config']['model_seed']==args.seed
        assert source_task['source_identity']==task['source_identity']
        assert source_task['target_identity']==task['target_identity']
        for name,key in [('model.pt','checkpoint_sha256'),('model.pt.json','sidecar_sha256')]:
            source_path=args.source_run/'source'/name
            assert hash_file(source_path)==source_result[key]
            hashes[str(source_path)]=source_result[key]
        source=load_model_checkpoint(args.source_run/'source/model.pt','cuda')
        receipt['source_checkpoint_sha256']=source_result['checkpoint_sha256']
        receipt['source_extra_updates']=10000
        baseline=None
        if args.low_backbone_lr:
            baseline_manifest=json.loads((args.baseline_run/'manifest.json').read_text())
            assert baseline_manifest['status']=='passed' and baseline_manifest['code']['commit']=='2c63ab7f42c480a1c37aea3238a9806926de76b4'
            assert baseline_manifest['development_hash']==record['validation_hash']
            assert baseline_manifest['source_checkpoint_sha256']==source_result['checkpoint_sha256']
            baseline_folder=args.baseline_run/'source_pretrained_full_finetune-5000'
            baseline_path=baseline_folder/'result.json';baseline=json.loads(baseline_path.read_text())
            hashes[str(baseline_path)]=hash_file(baseline_path)
            for name,key in [('model.pt','checkpoint_sha256'),('model.pt.json','sidecar_sha256')]:
                path=baseline_folder/name;assert hash_file(path)==baseline[key];hashes[str(path)]=baseline[key]
            assert baseline['steps']==5000 and baseline['development_hash']==record['validation_hash']
        runs=[]
        budgets=(5000,) if args.low_backbone_lr else (1000,5000)
        methods=('source_pretrained_full_finetune',) if args.low_backbone_lr else ('target_scratch_full','source_pretrained_full_finetune')
        write_json(out/'preregistered-settings.json',{'seed':args.seed,'budgets':budgets,'val_every':100,'methods':methods,'backbone_lr':1e-5 if args.low_backbone_lr else 5e-5,'head_lr':5e-4,'development_hash':record['validation_hash'],'test_evaluation':False})
        for method in methods:
            for budget in budgets:
                started=time.monotonic()
                result=adapt(scene,record,base,out/f'{method}-{budget}',method=method,source=source,steps=budget,val_every=100,device='cuda',seed=args.seed,progress=lambda event:write_json(out/'progress.json',event))
                best=min(result['validation'],key=lambda row:row['nll'])
                if baseline is not None:
                    assert result['initial_backbone_hash']==baseline['initial_backbone_hash']
                    assert result['initial_head_hash']==baseline['initial_head_hash']
                    assert [x['tile_id'] for x in result['trace']]==[x['tile_id'] for x in baseline['trace']]
                    receipt['baseline_result_sha256']=hashes[str(baseline_path)]
                runs.append({'method':method,'budget':budget,'selected':best,'wall_seconds':time.monotonic()-started,'initial_backbone_hash':result['initial_backbone_hash'],'initial_head_hash':result['initial_head_hash'],'sampled_tiles':[x['tile_id'] for x in result['trace']]})
                write_json(out/'runs.json',runs)
        assert len({r['initial_head_hash'] for r in runs})==1
        if not args.low_backbone_lr:
            for index in (0,2):
                assert runs[index]['sampled_tiles']==runs[index+1]['sampled_tiles'][:1000]
                assert runs[index]['initial_backbone_hash']==runs[index+1]['initial_backbone_hash']
        assert next(r for r in runs if r['method']=='source_pretrained_full_finetune')['initial_backbone_hash']==source_result['backbone_hash']
        receipt['runs']=[{k:v for k,v in r.items() if k!='sampled_tiles'} for r in runs]
        require_frozen_code(code)
        assert all(hash_file(x)==v for x,v in hashes.items()) and hash_file(image)==identity['image_sha256']
        receipt.update(status='passed',optimizer_steps=sum(budgets)*len(methods),validation_labels=45,test_evaluated=False,
                       raw_label_file_loaded=False,development_hash=record['validation_hash'])
    except BaseException as error:
        receipt.update(status='failed',error=repr(error));raise
    finally:write_json(out/'manifest.json',receipt)


if __name__=='__main__':main()
