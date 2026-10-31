"""Independent frozen G3 campaign: preflight, synthetic smoke, or declared full run."""
import argparse
import copy
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import torch
from model.research_config import digest
from research_runtime import code_snapshot,require_frozen_code,external_directory,write_json,hash_file
from loadData.transfer_protocol import validate_config,_validate_record,make_support,freeze_task
from loadData.spatial_tiles import make_tile_plan,observe_tile
from loadData.observation_sampling import BandMetadata,sample
from transfer.g3_training import Scene,METHODS,train_source,adapt_target,evaluate_target
ROOT=Path(__file__).resolve().parent


def read_protocol(root):
    root=Path(root).resolve(strict=True)
    manifest=json.loads((root/'manifest.json').read_text());task=json.loads((root/'task.json').read_text())
    config=json.loads((root/'resolved-config.json').read_text());validate_config(config)
    if manifest['status']!='passed' or task['task_hash']!=manifest['task_hash'] or task['task_hash']!=digest({k:v for k,v in task.items() if k!='task_hash'}) or task['config']!=config:
        raise ValueError('Frozen protocol manifest/task mismatch')
    plans={k:json.loads((root/(k+'-plan.json')).read_text()) for k in ('source','target')}
    support=json.loads((root/'support-10.json').read_text());_validate_record(support,plans['target'])
    if freeze_task(config,task['source_identity'],task['target_identity'],plans['source'],plans['target'],support,task['sensor_evidence'])!=task:
        raise ValueError('Task content differs from protocol contract')
    files={str(p):hash_file(p) for p in root.glob('*.json') if p.name!='delivery-check.json'}
    return task,plans,support,files


def load_scene(data_root,identity,plan,*,with_labels):
    from scipy.io import loadmat
    from loadData.dataset_registry import DATASETS
    from loadData.data_read import readWavelengths
    root=Path(data_root).resolve(strict=True);paths={}
    for kind in ('image','label'):
        path=(root/identity[kind+'_path']).resolve(strict=True)
        if not path.is_relative_to(root) or hash_file(path)!=identity[kind+'_sha256']:raise ValueError('Dataset file identity mismatch')
        paths[kind]=path
    key=DATASETS[identity['name']][1];cube=loadmat(paths['image'],variable_names=[key])[key]
    wave,_=readWavelengths(identity['name'],cube.shape[-1])
    bands=BandMetadata(identity['image_sha256'],tuple(range(len(wave))),tuple(wave),'nm','approximate',(True,)*len(wave))
    scene=Scene.prepare(identity['name'],cube,plan,bands,identity['class_count'])
    labels=loadmat(paths['label'],variable_names=[key+'_gt'])[key+'_gt'] if with_labels else None
    return scene,labels


def synthetic_task(config):
    rng=np.random.default_rng(8);shape=(96,96);plan=make_tile_plan(shape,32,2,config['split_seed'])
    labels=np.resize(np.arange(1,10),shape).astype(np.int64)
    scenes=[]
    for name,b in [('paviaU',6),('xuzhou',8)]:
        cube=rng.normal(size=(*shape,b)).astype(np.float32)
        bands=BandMetadata('synthetic-'+name,tuple(range(b)),tuple(np.linspace(400,1100,b)),'nm','approximate',(True,)*b)
        scenes.append(Scene.prepare(name,cube,plan,bands,9))
    support=make_support(labels,plan,list(range(1,10)),10,config['support_seed'])
    return scenes[0],labels,scenes[1],labels,support


def campaign(task,plans,support,base,out,*,device,mode,data_root=None,progress=None):
    config=task['config'];task_hash=task['task_hash']
    if mode=='smoke':
        source,source_labels,target,final_labels,support=synthetic_task(config)
        base=copy.deepcopy(base);base['model'].update(latent_channels=8,adapter_embed_dim=8,adapter_checkpoint_chunks=False)
        source_steps=target_steps=2;val_every=1
    else:
        # Target pixels/label values are not loaded into the source stage.
        source,source_labels=load_scene(data_root,task['source_identity'],plans['source'],with_labels=True)
        target=final_labels=None;source_steps=config['source_updates'];target_steps=config['target_updates'];val_every=100
    if mode=='preflight':
        target,_=load_scene(data_root,task['target_identity'],plans['target'],with_labels=False)
        result={'status':'passed','scope':'preflight_no_training','task_hash':task_hash,'source_bands':len(source.bands.band_ids),
                'target_bands':len(target.bands.band_ids),'source_statistics_hash':source.stats['statistics_hash'],
                'target_statistics_hash':target.stats['statistics_hash'],'support_labels':len(support['indices']),
                'target_label_values_loaded':False,'optimizer_steps':0}
        for role,scene in [('source',source),('target',target)]:
            tile=next(t for t in scene.plan['tiles'] if t['partition']=='train')
            _,record=sample(observe_tile(scene.cube,scene.plan,scene.stats,tile),scene.bands,{'mode':'none'})
            write_json(out/(role+'-metadata.json'),{'bands':scene.bands.__dict__,'record':record})
        write_json(out/'preflight.json',result);return result
    selected,source_result=train_source(source,source_labels,base,out/'source',steps=source_steps,val_every=val_every,
        model_seed=config['model_seed'],sampler_seed=config['source_sampler_seed'],device=device,task_hash=task_hash,progress=progress)
    # Source is fully selected before loading target inputs/normalization.
    del source,source_labels
    if target is None:target,_=load_scene(data_root,task['target_identity'],plans['target'],with_labels=False)
    results={}
    for method in METHODS:
        result=adapt_target(target,support,base,out/method,method=method,source=selected,steps=target_steps,
            model_seed=config['model_seed'],head_seed=config['target_head_seed'],sampler_seed=config['target_sampler_seed'],
            device=device,task_hash=task_hash,progress=progress)
        results[method]=result
        expected=source_result['initial_backbone_hash'] if method=='target_scratch_full' else source_result['backbone_hash']
        if result['initial_backbone_hash']!=expected:raise AssertionError('Target backbone initialization/transfer mismatch')
    if len({r['initial_head_hash'] for r in results.values()})!=1:raise AssertionError('Target heads did not share fresh initialization')
    if len({tuple(e['tile_id'] for e in r['trace']) for r in results.values()})!=1:raise AssertionError('Target sampled blocks differ')
    write_json(out/'target-checkpoints-frozen.json',{k:{'checkpoint_sha256':v['checkpoint_sha256'],'sidecar_sha256':v['sidecar_sha256']} for k,v in results.items()})
    # Only now may target final-test label values enter evaluation. No target validation call exists.
    if final_labels is None:
        from scipy.io import loadmat
        from loadData.dataset_registry import DATASETS
        identity=task['target_identity'];path=Path(data_root)/identity['label_path']
        if hash_file(path)!=identity['label_sha256']:raise ValueError('Target test-label file changed')
        key=DATASETS[identity['name']][1]+'_gt';final_labels=loadmat(path,variable_names=[key])[key]
    for method,result in results.items():
        metrics=evaluate_target(target,support,final_labels,out/method/'model.pt',expected_checkpoint_hash=result['checkpoint_sha256'],
            expected_sidecar_hash=result['sidecar_sha256'],task_hash=task_hash,method=method,device=device)
        write_json(out/method/'test.json',metrics);result['test']=metrics
    summary={'status':'completed','scope':'synthetic_short_smoke' if mode=='smoke' else 'frozen_G3_pilot',
             'source':source_result,'targets':results,'target_test_after_all_checkpoints_frozen':True,
             'actual_source_updates':source_steps,'actual_target_updates_per_method':target_steps,'target_validation_labels':0}
    write_json(out/'summary.json',summary);return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('preflight','smoke','run'));p.add_argument('--expected-commit',required=True)
    p.add_argument('--protocol-root',type=Path);p.add_argument('--data-root',type=Path);p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args();code=code_snapshot()
    if sys.platform!='linux' or not Path('/.dockerenv').exists() or sys.flags.optimize or code['dirty'] or code['commit']!=args.expected_commit:
        p.error('Require training container, assertions and clean exact SHA')
    if args.mode!='preflight' and not torch.cuda.is_available():p.error('CUDA required; no silent fallback')
    frozen={};out=external_directory(args.out_dir)
    if args.mode=='smoke':
        config=json.loads((ROOT/'configs/protocols/paviau_to_xuzhou_g3.json').read_text(encoding='utf-8-sig'))
        task={'config':config,'task_hash':'synthetic-smoke-'+digest(config)};plans=support=None
    else:
        if args.protocol_root is None or args.data_root is None:p.error('Frozen protocol and dataset root required')
        for path in (args.protocol_root.resolve(),args.data_root.resolve()):
            if path==out or path in out.parents or out in path.parents:p.error('Keep outputs separate from inputs')
        task,plans,support,frozen=read_protocol(args.protocol_root)
        for role in ('source','target'):
            identity=task[role+'_identity']
            for kind in ('image','label'):
                path=(args.data_root/identity[kind+'_path']).resolve(strict=True)
                if not path.is_relative_to(args.data_root.resolve(strict=True)):p.error('Dataset path escapes root')
                frozen[str(path)]=identity[kind+'_sha256']
    base=json.loads((ROOT/'configs/experiments/screen_m6_none.json').read_text())
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    manifest={'status':'running','code':code,'mode':args.mode,'task_hash':task['task_hash'],'command':[sys.executable,*sys.argv]}
    write_json(out/'manifest.json',manifest);write_json(out/'frozen-task.json',task)
    def progress(event):
        write_json(out/'progress.json',event)
        with (out/'progress.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
    try:
        result=campaign(task,plans,support,base,out,device='cuda' if args.mode!='preflight' else 'cpu',mode=args.mode,data_root=args.data_root,progress=progress)
        require_frozen_code(code)
        if any(hash_file(Path(path))!=sha for path,sha in frozen.items()):raise ValueError('Frozen protocol artifacts changed')
        manifest.update(status='passed',scope=result['scope'],source_updates=result.get('actual_source_updates',0),
                        target_updates_per_method=result.get('actual_target_updates_per_method',0))
        print(json.dumps(manifest),flush=True)
    except BaseException:
        manifest.update(status='failed',error=traceback.format_exc());raise
    finally:write_json(out/'manifest.json',manifest)

if __name__=='__main__':main()

