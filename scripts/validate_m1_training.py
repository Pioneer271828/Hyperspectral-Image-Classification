"""Remote-only M1-B acceptance and bounded real-development integration preflight."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def real_check(args, out):
    import numpy as np
    import torch
    from scipy.io import loadmat
    from loadData.data_read import readWavelengths
    from loadData.observation_sampling import BandMetadata
    from model.checkpoint_io import load_model_checkpoint
    from research_runtime import hash_file, write_json
    from transfer import development_training as dev
    from transfer.m4_ablation import input_identity
    from transfer.observation_policy import ObservationPolicy, default_conditions, evaluate_conditions
    from transfer.run_provenance import verify_run_artifacts
    inputs = input_identity(args.development_root, args.protocol_root, args.data_root)
    record = json.loads((args.development_root/'development.json').read_text())
    plan = json.loads((args.development_root/'parent-target-plan.json').read_text())
    cube = loadmat(inputs['image'], variable_names=['xuzhou'])['xuzhou']
    wave, _ = readWavelengths('xuzhou', cube.shape[-1])
    bands = BandMetadata(inputs['files'][inputs['image']], tuple(range(len(wave))), tuple(wave), 'nm', 'approximate', (True,)*len(wave))
    scene = dev.prepare_scene(cube, bands, plan, record)
    base = json.loads((ROOT/'configs/experiments/screen_m6_none.json').read_text())
    assert base['model'].get('region_residual_mode','none') == 'none'
    conditions = default_conditions(bands)
    configs = [{'mode':'random_subset','keep_fraction':p} for p in (.75,.5,.25)] + [conditions['gap']]
    policies = {name: ObservationPolicy.freeze(scene,record,updates=4,training_configs=cfg,
                train_seed=64,validation_seed=512) for name,cfg in [('full',[{'mode':'none'}]),('subset',configs)]}
    assert policies['full'].to_dict()['validation'] == policies['subset'].to_dict()['validation']
    # Persist both plans before either arm is trained, and record the image/protocol identities.
    write_json(out/'inputs.json',inputs)
    for name,p in policies.items(): write_json(out/(name+'-frozen-policy.json'),p.to_dict())
    results = {}; builder = dev.build_target
    for name, policy in policies.items():
        observed = []; grads = []; holders = {}
        def build(*a,**kw):
            model, cfg = builder(*a,**kw); holders['model']=model
            def hook(module, values, kwargs):
                observed.append({'training':model.training,'bands':values[0].shape[-1],
                                 'centers':len(kwargs['wavelengths'])})
            model.backbone.adapter.register_forward_pre_hook(hook,with_kwargs=True)
            parameter = next(model.backbone.adapter.parameters()); holders['parameter']=parameter
            holders['initial']=parameter.detach().cpu().clone()
            parameter.register_hook(lambda g:grads.append({'finite':bool(torch.isfinite(g).all()),'max_abs':float(g.abs().max())}))
            return model,cfg
        folder=out/name
        with patch.object(dev,'build_target',side_effect=build):
            result=dev.adapt(scene,record,base,folder,device='cuda',seed=64,steps=4,val_every=1,
                             sampling='class_first',observation_policy=policy)
        verified=verify_run_artifacts(folder)
        reloaded=ObservationPolicy(json.loads((folder/'observation-policy.json').read_text()))
        selected=load_model_checkpoint(folder/'model.pt','cuda')
        review=evaluate_conditions(selected,scene,record,reloaded)
        fused=review['conditions']['full']['fused']; chosen=min(result['validation'],key=lambda r:r['nll'])
        assert abs(fused['nll']-chosen['nll'])<1e-5
        assert all(x['bands']==x['centers'] for x in observed)
        widths=[x['bands'] for x in observed if x['training']]
        assert widths==[len(r['selected_positions']) for r in policy.to_dict()['training']]
        change=float((holders['parameter'].detach().cpu()-holders['initial']).abs().max())
        assert change>0 and len(grads)==4 and all(g['finite'] for g in grads) and any(g['max_abs']>0 for g in grads)
        review.update(observed_widths=observed,gradient_probes=grads,adapter_parameter_max_update=change,
            first_loss=result['trace'][0]['loss']['total'],last_loss=result['trace'][-1]['loss']['total'],
            validation_curve=result['validation'],selected_step=result['selected_step'],
            decision='engineering participation verified; four updates cannot establish accuracy benefit')
        write_json(folder/'review.json',review)
        results[name]={'verified':verified,'train_widths':widths,'adapter_max_update':change,
                       'selected_step':result['selected_step'],'policy_hash':policy.policy_hash,
                       'full_development_oa':fused['oa'],'review_path':str(folder/'review.json')}
    raw=[json.loads((out/n/'result.json').read_text()) for n in ('full','subset')]
    for field in ('initial_backbone_hash','initial_head_hash'):
        assert raw[0][field]==raw[1][field]
    assert [(r['tile_id'],r['anchor_class']) for r in raw[0]['trace']]==[(r['tile_id'],r['anchor_class']) for r in raw[1]['trace']]
    assert all(hash_file(p)==h for p,h in inputs['files'].items())
    return {'status':'passed','scope':'four_updates_per_arm_engineering_only','arms':results,
            'conditions':list(policies['full'].conditions),'source_bands':len(bands.band_ids),
            'development_hash':record['validation_hash'],'statistics_hash':scene.stats['statistics_hash'],
            'test_evaluated':False,'raw_target_gt_loaded':False,'accuracy_decision':'pending independent full-budget experiment'}


def main():
    from research_runtime import code_snapshot, require_frozen_code, external_directory, write_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit',required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--development-root',type=Path);p.add_argument('--protocol-root',type=Path);p.add_argument('--data-root',type=Path)
    args=p.parse_args();code=code_snapshot()
    container=subprocess.run(['systemd-detect-virt','--container'],capture_output=True,text=True).stdout.strip()
    if sys.platform!='linux' or not (Path('/.dockerenv').exists() or container in ('lxc','docker','podman')) or sys.flags.optimize or code['dirty'] or code['commit']!=args.expected_commit:
        p.error('Require container, assertions and clean exact published SHA')
    if any(x is not None for x in (args.development_root,args.protocol_root,args.data_root)) and not all(x is not None for x in (args.development_root,args.protocol_root,args.data_root)):
        p.error('Supply all three real-development input roots or none')
    out=external_directory(args.out_dir)
    for root in (args.development_root,args.protocol_root,args.data_root):
        if root is not None and (out==root.resolve() or root.resolve() in out.parents or out in root.resolve().parents):p.error('Outputs overlap inputs')
    out.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','code':code,'checks':[],'started_unix':time.time(),'command':[sys.executable,*sys.argv]}
    write_json(out/'manifest.json',manifest)
    try:
        import torch
        assert torch.cuda.is_available(), 'GPU acceptance required, no silent fallback'
        torch.set_num_threads(1)
        manifest['environment']={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
            'visible_gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'device':torch.cuda.get_device_name()}
        for name,pattern,extra in [('focused-cpu','test_observation_training.py',{'M1_DEVICE':'cpu'}),
                ('full-cpu','test_*.py',{'M1_DEVICE':'cpu','M4_DEVICE':'cpu','G3_DEVICE':'cpu','GRAPH_CACHE_DEVICE':'cpu','DYNAMIC_HEAD_DEVICE':'cpu'}),
                ('focused-cuda','test_observation_training.py',{'M1_DEVICE':'cuda'}),
                ('existing-m1-cuda','test_observation_sampling.py',{'M1_DEVICE':'cuda'})]:
            command=[sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v'];begin=time.monotonic()
            with (out/(name+'.log')).open('w') as log:
                ret=subprocess.run(command,cwd=ROOT,env={**os.environ,**extra},stdout=log,stderr=subprocess.STDOUT).returncode
            manifest['checks'].append({'name':name,'command':command,'environment':extra,'returncode':ret,'seconds':time.monotonic()-begin})
            write_json(out/'manifest.json',manifest);assert ret==0,name+' failed; see log'
        if args.data_root is not None: manifest['real']=real_check(args,out)
        subprocess.run(['git','diff','--check','39459615f2fb3c84d7a362fa5cf7421ca60dd766','HEAD'],cwd=ROOT,check=True)
        require_frozen_code(code);manifest['status']='passed'
    except BaseException:
        manifest.update(status='failed',error=traceback.format_exc());raise
    finally:
        manifest['finished_unix']=time.time();write_json(out/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='code'},indent=2))


if __name__=='__main__':main()
