"""Remote E1-A acceptance and optional THREE-update devtrain-only composition probe."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_runtime import ROOT, code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from transfer.container_runtime import training_container_kind

BASE = '88d982aa33c7200a60d22b8730c569b7a74f8741'


def real_probe(args, out):
    import numpy as np
    import torch
    from loadData.observation_sampling import sample
    from loadData.spatial_tiles import observe_tile, tile_slice
    from model.checkpoint_io import load_model_checkpoint
    from model.observation_coverage import CoverageConfig, CoverageModulation, describe
    from transfer import development_training as dev
    from transfer.g3_training import _forward, _optimizer, _update, build_target, state_hash
    from transfer.m1_ablation import input_identity, read_scene

    inputs = input_identity(args.development_root, args.protocol_root, args.data_root)
    scene, record = read_scene(inputs)
    labels = dev.sparse_labels(record, scene.plan, 'train_support')
    tile = next(t for t in scene.plan['tiles'] if record['roles'][t['id']]['role']=='devtrain'
                and (labels[tile_slice(t)] > 0).any())
    base = json.loads((ROOT/'configs/experiments/screen_m6_none.json').read_text())
    model, cfg = build_target(base, scene, 'target_scratch_full', None, model_seed=64, head_seed=66, device='cuda')
    adapter = model.backbone.adapter
    coverage_cfg = CoverageConfig(wavelength_min=adapter.wavelength_min, wavelength_max=adapter.wavelength_max,
                                  sigma=adapter.locality_sigma, window=adapter.locality_window)
    layer = CoverageModulation(adapter.latent_channels, adapter.value_dim, mode='affine').cuda()
    off = CoverageModulation(adapter.latent_channels, adapter.value_dim).cuda()
    obs = observe_tile(scene.cube, scene.plan, scene.stats, tile)
    wave = np.asarray(scene.bands.wavelengths); lo, hi = float(wave.min()), float(wave.max())
    configs = [{'mode':'none'}, {'mode':'range_crop','lower':lo,'upper':(lo+hi)/2,'unit':'nm'},
               {'mode':'contiguous_gap','lower':lo+.375*(hi-lo),'upper':lo+.625*(hi-lo),'unit':'nm'}]
    views = [sample(obs,scene.bands,c)[0] for c in configs]
    current = {'view':views[0], 'module':off}; captured = {}

    def hook(module, arguments, keywords, latent):
        b = current['view'].bands
        assert latent.shape[-1] == adapter.out_channels and arguments[0].shape[-1] == len(b.band_ids)
        desc = describe(b, adapter._anchors(latent.device,latent.dtype), coverage_cfg)
        captured.update(description=desc, latent=latent.detach().clone(), bands=len(b.band_ids))
        return current['module'](latent,desc)

    model.eval(); layer.eval()
    with torch.no_grad():
        raw = _forward(model,scene,tile,observation_view=views[0])[0]
        repeat = _forward(model,scene,tile,observation_view=views[0])[0]
    handle = adapter.register_forward_hook(hook,with_kwargs=True)
    try:
        with torch.no_grad():
            disabled = _forward(model,scene,tile,observation_view=views[0])[0]
            current['module'] = layer
            identity = _forward(model,scene,tile,observation_view=views[0])[0]
        differences = {name:float((raw-value).abs().max()) for name,value in
                       [('baseline_repeat',repeat),('disabled',disabled),('zero_init',identity)]}
        assert all(d <= 3e-6 for d in differences.values()), differences
        optimizer = _optimizer(model,cfg)
        optimizer.add_param_group({'params':list(layer.parameters()),'lr':cfg.training.lr})
        enrolled = {id(p) for group in optimizer.param_groups for p in group['params']}
        assert all(id(p) in enrolled for p in layer.parameters())
        before = {n:p.detach().clone() for n,p in layer.named_parameters()}; steps=[]
        torch.manual_seed(64)
        for step,view in enumerate(views,1):
            current['view']=view;layer.train()
            losses = _update(model,scene,tile,labels[tile_slice(tile)],cfg,optimizer,observation_view=view)
            d=captured['description']
            gradients={n:float(p.grad.abs().max()) for n,p in layer.named_parameters()}
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
            steps.append({'step':step,'bands':captured['bands'],'loss':losses,'gradients':gradients,
                          'descriptor':d.features.tolist(),'local_count':d.local_count.tolist(),
                          'usable_anchors':int(d.usable.sum()),'metadata':d.diagnostics})
        updates={n:float((p-before[n]).abs().max()) for n,p in layer.named_parameters()}
        assert all(v>0 for v in updates.values()), updates
        assert steps[0]['gradients']['generator.0.weight']==0 and steps[1]['gradients']['generator.0.weight']>0
        layer.eval();d=captured['description'];x=captured['latent']
        with torch.no_grad():
            modulated=layer(x,d); delta=float((modulated-x).abs().max())
            assert delta>0 and torch.isfinite(modulated).all()
            protected=(~d.usable).repeat_interleave(adapter.value_dim)
            assert torch.equal(modulated[...,protected],x[...,protected])
            altered=replace(d,features=d.features.clone());altered.features[:,3]+=1
            dependence=float((layer(x,d)-layer(x,altered)).abs().max());assert dependence>0
            assert layer(x,None) is x
        checkpoint=out/'coverage-module.pt'
        torch.save({'config':layer.export_config(),'coverage_config':coverage_cfg.export_config(),
                    'state_dict':layer.state_dict()},checkpoint)
        payload=torch.load(checkpoint,map_location='cuda',weights_only=True)
        restored=CoverageModulation(**payload['config']).cuda().eval()
        restored.load_state_dict(payload['state_dict'],strict=True)
        with torch.no_grad():torch.testing.assert_close(restored(x,d),modulated,rtol=0,atol=0)
    finally:
        handle.remove()
    legacy=None
    if args.legacy_checkpoint is not None:
        saved=hash_file(args.legacy_checkpoint);loaded=load_model_checkpoint(args.legacy_checkpoint,'cpu')
        legacy={'path':str(args.legacy_checkpoint),'sha256':saved,'state_hash':state_hash(loaded)}
        assert hash_file(args.legacy_checkpoint)==saved
    assert all(hash_file(p)==h for p,h in inputs['files'].items())
    result={'status':'passed','scope':'engineering_only_three_updates_on_one_released_devtrain_tile',
            'inputs':inputs,'tile_id':tile['id'],'training_labels_in_tile':int((labels[tile_slice(tile)]>0).sum()),
            'optimizer_updates':3,'validation_labels_used':0,'test_evaluated':False,'accuracy_evaluated':False,
            'seeds':{'model':64,'head':66,'training_torch':64},'conditions':configs,'steps':steps,
            'module_config':layer.export_config(),'coverage_config':coverage_cfg.export_config(),
            'module_parameters':sum(p.numel() for p in layer.parameters()),'parameter_max_updates':updates,
            'latent_max_delta':delta,'descriptor_dependence_max_delta':dependence,
            'probability_equivalence_deltas':differences,'unsupported_channels_unchanged':True,
            'standalone_checkpoint_sha256':hash_file(checkpoint),'legacy_checkpoint_loaded_unchanged':legacy,
            'limitation':'temporary hook is removed; no integrated E1 classifier checkpoint/CLI or accuracy evidence'}
    write_json(out/'real-probe.json',result)
    return {k:v for k,v in result.items() if k not in ('steps','inputs')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit',required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--development-root',type=Path);p.add_argument('--protocol-root',type=Path)
    p.add_argument('--data-root',type=Path);p.add_argument('--legacy-checkpoint',type=Path)
    a=p.parse_args();out=external_directory(a.out_dir)
    if ROOT.is_relative_to(out):raise ValueError('output contains source checkout')
    sources=[a.development_root,a.protocol_root,a.data_root]
    if any(sources) and not all(sources):raise ValueError('supply all real input roots together')
    for path in sources:
        if path is not None:
            src=path.resolve(strict=True)
            if out.is_relative_to(src) or src.is_relative_to(out):raise ValueError('output overlaps frozen inputs')
    out.mkdir(parents=True,exist_ok=False)
    state={'status':'running','started_unix':time.time(),'checks':[],'command':sys.argv,
           'scope':'E1-A standalone engineering; no long training or efficacy claim'}
    try:
        assert sys.platform=='linux' and __debug__ and training_container_kind()
        code=code_snapshot();assert code['commit']==a.expected_commit and not code['dirty']
        assert subprocess.check_output(['git','rev-parse','--is-shallow-repository'],text=True).strip()=='false'
        subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],check=True)
        changes=subprocess.check_output(['git','diff','--name-only','-z',BASE,'HEAD'],text=True).strip('\0').split('\0')
        allowed={'model/observation_coverage.py','tests/test_observation_coverage.py','scripts/validate_e1a.py',
                 'docs/references/e1-reference-review.json','docs/architecture/module-map.json','README.md'}
        assert all(n in allowed or n.endswith('.md') and n.startswith(('research/','docs/guides/')) for n in changes)
        import torch
        assert torch.cuda.is_available();torch.set_num_threads(1)
        state.update(code=code,python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
                     container=training_container_kind(),gpu=torch.cuda.get_device_name(0),
                     visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
             'E1_DEVICE':'cpu','M1_DEVICE':'cpu','M4_DEVICE':'cpu','G3_DEVICE':'cpu',
             'GRAPH_CACHE_DEVICE':'cpu','DYNAMIC_HEAD_DEVICE':'cpu'}
        for name,pattern,extra in [('focused-cpu','test_observation_coverage.py',{}),('full-cpu','test_*.py',{}),
            ('focused-cuda','test_observation_coverage.py',{'E1_DEVICE':'cuda'}),
            ('m1-cuda','test_observation_training.py',{'M1_DEVICE':'cuda'})]:
            command=[sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v'];begin=time.monotonic()
            with (out/(name+'.log')).open('w') as log:
                ret=subprocess.run(command,cwd=ROOT,env={**env,**extra},stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
            state['checks'].append({'name':name,'command':command,'environment':extra,'returncode':ret,'seconds':time.monotonic()-begin})
            write_json(out/'manifest.json',state);assert ret==0,name+' failed; logs retained'
        from scripts.validate_m2 import cuda_cache_lifecycle
        state['vca_cuda_cache_lifecycle']=cuda_cache_lifecycle()
        if all(sources):state['real_probe']=real_probe(a,out)
        import ast
        names=subprocess.check_output(['git','ls-files','-z'],text=True).strip('\0').split('\0')
        for name in names:
            if name.endswith('.py'):ast.parse((ROOT/name).read_text(encoding='utf-8-sig'),filename=name)
            if name.endswith('.json'):json.loads((ROOT/name).read_text(encoding='utf-8-sig'))
        state['static_files']={ext:sum(n.endswith(ext) for n in names) for ext in ('.py','.json')}
        subprocess.run(['git','diff','--check',BASE,'HEAD'],check=True);require_frozen_code(code)
        state['status']='passed'
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:
        state['finished_unix']=time.time();write_json(out/'manifest.json',state)
    print(json.dumps({k:v for k,v in state.items() if k!='code'},indent=2))


if __name__=='__main__':main()
