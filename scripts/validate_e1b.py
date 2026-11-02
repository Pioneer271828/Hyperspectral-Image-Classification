"""Remote E1-B acceptance; optional paired THREE-update engineering preflight."""
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

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_runtime import ROOT,code_snapshot,external_directory,require_frozen_code,hash_file,write_json
from transfer.container_runtime import training_container_kind

BASE='70edf753518dc50b3ac4f212cd79d1e593433f54'


def real_probe(args,out):
    import numpy as np
    import torch
    from transfer import development_training as dev
    from transfer.g3_training import _forward
    from transfer.m1_ablation import input_identity,read_scene
    from transfer.observation_policy import ObservationPolicy
    from transfer.run_provenance import verify_run_artifacts
    from model.checkpoint_io import load_model_checkpoint

    inputs=input_identity(args.development_root,args.protocol_root,args.data_root)
    scene,record=read_scene(inputs)
    wave=np.asarray(scene.bands.wavelengths);lo,hi=float(wave.min()),float(wave.max())
    configs=[{'mode':'none'},{'mode':'range_crop','lower':lo,'upper':(lo+hi)/2,'unit':'nm'},
             {'mode':'contiguous_gap','lower':lo+.375*(hi-lo),'upper':lo+.625*(hi-lo),'unit':'nm'}]
    policy=ObservationPolicy.freeze(scene,record,updates=3,training_configs=configs,
        train_seed=64,validation_seed=512,conditions={'full':{'mode':'none'}})
    write_json(out/'preregistered-preflight.json',{'scope':'engineering_only_no_accuracy_claim',
        'arms':['none','affine'],'updates_per_arm':3,'model_seed':64,'head_seed':66,'sampler_seed':68,
        'selection':'minimum_validation_nll_then_earliest_step','validation_labels':45,
        'test_evaluated':False,'inputs':inputs,'policy':policy.to_dict()})
    base=json.loads((ROOT/'configs/experiments/screen_m6_none.json').read_text())
    original=dev.build_target;results={};retained={};probes=[]
    for mode in ('none','affine'):
        value=copy.deepcopy(base);value['model']['coverage_mode']=mode
        def build(*a,**kw):
            model,cfg=original(*a,**kw)
            retained[mode]={'model':model,'initial':{n:p.detach().cpu().clone() for n,p in model.state_dict().items()}}
            if mode=='affine':
                layer=model.backbone.coverage_modulation
                def hook(m,arguments,output):
                    if not model.training:return
                    d=arguments[1];probes.append({'metadata_hash':d.diagnostics.get('metadata_hash'),
                        'usable_anchors':int(d.usable.sum()),'out_of_reference_count':d.diagnostics['out_of_reference_count']})
                layer.register_forward_hook(hook)
            return model,cfg
        with patch.object(dev,'build_target',side_effect=build):
            results[mode]=dev.adapt(scene,record,value,out/mode,steps=3,val_every=1,device='cuda',seed=64,
                observation_policy=policy,progress=lambda x:print(json.dumps(x),flush=True))
        assert verify_run_artifacts(out/mode)['status']=='verified'
        assert results[mode]['test_evaluated'] is False
    off,on=retained['none'],retained['affine']
    assert all(torch.equal(p,on['initial'][n]) for n,p in off['initial'].items())
    assert results['none']['initial_head_hash']==results['affine']['initial_head_hash']
    assert [r['tile_id'] for r in results['none']['trace']]==[r['tile_id'] for r in results['affine']['trace']]
    assert [r['metadata_hash'] for r in probes]==[r['selected_metadata_hash'] for r in policy.to_dict()['training']]
    layer=on['model'].backbone.coverage_modulation
    updates={n:float((p.detach().cpu()-on['initial']['backbone.coverage_modulation.'+n]).abs().max()) for n,p in layer.named_parameters()}
    gradients={n:float(p.grad.abs().max()) for n,p in layer.named_parameters()}
    assert all(v>0 for v in updates.values()) and all(np.isfinite(v) for v in gradients.values())
    assert results['affine']['runtime_config']['coverage']['optimized_parameters']==194
    # Use the actually selected integrated checkpoint, without rewriting its config.
    selected=load_model_checkpoint(out/'affine/model.pt','cuda')
    val=dev.evaluate_validation(selected,scene,record,observation_policy=policy)
    assert abs(val['nll']-min(r['nll'] for r in results['affine']['validation']))<3e-6
    tile=next(t for t in scene.plan['tiles'] if record['roles'][t['id']]['role']=='devval')
    view=policy.view(scene,record,tile,phase='validation')
    selected_layer=selected.backbone.coverage_modulation
    with torch.no_grad():
        active=_forward(selected,scene,tile,observation_view=view)[0]
        strength=selected_layer.strength
        try:
            selected_layer.strength=0.
            neutral=_forward(selected,scene,tile,observation_view=view)[0]
        finally:selected_layer.strength=strength
        effect=float((active-neutral).abs().max())
    assert effect>0 and torch.isfinite(active).all()
    legacy=None
    if args.legacy_checkpoint:
        files=[args.legacy_checkpoint,Path(str(args.legacy_checkpoint)+'.json')]
        before={str(p):hash_file(p) for p in files}
        old=load_model_checkpoint(args.legacy_checkpoint,'cpu')
        assert old.backbone.coverage_mode=='none'
        assert before=={str(p):hash_file(p) for p in files}
        legacy={'loaded_without_rewrite':True,'files':before}
        if (args.legacy_checkpoint.parent/'result.json').exists():legacy['provenance']=verify_run_artifacts(args.legacy_checkpoint.parent)
    assert all(hash_file(p)==h for p,h in inputs['files'].items())
    result={'status':'passed','scope':'production_integration_engineering_only',
        'updates_total':6,'train_labels':90,'validation_labels':45,'test_evaluated':False,
        'policy_hash':policy.policy_hash,'training_band_counts':[len(r['selected_positions']) for r in policy.to_dict()['training']],
        'probe':probes,'parameter_max_updates':updates,'final_gradients':gradients,
        'registered_parameters':194,'optimizer_enrollment_verified':True,
        'selected_checkpoint_modulation_probability_delta':effect,'selected_checkpoint_validation':val,
        'legacy':legacy,'validation_traces':{k:v['validation'] for k,v in results.items()},
        'analysis':'Three updates validate wiring/learning/restoration only; no seed-averaged accuracy or sensor-transfer gain claim.'}
    write_json(out/'real-probe.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit',required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--development-root',type=Path);p.add_argument('--protocol-root',type=Path)
    p.add_argument('--data-root',type=Path);p.add_argument('--legacy-checkpoint',type=Path)
    a=p.parse_args();out=external_directory(a.out_dir)
    roots=[a.development_root,a.protocol_root,a.data_root]
    if any(roots) and not all(roots):raise ValueError('supply all real input roots together')
    for root in [ROOT,*[r for r in roots if r is not None]]:
        src=root.resolve(strict=True)
        if out.is_relative_to(src) or src.is_relative_to(out):raise ValueError('output overlaps source/inputs')
    out.mkdir(parents=True,exist_ok=False)
    state={'status':'running','command':sys.argv,'checks':[],'started_unix':time.time()}
    try:
        assert sys.platform=='linux' and __debug__ and training_container_kind()
        code=code_snapshot();assert code['commit']==a.expected_commit and not code['dirty']
        assert subprocess.check_output(['git','rev-parse','--is-shallow-repository'],text=True).strip()=='false'
        subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],check=True)
        import torch
        assert torch.cuda.is_available();torch.set_num_threads(1)
        state.update(code=code,python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
            container=training_container_kind(),gpu=torch.cuda.get_device_name(0),visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
             'E1_DEVICE':'cpu','M1_DEVICE':'cpu','M4_DEVICE':'cpu','G3_DEVICE':'cpu','GRAPH_CACHE_DEVICE':'cpu','DYNAMIC_HEAD_DEVICE':'cpu'}
        checks=[('focused-cpu','test_coverage_integration.py',{}),('full-cpu','test_*.py',{}),
            ('focused-cuda','test_coverage_integration.py',{'E1_DEVICE':'cuda'}),
            ('e1a-cuda','test_observation_coverage.py',{'E1_DEVICE':'cuda'}),
            ('m1-functional-cuda','test_observation_training.py',{'M1_DEVICE':'cuda'}),
            ('m4-cuda','test_region_residual_integration.py',{'M4_DEVICE':'cuda'})]
        for name,pattern,extra in checks:
            command=[sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v'];start=time.monotonic()
            if name=='m1-functional-cuda':
                # Numerical old/new/repeat/full-policy checks are run in FP64
                # with stricter bounds by the E1-B suite; original CPU9 stay exact.
                source="import sys,unittest;sys.path.insert(0,'tests');from test_observation_training import ObservationTrainingTests as T;skip={'test_default_matches_actual_parent_short_training','test_same_version_repeat_numerical_control','test_full_policy_equivalence_and_checkpoint_roundtrip'};suite=unittest.TestSuite(T(n) for n in unittest.defaultTestLoader.getTestCaseNames(T) if n not in skip);r=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(not r.wasSuccessful())"
                command=[sys.executable,'-c',source]
            with (out/(name+'.log')).open('w') as log:
                rc=subprocess.run(command,cwd=ROOT,env={**env,**extra},stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
            state['checks'].append({'name':name,'command':command,'environment':extra,'returncode':rc,'seconds':time.monotonic()-start})
            write_json(out/'manifest.json',state);assert rc==0,name+' failed; logs retained'
        from scripts.validate_m2 import cuda_cache_lifecycle
        state['vca_cuda_cache_lifecycle']=cuda_cache_lifecycle()
        if all(roots):state['real_probe']=real_probe(a,out)
        import ast
        names=subprocess.check_output(['git','ls-files','-z'],text=True).strip('\0').split('\0')
        for n in names:
            if n.endswith('.py'):ast.parse((ROOT/n).read_text(encoding='utf-8-sig'),filename=n)
            if n.endswith('.json'):json.loads((ROOT/n).read_text(encoding='utf-8-sig'))
        state['static_files']={ext:sum(n.endswith(ext) for n in names) for ext in ('.py','.json')}
        subprocess.run(['git','diff','--check',BASE,'HEAD'],check=True);require_frozen_code(code)
        state['status']='passed'
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:
        state['finished_unix']=time.time();write_json(out/'manifest.json',state)
    print(json.dumps({k:v for k,v in state.items() if k!='code'},indent=2))


if __name__=='__main__':main()
