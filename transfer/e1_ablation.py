"""E1-C matched observation controls; immutable plans and independently reviewed cases."""
import copy
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch

from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from research_runtime import ROOT,external_directory,hash_file,require_frozen_code,tensor_hash,write_json
from train_spatial import SpatialConfig
from transfer import development_training as development
from transfer.m1_ablation import read_scene,SpectralProbe
from transfer.m4_ablation import input_identity,require
from transfer.observation_policy import ObservationPolicy,default_conditions,evaluate_conditions
from transfer.run_provenance import verify_run_artifacts

SPEC_PATH=ROOT/'configs/experiments/e1_c_paired_screen.json'
MODES=('none','anchor_only','binary','continuous','shuffled')
SEEDS=(64,512,2048)


def spec():
    value=json.loads(SPEC_PATH.read_text())
    require(value['modes']==list(MODES) and value['seeds']==list(SEEDS)
        and value['updates_per_run']==5000 and value['validation_every']==100
        and value['full_target_updates']==75000 and value['test_evaluation'] is False
        and value['source_pretraining_updates']==0 and value['method']=='target_scratch_full'
        and value['sampling']=='class_first','unregistered E1 policy')
    return value


def conditions(bands):
    wave=np.asarray(bands.wavelengths)[np.asarray(bands.valid)];lo,hi=float(wave.min()),float(wave.max());span=hi-lo
    require(span>0 and bands.wavelength_unit=='nm','physical full-observation range required')
    ranges={'lower_75':(0,.75),'upper_75':(.25,1),'lower_half':(0,.5),'upper_half':(.5,1),'middle_half':(.25,.75)}
    return {**default_conditions(bands),**{n:{'mode':'range_crop','lower':lo+a*span,'upper':lo+b*span,'unit':'nm'} for n,(a,b) in ranges.items()}}


def freeze_plan(code,inputs,output,*,smoke=False):
    s=spec();base=json.loads((ROOT/s['base_config']).read_text());c=SpatialConfig(base).research
    require(c.model.region_residual_mode=='none' and c.model.coverage_mode=='none' and c.model.graph_mode=='learned'
        and c.model.norm_type=='group_norm' and c.interaction.interaction_mode==c.interaction.graph_update_mode=='none','E1 baseline changed')
    require((c.training.lr,c.training.head_lr,c.training.weight_decay)==(s['backbone_lr'],s['head_lr'],s['weight_decay']),'optimizer changed')
    require(smoke or inputs['development_hash']==s['development_hash'],'wrong development protocol')
    return {'schema_version':1,'scope':'five_step_preflight' if smoke else 'registered_development_screen',
        'code':code,'spec':s,'base':base,'inputs':inputs,'output':str(external_directory(output)),
        'modes':list(MODES),'seeds':[64] if smoke else list(SEEDS),'updates':5 if smoke else 5000,'val_every':1 if smoke else 100}


def prepare_policies(plan,scene,record):
    require_frozen_code(plan['code']);require(record['validation_hash']==plan['inputs']['development_hash'],'wrong policy scene')
    root=Path(plan['output']);(root/'policies').mkdir(parents=True,exist_ok=False)
    bank=conditions(scene.bands);s=plan['spec'];evaluation={n:bank[n] for n in s['evaluation_conditions']}
    train=[bank[n] for n in s['training_cycle']];policies={};evaluation_hash=None
    for seed in plan['seeds']:
        policy=ObservationPolicy.freeze(scene,record,updates=plan['updates'],training_configs=train,
            train_seed=seed+s['observation_seed_offset'],validation_seed=s['validation_observation_seed'],conditions=evaluation)
        value=policy.to_dict();current=digest(value['validation'])
        require(evaluation_hash is None or evaluation_hash==current,'evaluation bank differs')
        # Range masks must really differ from every non-full training range.
        for name in s['primary_perturbations']:
            require(all(value['validation'][name]['selected_positions']!=r['selected_positions'] for r in value['training']),
                    'training-unseen range mask collides with training; insufficient spectral resolution')
        evaluation_hash=current;path=root/'policies'/f'{seed}.json';write_json(path,value)
        policies[str(seed)]={'path':str(path),'sha256':hash_file(path),'policy_hash':policy.policy_hash}
    result={**plan,'policies':policies,'evaluation_bank_hash':evaluation_hash}
    result['plan_hash']=digest(result);return result


def verify_plan(plan):
    require(plan['plan_hash']==digest({k:v for k,v in plan.items() if k!='plan_hash'}),'plan hash mismatch')
    s=spec();smoke=plan['scope']=='five_step_preflight'
    require(plan['spec']==s and plan['modes']==list(MODES) and plan['scope'] in ('five_step_preflight','registered_development_screen')
        and plan['seeds']==([64] if smoke else list(SEEDS)) and plan['updates']==(5 if smoke else 5000)
        and plan['val_every']==(1 if smoke else 100),'registered E1 specification changed')
    require(plan['base']==json.loads((ROOT/s['base_config']).read_text()),'base changed');require_frozen_code(plan['code'])
    for p,sha in plan['inputs']['files'].items():require(hash_file(p)==sha,'frozen input changed')
    require(set(plan['policies'])=={str(seed) for seed in plan['seeds']},'policy seeds differ')
    for seed,item in plan['policies'].items():
        path=Path(item['path']);require(path==Path(plan['output'])/'policies'/f'{seed}.json' and hash_file(path)==item['sha256'],'policy path/hash changed')
        policy=ObservationPolicy(json.loads(path.read_text()));v=policy.to_dict()
        require(policy.policy_hash==item['policy_hash'] and digest(v['validation'])==plan['evaluation_bank_hash']
            and v['updates']==plan['updates'] and v['identity']['development_hash']==plan['inputs']['development_hash'],'policy identity differs')


def case_base(plan,mode):
    require(mode in MODES,'unknown E1 arm');value=copy.deepcopy(plan['base'])
    if mode!='none':value['model'].update(coverage_mode='affine',coverage_condition=mode,**plan['spec']['coverage'])
    return value


def case_path(plan,seed,mode):
    require(seed in plan['seeds'] and mode in plan['modes'],'unregistered case')
    return Path(plan['output'])/'cases'/f'seed-{seed}'/mode


def policy_for(plan,seed):return ObservationPolicy(json.loads(Path(plan['policies'][str(seed)]['path']).read_text()))


class CoverageProbe(SpectralProbe):
    def attach(self,model):
        super().attach(model);layer=model.backbone.coverage_modulation;self.layer=layer
        self.common_initial=digest({n:tensor_hash(v) for n,v in model.backbone.state_dict().items() if not n.startswith('coverage_modulation.')})
        self.coverage={'parameters':0,'train_calls':0,'probes':[],'gradients':[]}
        if layer is None:return
        self.coverage_initial={n:p.detach().cpu().clone() for n,p in layer.named_parameters()}
        self.coverage['parameters']=sum(p.numel() for p in layer.parameters())
        def forward(module,args,out):
            if not model.training:return
            self.coverage['train_calls']+=1;step=self.coverage['train_calls']
            if step in self.points:self.coverage['probes'].append({'step':step,'metadata_hash':args[1].diagnostics['metadata_hash'],
                'usable':int(args[1].usable.sum()),'physical_descriptor_hash':tensor_hash(args[1].features)})
        self.handles.append(layer.register_forward_hook(forward))
        for name,p in layer.named_parameters():
            def grad(g,name=name):
                if self.coverage['train_calls'] in self.points:self.coverage['gradients'].append({'step':self.coverage['train_calls'],
                    'parameter':name,'finite':bool(torch.isfinite(g).all()),'max_abs':float(g.abs().max())})
            self.handles.append(p.register_hook(grad))

    def finish(self,policy,updates):
        result=super().finish(policy,updates)
        if self.layer is not None:
            self.coverage['parameter_max_updates']={n:float((p.detach().cpu()-self.coverage_initial[n]).abs().max()) for n,p in self.layer.named_parameters()}
            require(self.coverage['parameters']==194 and self.coverage['train_calls']==updates
                and all(x['finite'] for x in self.coverage['gradients']) and self.coverage['gradients']
                and all(v>0 for v in self.coverage['parameter_max_updates'].values()),'E1 parameters did not learn')
            for r in self.coverage['probes']:
                require(r['metadata_hash']==policy.to_dict()['training'][r['step']-1]['selected_metadata_hash'],'E1 observation identity differs')
        result['coverage']=self.coverage;return result


def verified_case(plan,seed,mode):
    verify_plan(plan);path=case_path(plan,seed,mode);r=json.loads((path/'case.json').read_text())
    require(r['status']=='completed' and r['plan_hash']==plan['plan_hash'] and r['seed']==seed and r['mode']==mode
        and r.get('receipt_hash')==digest({k:v for k,v in r.items() if k!='receipt_hash'}),'incomplete/corrupt case receipt')
    for p,sha in r['artifacts'].items():
        f=(path/p).resolve();require(f.is_relative_to(path.resolve()) and hash_file(f)==sha,'case artifact mismatch')
    actual=verify_run_artifacts(path/'training');result=json.loads((path/'training/result.json').read_text())
    cfg=result['resolved_config'];runtime=result['runtime_config']
    require(actual['optimizer_steps']==plan['updates'] and cfg['model']==SpatialConfig(case_base(plan,mode)).to_dict()['model']
        and runtime['seeds']=={'model':seed,'head':seed+2,'sampler':seed+4,'training_torch':seed}
        and runtime['validation_every_updates']==plan['val_every'] and result['method']==plan['spec']['method']
        and result['selection']==plan['spec']['selection'] and result['sampler']['mode']==plan['spec']['sampling']
        and result['development_hash']==plan['inputs']['development_hash'] and result['test_evaluated'] is False,'case policy/config differs')
    require(all(cfg['training'][k]==plan['spec'][s] for k,s in [('lr','backbone_lr'),('head_lr','head_lr'),('weight_decay','weight_decay')]),'optimizer differs')
    require(json.loads((path/'training/observation-policy.json').read_text())==policy_for(plan,seed).to_dict(),'case observations differ')
    require(r['initial_head_hash']==result['initial_head_hash'] and r['sample_trace_hash']==digest([(x['tile_id'],x['anchor_class']) for x in result['trace']]),'pair evidence differs')
    return r


def run_case(plan,scene,record,seed,mode,*,device='cuda',resume=False):
    verify_plan(plan);path=case_path(plan,seed,mode);policy=policy_for(plan,seed);policy.validate(scene,record,plan['updates'])
    if path.exists():
        require(resume,'existing case requires explicit verified resume');return {**verified_case(plan,seed,mode),'operation':'verified_existing'}
    path.mkdir(parents=True,exist_ok=False);receipt={'status':'running','plan_hash':plan['plan_hash'],'seed':seed,'mode':mode,
        'artifact_dir':str(path/'training'),'started_unix':time.time(),'pid':os.getpid(),'device':str(device),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
    write_json(path/'case.json',receipt);observer=CoverageProbe(plan['updates']);original=development.build_target
    def build(*a,**kw):
        model,cfg=original(*a,**kw);observer.attach(model);return model,cfg
    try:
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
        with patch.object(development,'build_target',side_effect=build):
            result=development.adapt(scene,record,case_base(plan,mode),path/'training',steps=plan['updates'],val_every=plan['val_every'],
                seed=seed,device=device,sampling='class_first',observation_policy=policy,progress=lambda event:write_json(path/'progress.json',event))
        participation=observer.finish(policy,plan['updates'])
        participation['peak_training_allocated_bytes']=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None
        selected=load_model_checkpoint(path/'training/model.pt',device);review=evaluate_conditions(selected,scene,record,policy)
        require(abs(review['conditions']['full']['fused']['nll']-min(r['nll'] for r in result['validation']))<1e-5,'selected NLL differs')
        loss=[r['loss']['total'] for r in result['trace']];window=min(100,len(loss))
        review.update(selected_step=result['selected_step'],validation_curve=result['validation'],participation=participation,
            first_window_train_loss=float(np.mean(loss[:window])),last_window_train_loss=float(np.mean(loss[-window:])),
            decision='pending frozen paired batch; no interim tuning')
        write_json(path/'review.json',review)
        receipt.update(status='completed',optimizer_steps=plan['updates'],common_initial_backbone_hash=observer.common_initial,
            initial_head_hash=result['initial_head_hash'],sample_trace_hash=digest([(x['tile_id'],x['anchor_class']) for x in result['trace']]),
            statistics_hash=scene.stats['statistics_hash'],evaluation_bank_hash=plan['evaluation_bank_hash'],
            perturbation_sequence_hash=digest(policy.to_dict()['training']),finished_unix=time.time(),
            artifacts={str(p.relative_to(path)):hash_file(p) for p in path.rglob('*') if p.is_file() and p.name!='case.json'})
        receipt['receipt_hash']=digest(receipt);write_json(path/'case.json',receipt);verified_case(plan,seed,mode)
        return {**receipt,'operation':'trained'}
    except BaseException as error:
        for h in observer.handles:h.remove()
        receipt.update(status='failed',error=repr(error),finished_unix=time.time());write_json(path/'case.json',receipt);raise


def decision(summary,pairs,scope):
    if scope=='five_step_preflight':return 'engineering_preflight_only'
    continuous=summary['continuous'];none=summary['none']
    good=(continuous['full_oa']['mean']>=none['full_oa']['mean'] and continuous['full_aa']['mean']>=none['full_aa']['mean']
        and sum(p['full_oa_vs_none']>=0 for p in pairs)>=2
        and all(continuous['range_oa']['mean']>summary[c]['range_oa']['mean'] and sum(p['range_oa_vs_'+c]>0 for p in pairs)>=2
            for c in MODES if c!='continuous'))
    return 'development_coverage_candidate_pending_confirmation' if good else 'no_joint_coverage_gain_keep_e1_off'


def summarize(plan):
    verify_plan(plan);rows=[];pairs=[]
    for seed in plan['seeds']:
        group=[verified_case(plan,seed,m) for m in MODES]
        for key in ('common_initial_backbone_hash','initial_head_hash','sample_trace_hash','statistics_hash','evaluation_bank_hash','perturbation_sequence_hash'):
            require(len({r[key] for r in group})==1,'paired mismatch: '+key)
        for mode,r in zip(MODES,group):
            review=json.loads((case_path(plan,seed,mode)/'review.json').read_text());scores=review['conditions'];full=scores['full']['fused']
            rows.append({'seed':seed,'mode':mode,'full_oa':full['oa'],'full_aa':full['aa'],'full_kappa':full['kappa'],
                'range_oa':float(np.mean([scores[n]['fused']['oa'] for n in plan['spec']['primary_perturbations']])),
                'perturbed_oa':float(np.mean([v['fused']['oa'] for n,v in scores.items() if n!='full'])),
                'condition_oa':{n:v['fused']['oa'] for n,v in scores.items()},'selected_step':review['selected_step'],
                'seconds':r['finished_unix']-r['started_unix']})
        by={r['mode']:r for r in rows if r['seed']==seed}
        pairs.append({'seed':seed,'full_oa_vs_none':by['continuous']['full_oa']-by['none']['full_oa'],
            **{'range_oa_vs_'+c:by['continuous']['range_oa']-by[c]['range_oa'] for c in MODES if c!='continuous'}})
    summary={m:{k:{'mean':float(np.mean([r[k] for r in rows if r['mode']==m])),
        'sample_sd':float(np.std([r[k] for r in rows if r['mode']==m],ddof=1)) if len(plan['seeds'])>1 else None}
        for k in ('full_oa','full_aa','full_kappa','range_oa','perturbed_oa')} for m in MODES}
    return {'status':'completed','scope':plan['scope'],'plan_hash':plan['plan_hash'],'rows':rows,'summary':summary,
        'paired_differences':pairs,'decision':decision(summary,pairs,plan['scope']),'target_updates':len(rows)*plan['updates'],
        'source_pretraining_updates':0,'test_evaluated':False,'boundary':None}
