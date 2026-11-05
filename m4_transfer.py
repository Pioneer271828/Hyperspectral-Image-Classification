"""Matched M4 source/target engineering adapters; legacy none is an exact delegate."""
import copy,math,re,time
from dataclasses import replace
from pathlib import Path
import torch
from joint_baseline import NAMES,load_config
from joint_training import train_step
from loadData.dataset_registry import DATASETS,file_hash
from model.checkpoint_io import load_model_checkpoint
from source_training import source_settings,SourceRuntime
from target_training import target_settings,TargetRuntime,observation,HEAD_ONLY
from target_initialization import initialize_target,domain_keys,state_id,METHODS
from target_bn_calibration import calibrate_target,POLICY,CONTEXT,GRAPH_POLICY
from scripts.train_joint_m4 import graph_binding

MODES=('none','capacity_control','x_residual')
SOURCE_PROTOCOL='g4_m4_source_engineering_v1'
TARGET_PROTOCOL='g4_m4_target_engineering_v1'


def source_options(held_out,mode,*,sources=None,seed=64):
    if mode not in MODES:raise ValueError('Unregistered M4 transfer mode')
    value=source_settings(held_out,sources,seed)
    if mode=='none':return value
    value=copy.deepcopy(value);value.update(protocol=SOURCE_PROTOCOL,m4_mode=mode)
    value['config']['name']+='_m4_'+mode;value['config']['model']['region_residual_mode']=mode
    return value


def target_options(target,method,mode,*,seed=64,support_seed=64,per_class=10,device='cpu'):
    if mode not in MODES:raise ValueError('Unregistered M4 transfer mode')
    value=target_settings(target,method,seed=seed,support_seed=support_seed,per_class=per_class,device=device)
    if mode=='none':return value
    value=copy.deepcopy(value);value.update(protocol=TARGET_PROTOCOL,m4_mode=mode)
    value['config']['name']+='_m4_'+mode;value['config']['model']['region_residual_mode']=mode
    return value


def _before(model):return {n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('backbone.region_residual.')}


def _evidence(row,model,before,*,frozen=False):
    params={n:p for n,p in model.named_parameters() if n in before}
    if not params:raise ValueError('Active M4 parameters missing')
    if (not frozen and any(p.grad is None for p in params.values())) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params.values()):
        raise ValueError('Missing/nonfinite M4 parameter gradient')
    changed=sum(int(torch.count_nonzero(p.detach()-before[n])) for n,p in params.items())
    if frozen and (changed or any(p.grad is not None for p in params.values())):raise ValueError('Frozen M4 changed')
    row['m4_parameter_gradient_norm']=math.sqrt(sum(float(p.grad.double().square().sum()) for p in params.values() if p.grad is not None))
    row['m4_changed_elements']=changed
    projection=model.backbone.region_residual.projection
    if projection is not None:
        row['m4_hint_weight_norm']=float(projection.weight.detach()[:,64:].double().norm())
        row['m4_hint_gradient_norm']=float(projection.weight.grad[:,64:].double().norm()) if projection.weight.grad is not None else 0.
    return row


class M4SourceRuntime(SourceRuntime):
    def validate(self,settings):
        mode=settings.get('m4_mode','none')
        if mode=='none':return super().validate(settings)
        expected=source_options(settings.get('held_out'),mode,sources=settings.get('source_tasks'),seed=settings.get('model_seed'))
        if {k:v for k,v in settings.items() if k not in ('executed_optimizer_steps','selected_step')}!=expected:
            raise ValueError('M4 source policy/configuration differs')
        return True

    def train_step(self,model,optimizer,name,item,config):
        if config.model.region_residual_mode=='none':return train_step(model,optimizer,name,item,config)
        before=_before(model)
        return _evidence(train_step(model,optimizer,name,item,config),model,before)

    def load_selected(self,path,items,raw_source_path,*,device='cpu'):
        model=super().load_selected(path,items,raw_source_path,device=device)
        settings=model.training_config['joint_runtime']
        if settings.get('m4_mode','none')!='none':
            provenance=model.training_config['provenance']
            if (provenance.get('m4_mode')!=settings['m4_mode']
                    or provenance.get('m4_graph_binding')!=graph_binding(items)):
                raise ValueError('M4 source graph/mode binding differs')
        return model


def _source_candidate(path,weight_sha256,sidecar_sha256,target,mode):
    if (mode=='none' or path is None or any(not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h)
            for h in (weight_sha256,sidecar_sha256))):raise ValueError('Pinned matched M4 source pair required')
    path=Path(path);sidecar=Path(str(path)+'.json')
    if file_hash(path)!=weight_sha256 or file_hash(sidecar)!=sidecar_sha256:raise ValueError('M4 source pair changed')
    raw=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(raw,dict) or any(not torch.is_tensor(t) or t.is_floating_point() and
            (t.dtype!=torch.float32 or not torch.isfinite(t).all()) for t in raw.values()):
        raise ValueError('Finite FP32 M4 source state required')
    with torch.random.fork_rng(devices=[]):source=load_model_checkpoint(path,device='cpu')
    meta=source.training_config;settings=meta.get('joint_runtime',{});M4SourceRuntime().validate(settings)
    tasks=tuple(settings['source_tasks']);provenance=meta.get('provenance',{});binding=provenance.get('m4_graph_binding',{})
    if (settings.get('protocol')!=SOURCE_PROTOCOL or settings.get('m4_mode')!=mode
            or source.backbone.region_residual_mode!=mode or settings['held_out']!=target or target in tasks
            or tasks!=tuple(n for n in NAMES if n in tasks) or tuple(source.heads)!=tasks
            or any(source.heads[n].out_features!=DATASETS[n][2] for n in tasks)
            or meta.get('research_config')!=settings['config'] or settings.get('executed_optimizer_steps')!=settings['optimizer_steps']
            or meta.get('checkpoint_optimizer_step')!=settings.get('selected_step')
            or settings.get('selected_step') not in range(len(tasks),settings['optimizer_steps']+1,len(tasks))
            or provenance.get('policy')!='selective_acc1p_source_v1' or provenance.get('held_out')!=target
            or tuple(provenance.get('sources',[]))!=tasks or provenance.get('target_payload_accessed') is not False
            or not re.fullmatch('[0-9a-f]{64}',provenance.get('source_identity',''))
            or provenance.get('m4_mode')!=mode or set(binding)!=set(tasks)
            or any(set(v)!={'Q','A'} or any(not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) for h in v.values()) for v in binding.values())
            or any(target in n.split('.') for n in source.state_dict())):
        raise ValueError('M4 source-only target/mode/provenance/budget differs')
    return source,{'weight_sha256':weight_sha256,'sidecar_sha256':sidecar_sha256,'source_identity':provenance['source_identity'],
        'source_tasks':list(tasks),'training_profile':settings['profile'],'source_model_seed':settings['model_seed'],
        'declared_optimizer_steps':settings['optimizer_steps'],'checkpoint_optimizer_step':meta['checkpoint_optimizer_step'],
        'source_protocol':SOURCE_PROTOCOL,'m4_mode':mode,'source_graph_binding':binding}


def initialize_m4_target(target,method,mode,*,model_seed,head_seed,source_path=None,
                         source_weight_sha256=None,source_sidecar_sha256=None,device='cpu'):
    kwargs=dict(model_seed=model_seed,head_seed=head_seed,source_path=source_path,
        source_weight_sha256=source_weight_sha256,source_sidecar_sha256=source_sidecar_sha256,device=device)
    if mode=='none':return initialize_target(target,method,**kwargs)
    if (mode not in MODES or target not in NAMES or method not in METHODS
            or any(type(s)is not int or not 0<=s<2**32 for s in (model_seed,head_seed))):
        raise ValueError('Registered M4 role/mode and explicit uint32 seeds required')
    if method==METHODS[0] and any(x is not None for x in (source_path,source_weight_sha256,source_sidecar_sha256)):
        raise ValueError('M4 scratch must not receive source inputs')
    base=load_config('configs/experiments/base97_joint.json')
    cfg=replace(base,name='g4_target_init_'+target,device=str(torch.device(device)),data=replace(base.data,datasets=[target]),
        training=replace(base.training,seed=model_seed),model=replace(base.model,region_residual_mode=mode))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(model_seed);model=cfg.build_model({target:DATASETS[target][2]})
        torch.random.default_generator.manual_seed(head_seed)
        for family in (model.heads,model.hgnn_heads,model.cnn_heads):family[target].reset_parameters()
    fresh=domain_keys(model.backbone);copied=[];receipt=None
    if method!=METHODS[0]:
        source,receipt=_source_candidate(source_path,source_weight_sha256,source_sidecar_sha256,target,mode)
        a=dict(source.export_config()['backbone_kwargs']);b=dict(model.export_config()['backbone_kwargs'])
        a.pop('dataset_names',None);b.pop('dataset_names',None)
        if a!=b:raise ValueError('M4 source/target architecture differs')
        before=state_id(source);src=source.backbone.state_dict();dst=model.backbone.state_dict();shared=set(dst)-fresh
        if shared!=set(src)-domain_keys(source.backbone):raise ValueError('M4 shared state keys differ')
        for name in sorted(shared):
            if src[name].shape!=dst[name].shape or src[name].dtype!=dst[name].dtype:raise ValueError('M4 shared tensor differs')
            dst[name]=src[name].detach().clone();copied.append(name)
        model.backbone.load_state_dict(dst,strict=True)
        if state_id(source)!=before:raise ValueError('M4 source mutated')
        if file_hash(source_path)!=source_weight_sha256 or file_hash(str(source_path)+'.json')!=source_sidecar_sha256:
            raise ValueError('M4 source pair changed during copy')
    frozen=method==HEAD_ONLY;model.backbone.requires_grad_(not frozen);model=model.to(device);model.train()
    if frozen:model.backbone.eval()
    record={'policy':'g4_target_shared_weights_fresh_domains_v1','target':target,'method':method,'m4_mode':mode,
        'model_seed':model_seed,'head_seed':head_seed,'source':receipt,'copied_backbone_keys':copied,
        'fresh_target_domain_keys':sorted(fresh),'source_heads_copied':False,'source_domain_state_copied':False,
        'head_only_requires_target_bn_calibration':frozen,'source_graph_context':'source_train_label_lda_slic' if receipt else None,
        'target_graph_required':'label_free_target_scene','target_initial_config':cfg.to_dict(),
        'initial_state_sha256':state_id(model),'initial_device':str(next(model.parameters()).device),
        'target_training_budget_registered':False,'optimizer_updates':0,'model_test_scoring':False}
    model.training_config={'target_initialization':copy.deepcopy(record)}
    return model,record


class M4TargetRuntime(TargetRuntime):
    def validate(self,settings):
        mode=settings['config']['model']['region_residual_mode']
        expected=target_options(settings.get('target'),settings.get('method'),mode,seed=settings.get('model_seed'),
            support_seed=settings.get('support_seed'),per_class=settings.get('support_per_class'),device=settings['config']['device'])
        if {k:v for k,v in settings.items() if k not in ('executed_optimizer_steps','selected_step')}!=expected:
            raise ValueError('M4 target policy/configuration differs')
        return True

    def build_model(self,config,counts):
        mode=config.model.region_residual_mode
        if mode=='none':return super().build_model(config,counts)
        target=self.settings['target']
        if counts!={target:DATASETS[target][2]}:raise ValueError('M4 target class count differs')
        model,self.initialization=initialize_m4_target(target,self.settings['method'],mode,
            model_seed=self.settings['model_seed'],head_seed=self.settings['head_seed'],device=config.device,**self.source_spec)
        model.eval()
        if self.settings['method']==HEAD_ONLY:
            if next(model.parameters()).is_cuda:torch.cuda.synchronize()
            start=time.perf_counter()
            model,self.initial_calibration=calibrate_target(model,observation(model,self.items[target]),mode=POLICY,
                context=CONTEXT,graph_policy=GRAPH_POLICY,source_state_sha256=state_id(model))
            if next(model.parameters()).is_cuda:torch.cuda.synchronize()
            self.initial_calibration_seconds=time.perf_counter()-start
            self.initialization=copy.deepcopy(model.training_config['target_initialization'])
        return model

    def train_step_at(self,cursor,model,optimizer,name,item,config):
        if config.model.region_residual_mode=='none':return super().train_step_at(cursor,model,optimizer,name,item,config)
        before=_before(model)
        row=super().train_step_at(cursor,model,optimizer,name,item,config)
        return _evidence(row,model,before,frozen=self.settings['method']==HEAD_ONLY)
