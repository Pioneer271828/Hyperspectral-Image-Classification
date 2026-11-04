"""Pinned source-only shared weights; fresh target heads and domain BN state."""
import copy,re
from dataclasses import replace
from pathlib import Path
import torch
from joint_baseline import NAMES,load_config
from loadData.dataset_registry import DATASETS,file_hash
from model.checkpoint_io import load_model_checkpoint
from model.VCASAHGCN import DomainSpecificBatchNorm1d,DomainSpecificBatchNorm2d
from model.research_config import digest
from research_runtime import tensor_hash
from source_training import SourceRuntime
from source_pretrain import SourcePretrainRuntime,EXPECTED

METHODS=('target_scratch_full','source_pretrained_full_finetune','source_pretrained_head_only')


def domain_keys(backbone):
    prefixes=[name+'.norms.' for name,module in backbone.named_modules()
        if isinstance(module,(DomainSpecificBatchNorm1d,DomainSpecificBatchNorm2d))]
    return {k for k in backbone.state_dict() if any(k.startswith(p) for p in prefixes)}


def state_id(model):return digest({k:tensor_hash(v) for k,v in model.state_dict().items()})


def _source(path,weight_sha256,sidecar_sha256,target):
    path=Path(path);sidecar=Path(str(path)+'.json')
    if any(not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) for h in (weight_sha256,sidecar_sha256)):
        raise ValueError('Source weight and sidecar SHA256 pins required')
    if file_hash(path)!=weight_sha256 or file_hash(sidecar)!=sidecar_sha256:raise ValueError('Source checkpoint pair changed')
    raw=torch.load(path,map_location='cpu',weights_only=True)
    if any(t.is_floating_point() and (t.dtype!=torch.float32 or not torch.isfinite(t).all()) for t in raw.values()):
        raise ValueError('Finite FP32 source baseline checkpoint required')
    with torch.random.fork_rng(devices=[]):model=load_model_checkpoint(path,device='cpu')
    meta=model.training_config;settings=meta.get('joint_runtime',{});provenance=meta.get('provenance',{})
    if settings.get('protocol')==EXPECTED['protocol']:SourcePretrainRuntime().validate(settings)
    elif settings.get('protocol')=='source_only_engineering_v1':SourceRuntime().validate(settings)
    else:raise ValueError('Accepted source-only training protocol required; joint checkpoint forbidden')
    tasks=tuple(settings['source_tasks'])
    if (target!=settings['held_out'] or target in tasks or tasks!=tuple(n for n in NAMES if n in tasks)
            or tuple(model.heads)!=tasks or tuple(settings['config']['data']['datasets'])!=tasks
            or meta.get('research_config')!=settings['config'] or provenance.get('policy')!='selective_acc1p_source_v1'
            or provenance.get('held_out')!=target or tuple(provenance.get('sources',[]))!=tasks
            or provenance.get('target_payload_accessed') is not False
            or not re.fullmatch('[0-9a-f]{64}',provenance.get('source_identity',''))
            or any(target in k.split('.') for k in model.state_dict())
            or any(model.heads[n].out_features!=DATASETS[n][2] for n in tasks)):
        raise ValueError('Source-only checkpoint target/config/provenance mismatch')
    if file_hash(path)!=weight_sha256 or file_hash(sidecar)!=sidecar_sha256:raise ValueError('Source checkpoint pair changed during load')
    return model,{'weight_sha256':weight_sha256,'sidecar_sha256':sidecar_sha256,
        'source_identity':provenance['source_identity'],'source_tasks':list(tasks),
        'training_profile':settings['profile'],'declared_optimizer_steps':settings['optimizer_steps'],
        'checkpoint_optimizer_step':meta.get('checkpoint_optimizer_step')}


def initialize_target(target,method,*,model_seed,head_seed,source_path=None,
                      source_weight_sha256=None,source_sidecar_sha256=None,device='cpu'):
    if target not in NAMES or method not in METHODS:raise ValueError('Unknown target initialization role/method')
    if any(type(s)is not int or not 0<=s<2**32 for s in (model_seed,head_seed)):raise ValueError('Explicit uint32 initialization seeds required')
    if method=='target_scratch_full' and any(x is not None for x in (source_path,source_weight_sha256,source_sidecar_sha256)):
        raise ValueError('Scratch must not receive a source checkpoint')
    base=load_config('configs/experiments/base97_joint.json')
    config=replace(base,name='g4_target_init_'+target,data=replace(base.data,datasets=[target]))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(model_seed);model=config.build_model({target:DATASETS[target][2]})
        torch.random.default_generator.manual_seed(head_seed)
        for family in (model.heads,model.hgnn_heads,model.cnn_heads):family[target].reset_parameters()
    fresh_domain=domain_keys(model.backbone);copied=[];receipt=None
    if method!='target_scratch_full':
        if source_path is None:raise ValueError('Transfer requires pinned source checkpoint')
        source,receipt=_source(source_path,source_weight_sha256,source_sidecar_sha256,target)
        source_cfg=dict(source.export_config()['backbone_kwargs']);target_cfg=dict(model.export_config()['backbone_kwargs'])
        source_cfg.pop('dataset_names',None);target_cfg.pop('dataset_names',None)
        if source_cfg!=target_cfg:raise ValueError('Source/target backbone architecture mismatch')
        before=state_id(source);source_state=source.backbone.state_dict();target_state=model.backbone.state_dict()
        shared=set(target_state)-fresh_domain
        if shared!=set(source_state)-domain_keys(source.backbone):raise ValueError('Shared backbone keys differ')
        for name in sorted(shared):
            if source_state[name].shape!=target_state[name].shape or source_state[name].dtype!=target_state[name].dtype:
                raise ValueError('Shared backbone tensor mismatch')
            target_state[name]=source_state[name].detach().clone();copied.append(name)
        model.backbone.load_state_dict(target_state,strict=True)
        if state_id(source)!=before:raise ValueError('Source model mutated')
        if (file_hash(source_path)!=source_weight_sha256
                or file_hash(str(source_path)+'.json')!=source_sidecar_sha256):raise ValueError('Source pair changed during copy')
    frozen=method=='source_pretrained_head_only';model.backbone.requires_grad_(not frozen)
    model=model.to(device);model.train()
    if frozen:model.backbone.eval()
    record={'policy':'g4_target_shared_weights_fresh_domains_v1','target':target,'method':method,
        'model_seed':model_seed,'head_seed':head_seed,'source':receipt,'copied_backbone_keys':copied,
        'fresh_target_domain_keys':sorted(fresh_domain),'source_heads_copied':False,
        'source_domain_state_copied':False,'head_only_requires_target_bn_calibration':frozen,
        'source_graph_context':'source_train_label_lda_slic' if receipt else None,
        'target_graph_required':'label_free_target_scene','target_initial_config':config.to_dict(),
        'initial_state_sha256':state_id(model),'optimizer_updates':0,'model_test_scoring':False}
    model.training_config={'target_initialization':copy.deepcopy(record)}
    return model,record
