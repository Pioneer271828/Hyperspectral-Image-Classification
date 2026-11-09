"""Explicit target population statistics; no labels, no optimizer or default change."""
import copy,re
import torch
from joint_baseline import forward
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest
from research_runtime import tensor_hash
from target_initialization import state_id

POLICY='g4_scene_population_all_bn_v1'
CONTEXT='g4_full_target_scene_transductive'
GRAPH_POLICY='g4_target_label_free_static_v1'
OBS_KEYS={'net_input','wavelengths','fwhm','Q','A','superpixel_scale'}


def observation_id(observation):
    if not isinstance(observation,dict) or set(observation)!=OBS_KEYS:raise ValueError('Only unlabeled observation fields accepted')
    x=observation['net_input'];q=observation['Q'];a=observation['A'];wave=observation['wavelengths']
    if (not torch.is_tensor(x) or x.ndim!=3 or min(x.shape[:2])<2 or x.dtype not in (torch.float32,torch.float64)
            or not torch.is_tensor(q) or q.layout!=torch.sparse_coo or q.ndim!=2
            or not torch.is_tensor(a) or a.layout not in (torch.strided,torch.sparse_coo)
            or a.shape!=(q.shape[1],q.shape[1]) or q.shape[0]!=x.shape[0]*x.shape[1]
            or not torch.is_tensor(wave) or wave.ndim!=1 or len(wave)!=x.shape[2]-2
            or type(observation['superpixel_scale'])is not int or observation['superpixel_scale']<=0):
        raise ValueError('Target observation geometry/dtype mismatch')
    for value in (x,q,a,wave,observation['fwhm']):
        if value is None:continue
        if not torch.is_tensor(value) or value.dtype!=x.dtype:raise ValueError('Target observation tensors must share model dtype')
        values=value.coalesce().values() if value.layout==torch.sparse_coo else value
        if not torch.isfinite(values).all():raise ValueError('Nonfinite target observation')
    return {k:tensor_hash(v) if torch.is_tensor(v) else v for k,v in observation.items()}


def backbone_id(model):return digest({k:tensor_hash(v) for k,v in model.backbone.state_dict().items()})


def calibrate_target(model,observation=None,*,mode='none',context=None,graph_policy=None,source_state_sha256=None):
    if mode=='none':return model,{'policy':'none','optimizer_updates':0}
    if mode!=POLICY or context!=CONTEXT or graph_policy!=GRAPH_POLICY:raise ValueError('Explicit G4 target context/graph policy required')
    if (not isinstance(source_state_sha256,str) or not re.fullmatch('[0-9a-f]{64}',source_state_sha256)
            or state_id(model)!=source_state_sha256):raise ValueError('Pinned calibration source state differs')
    if not isinstance(model,MultiDatasetVCASAHGCN) or len(model.heads)!=1 or any(m.training for m in model.modules()):
        raise ValueError('Single-target fully eval model required')
    target=next(iter(model.heads));initial=getattr(model,'training_config',{}).get('target_initialization',{})
    if (initial.get('policy')!='g4_target_shared_weights_fresh_domains_v1' or initial.get('target')!=target
            or model.backbone.graph_mode!='static' or model.backbone.norm_type!='domain_bn'
            or model.backbone.coverage_modulation is not None):raise ValueError('Accepted target initialization required')
    identity=observation_id(observation);reference=next(model.parameters())
    if observation['net_input'].dtype!=reference.dtype:raise ValueError('Observation/model dtype mismatch')
    result=copy.deepcopy(model);before={n:t.detach().clone() for n,t in result.state_dict().items()}
    layers={n:m for n,m in result.named_modules() if isinstance(m,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d))}
    if (not layers or any(not n.endswith('.norms.'+target) or not m.track_running_stats
            or m.running_mean is None or m.running_var is None for n,m in layers.items())):
        raise ValueError('Every BN must belong only to the declared target with tracked state')
    seen=[];rows=[];changed=set();hooks=[]
    def make_hook(name):
        def hook(module,args):
            if name in seen:raise ValueError('Reused target BN is outside this calibration contract')
            x=args[0]
            if x.ndim not in (2,3,4) or not torch.isfinite(x).all():raise ValueError('Invalid BN population')
            axes=(0,*range(2,x.ndim));variance,mean=torch.var_mean(x,dim=axes,unbiased=False)
            if not torch.isfinite(variance).all() or not torch.isfinite(mean).all():raise ValueError('Nonfinite target moments')
            module.running_mean.copy_(mean);module.running_var.copy_(variance)
            seen.append(name);changed.update((name+'.running_mean',name+'.running_var'))
            rows.append({'layer':name,'population_units':'regions' if isinstance(module,torch.nn.BatchNorm1d) else 'pixels',
                'samples_per_channel':x.numel()//x.shape[1],'variance':'population_correction0'})
        return hook
    try:
        for name,bn in layers.items():hooks.append(bn.register_forward_pre_hook(make_hook(name)))
        moved={k:v.to(reference.device) if torch.is_tensor(v) else v for k,v in observation.items()}
        with torch.no_grad():outputs=forward(result,target,moved)
        if set(seen)!=set(layers) or any(not torch.isfinite(v).all() for v in outputs[:3]):
            raise ValueError('Missing BN execution/nonfinite target forward')
    finally:
        for hook in hooks:hook.remove()
    for name,value in result.state_dict().items():
        if name not in changed and not torch.equal(value,before[name]):raise RuntimeError('Calibration changed protected state: '+name)
    if state_id(model)!=source_state_sha256:raise RuntimeError('Original target model mutated')
    receipt={'schema_version':1,'policy':POLICY,'context':context,'graph_policy':graph_policy,'target':target,
        'source_state_sha256':source_state_sha256,'observation':identity,'layers':rows,
        'calibrated_buffers':sorted(changed),'parameters_affine_counters_unchanged':True,
        'state_sha256':state_id(result),'backbone_sha256':backbone_id(result),
        'optimizer_updates':0,'model_test_scoring':False}
    result.training_config=copy.deepcopy(model.training_config)
    result.training_config['target_bn_calibration']=receipt
    result.training_config['target_initialization']['head_only_requires_target_bn_calibration']=False
    return result,receipt


def verify_target_binding(model,observation,*,context,graph_policy,source_state_sha256,backbone_only=False):
    receipt=getattr(model,'training_config',{}).get('target_bn_calibration',{})
    if (context!=CONTEXT or graph_policy!=GRAPH_POLICY or receipt.get('policy')!=POLICY
            or receipt.get('context')!=context or receipt.get('graph_policy')!=graph_policy
            or receipt.get('source_state_sha256')!=source_state_sha256
            or receipt.get('observation')!=observation_id(observation)
            or receipt.get('backbone_sha256')!=backbone_id(model)
            or (not backbone_only and receipt.get('state_sha256')!=state_id(model))):
        raise ValueError('Target calibration model/observation/context binding mismatch')
    if any(m.training for m in model.backbone.modules()) or (not backbone_only and any(m.training for m in model.modules())):
        raise ValueError('Target calibration inference/backbone requires eval mode')
    return receipt
