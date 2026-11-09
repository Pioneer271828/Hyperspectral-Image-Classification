"""Opt-in, label-free scene-population calibration of front-end DomainBN.

Independent project adaptation of BN moments / layer-wise precise statistics;
references and upstream inspection: docs/references/bn-calibration-review.json.
No training entry enables this policy implicitly.
"""
import copy

import torch

from model.VCASAHGCN import MultiDatasetVCASAHGCN, DomainSpecificBatchNorm2d
from model.research_config import digest
from research_runtime import tensor_hash

CONTEXT = 'acc1p_full_scene_transductive'
POLICY = 'scene_population_denoise_bn_v1'
OBSERVATION_KEYS = {'net_input', 'wavelengths', 'fwhm'}


def state_digest(model):
    return digest({n:tensor_hash(t) for n,t in model.state_dict().items()})


def observation_id(observations):
    result = {}
    for name, item in observations.items():
        if not isinstance(item,dict) or set(item)!=OBSERVATION_KEYS:
            raise ValueError('Calibration accepts only net_input/wavelengths/fwhm; no labels')
        if any(v is not None and (not torch.is_tensor(v) or not torch.isfinite(v).all())
               for v in item.values()):
            raise ValueError('Calibration observations must be finite tensors or optional None')
        x=item['net_input']
        if (x is None or x.ndim!=3 or min(x.shape[:2])<2
                or x.dtype not in (torch.float32,torch.float64)):
            raise ValueError('Calibration requires a nonempty FP32/FP64 HWC scene')
        result[name]={k:tensor_hash(v) if v is not None else None for k,v in item.items()}
    return result


def calibrate_denoise(model, observations=None, *, mode='none', context=None,
                      source_checkpoint_sha256=None):
    """Return (model, receipt); enabled mode calibrates a separate eval clone.

    A scene is the explicitly allowed population, including unlabeled pixels.
    Variance uses correction0 to reproduce BN's scene-batch normalization,
    distinct from PyTorch's unbiased EMA update. Each next layer sees the
    already calibrated preceding layers. This is not a mini-batch estimator
    or an optimizer-resume checkpoint.
    """
    if mode=='none':
        return model, {'policy':'none','optimizer_updates':0}
    if mode!=POLICY or context!=CONTEXT:
        raise ValueError('Unsupported calibration mode or data-access context')
    if (not isinstance(source_checkpoint_sha256,str) or len(source_checkpoint_sha256)!=64
            or any(c not in '0123456789abcdef' for c in source_checkpoint_sha256)):
        raise ValueError('Explicit source checkpoint SHA256 required')
    if not isinstance(model,MultiDatasetVCASAHGCN) or any(m.training for m in model.modules()):
        raise ValueError('Calibration requires a fully eval MultiDatasetVCASAHGCN')
    if not observations or set(observations)-set(model.heads):
        raise ValueError('Nonempty registered scene domains required')
    if model.backbone.coverage_modulation is not None:
        raise ValueError('This calibration gate supports the unmodulated baseline only')
    identities=observation_id(observations)
    for block in model.backbone.CNN_denoise:
        if not isinstance(block.norm.norm,DomainSpecificBatchNorm2d):
            raise ValueError('Front-end DomainBN required')
    # Keep the source object, buffers, modes, and caches untouched, even on failure.
    result=copy.deepcopy(model)
    previous={n:t.detach().clone() for n,t in result.state_dict().items()}
    targets=set();details={}
    reference=next(result.parameters())
    with torch.no_grad():
        for name,item in observations.items():
            item={k:v.to(device=reference.device,dtype=reference.dtype) if v is not None else None
                  for k,v in item.items()}
            spectral,extra=result.backbone._split_spectral_extra(item['net_input'],item['wavelengths'])
            latent=result.backbone.adapter(spectral,wavelengths=item['wavelengths'],fwhm=item['fwhm'])
            noise=torch.cat((latent,extra),dim=2).permute(2,0,1).unsqueeze(0)
            details[name]=[]
            for i,block in enumerate(result.backbone.CNN_denoise):
                bn=block.norm.norm.norms[name]
                if not bn.track_running_stats or bn.running_mean is None or bn.running_var is None:
                    raise ValueError('Tracked DomainBN statistics required')
                variance,mean=torch.var_mean(noise,dim=(0,2,3),unbiased=False)
                if not torch.isfinite(variance).all() or not torch.isfinite(mean).all():
                    raise ValueError('Nonfinite calibration features')
                prefix=f'backbone.CNN_denoise.{i}.norm.norm.norms.{name}.'
                targets.update((prefix+'running_mean',prefix+'running_var'))
                details[name].append({'layer':prefix[:-1],'samples_per_channel':noise.numel()//noise.shape[1],
                    'mean_max_abs_change':float((mean-bn.running_mean).abs().max()),
                    'var_max_abs_change':float((variance-bn.running_var).abs().max())})
                bn.running_mean.copy_(mean);bn.running_var.copy_(variance)
                noise=block(noise,name)
    for name,value in result.state_dict().items():
        if name not in targets and not torch.equal(value,previous[name]):
            raise RuntimeError('Calibration changed protected state: '+name)
    receipt={'schema_version':1,'policy':POLICY,'context':context,
             'source_checkpoint_sha256':source_checkpoint_sha256,
             'scene_identity':identities,'variance':'scene_population_correction0',
             'calibrated_buffers':sorted(targets),'layers':details,'optimizer_updates':0,
             'parameters_unchanged':True,'state_sha256':state_digest(result)}
    result.training_config=copy.deepcopy(getattr(model,'training_config',{}))
    result.training_config['bn_calibration']=receipt
    return result,receipt


def verify_calibration_binding(model, observations, *, context, source_checkpoint_sha256):
    """Check a saved/reloaded calibrated model before using it on bound scenes."""
    value=getattr(model,'training_config',{}).get('bn_calibration',{})
    if (context!=CONTEXT or value.get('schema_version')!=1 or value.get('policy')!=POLICY or value.get('context')!=context
            or value.get('source_checkpoint_sha256')!=source_checkpoint_sha256
            or value.get('scene_identity')!=observation_id(observations)
            or value.get('state_sha256')!=state_digest(model)):
        raise ValueError('Calibrated checkpoint state/scene/context binding mismatch')
    if any(m.training for m in model.modules()):
        raise ValueError('Calibrated checkpoint must be used in eval mode')
    return value
