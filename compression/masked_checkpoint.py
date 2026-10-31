"""Explicit masked training checkpoints; legacy checkpoint functions remain unchanged."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from .weight_masks import MaskedWeights
from torch.nn.utils import parametrize

OPTIMIZERS = (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW)
SCHEDULERS = (torch.optim.lr_scheduler.StepLR, torch.optim.lr_scheduler.ReduceLROnPlateau)


def _cpu(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value, list): return [_cpu(v) for v in value]
    if isinstance(value, tuple): return tuple(_cpu(v) for v in value)
    return copy.deepcopy(value)


def _names(model, optimizer, controller=None):
    mapping = {}
    for name, parameter in model.named_parameters():
        if controller is not None:
            for path in controller._masks:
                if name == path[:-len('weight')]+'parametrizations.weight.original': name=path
        if 'parametrizations.' in name: raise ValueError('Unmanaged parametrization')
        mapping[id(parameter)] = name
    groups = []
    seen = set()
    for group in optimizer.param_groups:
        names=[]
        for parameter in group['params']:
            if id(parameter) not in mapping or id(parameter) in seen: raise ValueError('Foreign/duplicate optimizer parameter')
            names.append(mapping[id(parameter)]);seen.add(id(parameter))
        groups.append(names)
    return groups


def _devices(model):
    return sorted({str(p.device) for p in model.parameters()})


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _validate_runtime(model, optimizer, scheduler, sampler):
    if type(optimizer) not in OPTIMIZERS: raise ValueError('Only SGD/Adam/AdamW are supported')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Save/load only at a step boundary after zero_grad(set_to_none=True)')
    if scheduler is not None and (type(scheduler) not in SCHEDULERS or scheduler.optimizer is not optimizer):
        raise ValueError('Supported scheduler must reference the supplied optimizer')
    if sampler is not None and (not isinstance(sampler,np.random.Generator) or type(sampler.bit_generator) is not np.random.PCG64):
        raise ValueError('Only an explicit NumPy PCG64 sampler is supported')


def _architecture(model):
    modules=[]
    for name,module in model.named_modules():
        if 'parametrizations' in name.split('.'): continue
        kind=type(module).__bases__[0] if parametrize.is_parametrized(module) else type(module)
        modules.append((name,kind.__module__+'.'+kind.__qualname__,module.extra_repr()))
    return {'modules':modules,'export_config':_json(model.export_config()) if hasattr(model,'export_config') else None}
def _capture_rng(model):
    state=np.random.get_state()
    return {'python':random.getstate(), 'numpy':(state[0],state[1].tolist(),state[2],state[3],state[4]),
            'torch':torch.get_rng_state(),
            'cuda':{device:torch.cuda.get_rng_state(torch.device(device)).cpu()
                    for device in _devices(model) if device.startswith('cuda')}}


def _restore_rng(rng):
    random.setstate(rng['python'])
    name,keys,pos,has_gauss,cached=rng['numpy']
    np.random.set_state((name,np.array(keys,dtype=np.uint32),pos,has_gauss,cached))
    torch.set_rng_state(rng['torch'])
    for device,state in rng['cuda'].items(): torch.cuda.set_rng_state(state,torch.device(device))


def save_masked_training(path, controller, optimizer, *, identity, progress, scheduler=None, sampler=None):
    """Atomically create a new weights_only-compatible file; do not overwrite prior work."""
    if not isinstance(controller,MaskedWeights) or not controller.active: raise ValueError('Active mask controller required')
    model=controller.model
    _validate_runtime(model,optimizer,scheduler,sampler)
    identity,progress=_json(identity),_json(progress)
    if not identity or type(progress.get('step')) is not int or progress['step']<0: raise ValueError('Identity and actual step required')
    state,snapshots=controller._checkpoint_payload()
    payload={'schema_version':1,'kind':'masked_training_checkpoint','identity':identity,'progress':progress,
             'devices':_devices(model),'architecture':_architecture(model),'training_config':_json(getattr(model,'training_config',{})),
             'model':state,'masks':controller.mask_plan(),
             'recovery':controller.recovery,'masked_restore_values':snapshots,
             'model_modes':{name:module.training for name,module in model.named_modules()
                            if 'parametrizations' not in name.split('.')},
             'optimizer_type':type(optimizer).__name__,'optimizer_names':_names(model,optimizer,controller),
             'optimizer':_cpu(optimizer.state_dict()),
             'scheduler_type':type(scheduler).__name__ if scheduler is not None else None,
             'scheduler':_cpu(scheduler.state_dict()) if scheduler is not None else None,
             'sampler':copy.deepcopy(sampler.bit_generator.state) if sampler is not None else None,
             'rng':_capture_rng(model)}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    descriptor,temp=tempfile.mkstemp(prefix='.masked-checkpoint-',dir=path.parent)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        os.link(temp,path)  # Atomic no-clobber publication on the training container filesystem.
    finally:
        Path(temp).unlink(missing_ok=True)
    return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'step':progress['step']}


def load_masked_training(path, model, optimizer, *, expected_sha256, expected_identity, scheduler=None, sampler=None):
    """Load into a fresh same-structure model/optimizer; return an already active controller.

    Parameter groups must match canonical names AND order, not parametrized traversal order.
    Validate identity/shape/group contracts before modifying the supplied model.
    """
    _validate_runtime(model,optimizer,scheduler,sampler)
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256: raise ValueError('Checkpoint hash mismatch')
    import io
    payload=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if payload.get('kind')!='masked_training_checkpoint' or payload.get('schema_version')!=1:
        raise ValueError('Unsupported masked checkpoint')
    if payload['identity']!=_json(expected_identity): raise ValueError('Experiment identity mismatch')
    if payload['devices']!=_devices(model): raise ValueError('Device migration is not supported in this version')
    if payload['optimizer_type']!=type(optimizer).__name__ or payload['optimizer_names']!=_names(model,optimizer):
        raise ValueError('Optimizer type, canonical parameter groups or order mismatch')
    if payload['scheduler_type']!=(type(scheduler).__name__ if scheduler is not None else None):
        raise ValueError('Scheduler mismatch')
    if (payload['sampler'] is None)!=(sampler is None): raise ValueError('Explicit sampler mismatch')
    if payload['architecture']!=_architecture(model): raise ValueError('Model configuration mismatch')
    current=model.state_dict()
    if current.keys()!=payload['model'].keys() or any(v.shape!=payload['model'][k].shape or v.dtype!=payload['model'][k].dtype for k,v in current.items()):
        raise ValueError('Model architecture/dtype mismatch')
    controller=MaskedWeights.from_plan(model,payload['masks'],recovery=payload['recovery'])
    # Validate mask targets before loading weights; immediately remove without changing fresh values.
    controller.attach();controller.detach()
    model.load_state_dict(payload['model'],strict=True)
    model.training_config=copy.deepcopy(payload['training_config'])
    controller.attach()
    try:
        for path,entry in zip(controller._masks,controller._attached):
            saved=payload['masked_restore_values'][path]
            if saved.shape!=entry[3].shape or saved.dtype!=entry[3].dtype: raise ValueError('Masked restore snapshot mismatch')
        controller._attached=[(m,p,k,payload['masked_restore_values'][path].to(p.device).clone())
                              for path,(m,p,k,_) in zip(controller._masks,controller._attached)]
        optimizer.load_state_dict(payload['optimizer'])
        if scheduler is not None: scheduler.load_state_dict(payload['scheduler'])
        for name,training in payload['model_modes'].items(): model.get_submodule(name).training=training
        if sampler is not None: sampler.bit_generator.state=payload['sampler']
        _restore_rng(payload['rng'])
    except BaseException:
        controller.detach()
        raise
    return controller,copy.deepcopy(payload['progress'])

