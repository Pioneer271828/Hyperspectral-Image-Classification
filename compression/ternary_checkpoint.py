"""Authenticated independent ternary training or packed-inference model file."""
import copy
import hashlib
import io
import os
from pathlib import Path
import tempfile

import torch

from compression.int8_checkpoint import _check_state
from compression.int8_model import _paths, _replace
from compression.ternary import TernaryMatrix
from compression.ternary_model import PackedTernary, validate_model
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest


def save_ternary(model, path):
    plan = validate_model(model); state = model.state_dict(); _check_state(state)
    packet = dict(format='vca-ternary-cpu-v1', architecture=model.export_config(), plan=plan,
                  state=state, trainable={k:p.requires_grad for k,p in model.named_parameters()},
                  training_config=copy.deepcopy(getattr(model,'training_config',{})))
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as handle:
            temp=Path(handle.name); torch.save(packet,handle); handle.flush(); os.fsync(handle.fileno())
        os.link(temp,path)
    finally:
        if temp is not None: temp.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ternary(path, expected_sha256):
    payload=Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=expected_sha256: raise ValueError('ternary file hash mismatch')
    packet=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    if (not isinstance(packet,dict) or set(packet)!={'format','architecture','plan','state','trainable','training_config'}
            or packet['format']!='vca-ternary-cpu-v1'): raise ValueError('not a ternary checkpoint')
    plan=packet['plan']; spec=plan['spec']
    if digest(spec)!=plan['spec_hash'] or spec['architecture']!=packet['architecture']:
        raise ValueError('ternary specification hash/architecture mismatch')
    paths=_paths(spec['paths'])
    if plan['phase'] not in ('training','packed'): raise ValueError('invalid phase')
    _check_state(packet['state'])
    # load_state_dict silently casts tensors; prohibit this before copying.
    for key,value in packet['state'].items():
        if not torch.is_tensor(value): raise ValueError('tensor-only ternary model state required')
        if key.endswith('.payload') and value.dtype!=torch.uint8: raise ValueError('packed payload dtype mismatch')
    cfg=packet['architecture']
    with torch.random.fork_rng(devices=[]):
        model=MultiDatasetVCASAHGCN(cfg['class_counts'],**cfg['backbone_kwargs'])
        for path in paths:
            layer=TernaryMatrix(model.get_submodule(path),threshold_ratio=spec['threshold_ratio'])
            _replace(model,path,PackedTernary(layer) if plan['phase']=='packed' else layer)
    reference=model.state_dict()
    if set(reference)!=set(packet['state']) or any(reference[k].dtype!=v.dtype or reference[k].shape!=v.shape for k,v in packet['state'].items()):
        raise ValueError('ternary state schema mismatch')
    model.load_state_dict(packet['state'],strict=True); model._tq_plan=copy.deepcopy(plan)
    params=dict(model.named_parameters()); trainable=packet['trainable']
    if set(params)!=set(trainable) or any(type(v) is not bool for v in trainable.values()): raise ValueError('trainability mismatch')
    for name,p in params.items(): p.requires_grad_(trainable[name])
    model.training_config=copy.deepcopy(packet['training_config']); model.backbone.adapter.cache_attention=False
    validate_model(model)
    return model.eval()
