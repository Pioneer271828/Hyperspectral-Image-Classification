"""Separate CPU quantized VCA checkpoint; authenticate bytes before loading."""
import copy
import hashlib
import io
import os
from pathlib import Path
import tempfile

import torch

from compression.int8 import CalibratedMatrix
from compression.int8_model import _paths, _replace, validate_plan
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest


def _check_state(value):
    if torch.is_tensor(value):
        x = value.dequantize() if value.is_quantized else value
        if x.device.type != 'cpu' or not torch.isfinite(x).all():
            raise ValueError('nonfinite or non-CPU checkpoint tensor')
    elif isinstance(value, dict):
        for v in value.values(): _check_state(v)
    elif isinstance(value, (tuple, list)):
        for v in value: _check_state(v)


def save_int8(model, path):
    plan = validate_plan(model)
    state = model.state_dict()
    _check_state(state)
    packet = dict(format='vca-int8-cpu-v1', architecture=model.export_config(),
                  plan=plan, state=state, torch_version=str(torch.__version__),
                  trainable={k: p.requires_grad for k, p in model.named_parameters()},
                  training_config=copy.deepcopy(getattr(model, 'training_config', {})))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temp = Path(handle.name)
            torch.save(packet, handle)
            handle.flush(); os.fsync(handle.fileno())
        os.link(temp, path)
    finally:
        if temp is not None: temp.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_int8(path, expected_sha256):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('quantized checkpoint hash mismatch')
    packet = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if (not isinstance(packet, dict) or set(packet) != {'format', 'architecture', 'plan',
            'state', 'torch_version', 'trainable', 'training_config'}
            or packet['format'] != 'vca-int8-cpu-v1'):
        raise ValueError('not an INT8 model checkpoint')
    if packet['torch_version'] != str(torch.__version__):
        raise ValueError('quantized checkpoint requires the recorded torch version')
    plan = packet['plan']; contract = plan['contract']
    if digest(contract) != plan['contract_hash'] or contract['architecture'] != packet['architecture']:
        raise ValueError('calibration identity mismatch')
    paths = _paths(contract['paths'])
    if contract['engine'] != torch.backends.quantized.engine:
        raise ValueError('quantized engine differs from recorded engine')
    if plan['phase'] not in ('ptq', 'qat', 'integer'):
        raise ValueError('unknown quantized checkpoint phase')
    _check_state(packet['state'])
    cfg = packet['architecture']
    with torch.random.fork_rng(devices=[]):
        model = MultiDatasetVCASAHGCN(cfg['class_counts'], **cfg['backbone_kwargs'])
        for path in paths:
            layer = CalibratedMatrix(model.get_submodule(path), qat=contract['qat'])
            if plan['phase'] == 'integer':
                # Shape-only packing placeholder; no input/labels or calibration
                # performed. All packed weights and qparams are replaced below.
                layer.weight_quant(layer.layer.weight)
                layer.batches_seen.fill_(1)
                layer.freeze()
                layer = layer.convert()
            _replace(model, path, layer)
    model.load_state_dict(packet['state'], strict=True)
    model._int8_plan = copy.deepcopy(plan)
    params = dict(model.named_parameters())
    if set(params) != set(packet['trainable']) or any(type(v) is not bool for v in packet['trainable'].values()):
        raise ValueError('trainable parameter contract differs')
    for name, param in params.items(): param.requires_grad_(packet['trainable'][name])
    model.training_config = packet['training_config']
    model.backbone.adapter.cache_attention = False
    validate_plan(model)
    _check_state(model.state_dict())
    return model.eval()
