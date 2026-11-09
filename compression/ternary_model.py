"""Default-off CNN weight ternarization and packed-weight FP32 inference."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from compression.int8_model import ALLOWED_PATHS, _paths, _replace, _dense
from compression.ternary import TernaryMatrix, finite
from compression.ternary_codec import encode, decode
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest
from transfer.g3_training import state_hash


class PackedTernary(nn.Module):
    """No master weight: decode canonical 2-bit bytes for each FP32 forward.

    This deliberately includes decoding overhead and claims no sparse skipping.
    """
    def __init__(self, layer):
        super().__init__()
        if type(layer) is not TernaryMatrix: raise ValueError('ternary layer required')
        payload = encode(layer.layer.weight, layer.positive, layer.negative, float(layer.threshold_ratio))
        self.kind = layer.kind
        self.weight_shape = tuple(layer.layer.weight.shape)
        self.register_buffer('payload', torch.tensor(list(payload), dtype=torch.uint8))
        self.register_buffer('bias', torch.empty(0) if layer.layer.bias is None else layer.layer.bias.detach().clone())
        self.has_bias = layer.layer.bias is not None
        self.eval()

    def decoded_weight(self):
        if self.payload.dtype != torch.uint8 or self.payload.device.type != 'cpu' or self.payload.ndim != 1:
            raise ValueError('CPU uint8 payload vector required')
        value = decode(bytes(self.payload.tolist()))
        if tuple(value.shape) != self.weight_shape: raise ValueError('packed weight shape mismatch')
        if self.has_bias:
            finite(self.bias)
            if tuple(self.bias.shape) != (self.weight_shape[0],): raise ValueError('packed bias shape mismatch')
        elif self.bias.numel() or self.bias.dtype != torch.float32 or self.bias.device.type != 'cpu':
            raise ValueError('unexpected packed bias')
        return value

    def forward(self, x):
        finite(x)
        if self.training or x.requires_grad: raise ValueError('packed model is inference-only')
        w = self.decoded_weight(); bias = self.bias if self.has_bias else None
        if self.kind == 'linear':
            if x.ndim < 2 or x.shape[-1] != w.shape[1]: raise ValueError('linear input mismatch')
            out = F.linear(x, w, bias)
        else:
            if x.ndim != 4 or x.shape[1] != w.shape[1]: raise ValueError('convolution input mismatch')
            out = F.conv2d(x, w, bias)
        finite(out)
        return out


def ternarize_model(model, *, paths=(), threshold_ratio=None):
    _dense(model)
    if hasattr(model, '_tq_plan') or any(isinstance(m, (TernaryMatrix, PackedTernary)) for m in model.modules()):
        raise ValueError('already ternarized')
    if not paths:
        if threshold_ratio is not None: raise ValueError('disabled plan must not carry a threshold')
        return copy.deepcopy(model)
    paths = _paths(paths)
    if type(threshold_ratio) not in (int, float) or not math.isfinite(threshold_ratio) or not 0 <= threshold_ratio <= 1:
        raise ValueError('explicit finite threshold ratio required')
    ratio = float(torch.tensor(threshold_ratio, dtype=torch.float32))
    result = copy.deepcopy(model).eval()
    for path in paths:
        source = result.get_submodule(path)
        wrapped = TernaryMatrix(source, threshold_ratio=ratio)
        wrapped.positive.requires_grad_(source.weight.requires_grad)
        wrapped.negative.requires_grad_(source.weight.requires_grad)
        _replace(result, path, wrapped)
    result.backbone.adapter.cache_attention = False
    spec = dict(schema='vca-ternary-spec-v1', paths=paths, threshold_ratio=ratio,
                source_state=state_hash(model), architecture=model.export_config())
    result._tq_plan = dict(phase='training', spec=spec, spec_hash=digest(spec))
    return result.eval()


def validate_model(model):
    if type(model) is not MultiDatasetVCASAHGCN or not hasattr(model, '_tq_plan') or hasattr(model, '_int8_plan'):
        raise ValueError('explicit unmixed ternary model required')
    plan = model._tq_plan
    if (set(plan) != {'phase', 'spec', 'spec_hash'} or plan['phase'] not in ('training', 'packed')
            or digest(plan['spec']) != plan['spec_hash']): raise ValueError('invalid ternary plan')
    spec = plan['spec']
    if (set(spec) != {'schema','paths','threshold_ratio','source_state','architecture'}
            or spec['schema'] != 'vca-ternary-spec-v1' or spec['architecture'] != model.export_config()
            or type(spec['source_state']) is not str or len(spec['source_state']) != 64
            or any(c not in '0123456789abcdef' for c in spec['source_state'])
            or type(spec['threshold_ratio']) is not float or not math.isfinite(spec['threshold_ratio'])
            or not 0 <= spec['threshold_ratio'] <= 1): raise ValueError('invalid ternary specification')
    paths = _paths(spec['paths']); expected = TernaryMatrix if plan['phase']=='training' else PackedTernary
    actual = {n for n,m in model.named_modules() if isinstance(m,(TernaryMatrix,PackedTernary))}
    if actual != set(paths): raise ValueError('ternary topology mismatch')
    for path in paths:
        module = model.get_submodule(path)
        if type(module) is not expected: raise ValueError('ternary phase mismatch')
        if expected is TernaryMatrix:
            if float(module.threshold_ratio) != spec['threshold_ratio']: raise ValueError('threshold mismatch')
            module.quantized_weight()
        else: module.decoded_weight()
    return copy.deepcopy(plan)


def pack_model(model):
    plan = validate_model(model)
    if plan['phase'] != 'training': raise ValueError('already packed')
    result = copy.deepcopy(model).eval()
    for path in plan['spec']['paths']: _replace(result,path,PackedTernary(result.get_submodule(path)))
    plan['phase'] = 'packed'; result._tq_plan = plan
    return result.eval()
