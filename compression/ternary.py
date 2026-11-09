"""Independent TTQ-style CPU weight simulation; no activation quantization.

Published threshold/latent-gradient rule, explicit signed scale derivative.
This adapter is not a reproduction of the author's TensorFlow experiment.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def finite(x):
    if (not torch.is_tensor(x) or x.dtype != torch.float32 or x.device.type != 'cpu'
            or not x.numel() or not torch.isfinite(x).all()):
        raise ValueError('nonempty finite CPU FP32 tensor required')


def assignments(weight, ratio):
    finite(weight)
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('finite explicit threshold ratio in [0,1] required')
    threshold = weight.detach().abs().max() * ratio
    return weight > threshold, weight < -threshold


def scales(positive, negative):
    for value in (positive, negative):
        finite(value)
        if value.numel() != 1 or not bool(value > 0):
            raise ValueError('finite positive scalar magnitudes required')


class _Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, positive, negative, ratio):
        scales(positive, negative)
        pos, neg = assignments(weight, ratio)
        ctx.save_for_backward(pos, neg, positive, negative)
        return pos.to(weight.dtype) * positive - neg.to(weight.dtype) * negative

    @staticmethod
    def backward(ctx, gradient):
        pos, neg, positive, negative = ctx.saved_tensors
        # The master-weight rule is an STE, not the derivative of hard masks.
        multiplier = torch.where(pos, positive, torch.where(neg, negative, 1.))
        return (gradient * multiplier,
                gradient[pos].sum().reshape_as(positive),
                -gradient[neg].sum().reshape_as(negative), None)


def ternarize(weight, positive, negative, ratio):
    return _Quantize.apply(weight, positive, negative, ratio)


class TernaryMatrix(nn.Module):
    """Owned master weights and positive magnitudes; invalid updates fail closed.

    Threshold equality maps to zero. Scales initialize from each selected sign's
    mean magnitude (empty sign: max magnitude, all-zero: 1). No hidden optimizer
    projection, normalization, or changes to the source layer are performed.
    """
    def __init__(self, layer, *, threshold_ratio):
        super().__init__()
        if type(layer) is nn.Linear:
            self.kind = 'linear'
        elif (type(layer) is nn.Conv2d and layer.kernel_size == (1, 1)
              and layer.stride == (1, 1) and layer.padding == (0, 0)
              and layer.dilation == (1, 1) and layer.groups == 1
              and layer.padding_mode == 'zeros'):
            self.kind = 'conv1x1'
        else:
            raise ValueError('plain Linear or ungrouped unit-stride 1x1 Conv2d required')
        if layer._forward_hooks or layer._forward_pre_hooks or layer._backward_hooks:
            raise ValueError('layer hooks unsupported')
        for p in layer.parameters(): finite(p)
        pos, neg = assignments(layer.weight, threshold_ratio)
        self.layer = copy.deepcopy(layer)
        self.register_buffer('threshold_ratio', torch.tensor(float(threshold_ratio)))
        fallback = layer.weight.detach().abs().max().clamp_min(torch.finfo(torch.float32).tiny)
        if not bool(layer.weight.detach().abs().max()): fallback = torch.tensor(1.)
        # FP64 reduction avoids overflow while averaging finite large FP32 weights.
        self.positive = nn.Parameter(layer.weight.detach()[pos].double().mean().float() if pos.any() else fallback.clone())
        self.negative = nn.Parameter(-layer.weight.detach()[neg].double().mean().float() if neg.any() else fallback.clone())

    def quantized_weight(self):
        finite(self.threshold_ratio)
        if self.threshold_ratio.numel() != 1: raise ValueError('scalar threshold required')
        return ternarize(self.layer.weight, self.positive, self.negative, float(self.threshold_ratio))

    def forward(self, x):
        finite(x)
        weight = self.quantized_weight()
        bias = self.layer.bias
        if bias is not None: finite(bias)
        if self.kind == 'linear':
            if x.ndim < 2 or x.shape[-1] != weight.shape[1]: raise ValueError('linear shape mismatch')
            result = F.linear(x, weight, bias)
        else:
            if x.ndim != 4 or x.shape[1] != weight.shape[1]: raise ValueError('convolution shape mismatch')
            result = F.conv2d(x, weight, bias)
        finite(result)
        return result
