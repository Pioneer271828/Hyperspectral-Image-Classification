"""Opt-in CPU PTQ/QAT matrix layers; existing model defaults are untouched.

Uses installed torch.ao observers/fake quantization and real quantized kernels.
The role argument is a caller contract, not authentication of dataset provenance.
Whole-model integration must bind it to a verified data manifest.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.ao.quantization import FakeQuantize, MinMaxObserver, PerChannelMinMaxObserver


TRAIN_ROLES = frozenset({'source_train', 'target_support', 'development_train'})


def _finite_cpu(x):
    if (not isinstance(x, torch.Tensor) or x.device.type != 'cpu'
            or x.dtype != torch.float32 or not x.numel() or not torch.isfinite(x).all()):
        raise ValueError('nonempty finite CPU float32 tensor required')


def _kind(layer):
    if type(layer) is nn.Linear:
        kind = 'linear'
    elif (type(layer) is nn.Conv2d and layer.kernel_size == (1, 1)
          and layer.stride == (1, 1) and layer.padding == (0, 0)
          and layer.dilation == (1, 1) and layer.groups == 1
          and layer.padding_mode == 'zeros'):
        kind = 'conv1x1'
    else:
        raise ValueError('only plain Linear or ungrouped unit-stride 1x1 Conv2d')
    if layer._forward_hooks or layer._forward_pre_hooks or layer._backward_hooks:
        raise ValueError('layer hooks are unsupported')
    for p in layer.parameters():
        _finite_cpu(p)
    # Bound worst-case int32 accumulator independently of observed ranges.
    if layer.weight[0].numel() * 127 * 127 > torch.iinfo(torch.int32).max:
        raise ValueError('input width exceeds int32 accumulator bound')
    return kind


def _activation(maximum):
    return FakeQuantize(observer=MinMaxObserver, quant_min=0, quant_max=maximum,
                        dtype=torch.quint8, qscheme=torch.per_tensor_affine)


class CalibratedMatrix(nn.Module):
    """Owned FP32 master, explicit training-only observation, then frozen ranges.

PTQ observes dense arithmetic. QAT observes simulated arithmetic and retains
trainable FP32 masters. Frozen forward is simulated, *not* an INT8 kernel.
Input uses 0..127 in uint8 storage to avoid x86 pairwise saturation; output
uses 0..255. This is a reduced-range activation baseline, stated explicitly.
"""
    def __init__(self, layer, *, qat=False):
        super().__init__()
        if type(qat) is not bool:
            raise ValueError('qat must be boolean')
        self.kind = _kind(layer)
        self.layer = copy.deepcopy(layer)
        self.qat = qat
        self.input_quant = _activation(127)
        self.weight_quant = FakeQuantize(observer=PerChannelMinMaxObserver,
            quant_min=-127, quant_max=127, dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric, ch_axis=0)
        self.output_quant = _activation(255)
        self.register_buffer('batches_seen', torch.tensor(0, dtype=torch.int64))
        self.register_buffer('ranges_frozen', torch.tensor(False))
        for quant in self.quantizers():
            quant.enable_fake_quant(qat)
        self.train(layer.training)

    def quantizers(self):
        return self.input_quant, self.weight_quant, self.output_quant

    def _validate(self, x):
        _finite_cpu(x)
        _kind(self.layer)
        n = self.layer.weight.shape[1]
        if self.kind == 'linear':
            if x.ndim < 2 or x.shape[-1] != n:
                raise ValueError('Linear input must have rank >=2 and matching width')
        elif x.ndim != 4 or x.shape[1] != n:
            raise ValueError('Conv input must be NCHW with matching channels')

    def _run(self, x):
        x = self.input_quant(x)
        w = self.weight_quant(self.layer.weight)
        y = (F.linear(x, w, self.layer.bias) if self.kind == 'linear' else
             F.conv2d(x, w, self.layer.bias))
        _finite_cpu(y)
        return self.output_quant(y)

    def observe(self, x, *, role):
        if role not in TRAIN_ROLES:
            raise ValueError('calibration requires an explicit training role')
        if bool(self.ranges_frozen):
            raise ValueError('calibration ranges are frozen')
        self._validate(x)
        result = self._run(x)
        self.batches_seen.add_(1)
        return result

    def freeze(self):
        if not int(self.batches_seen):
            raise ValueError('cannot freeze without calibration')
        for quant in self.quantizers():
            if not torch.isfinite(quant.scale).all() or not (quant.scale > 0).all():
                raise ValueError('invalid calibration scale')
            quant.disable_observer()
            quant.enable_fake_quant()
        self.ranges_frozen.fill_(True)
        return self

    def forward(self, x):
        if not bool(self.ranges_frozen):
            raise ValueError('use observe with training role before freezing')
        self._validate(x)
        return self._run(x)

    def convert(self):
        if not bool(self.ranges_frozen):
            raise ValueError('freeze calibration before conversion')
        _kind(self.layer)
        return IntegerMatrix(self)


class IntegerMatrix(nn.Module):
    """Real CPU INT8 kernel with explicit FP32 quantize/dequantize boundaries.

No FP32 master weight is retained. This is inference-only; it does not claim
an integer-only full network or automatic end-to-end acceleration.
"""
    def __init__(self, calibrated):
        super().__init__()
        if type(calibrated) is not CalibratedMatrix or not bool(calibrated.ranges_frozen):
            raise ValueError('frozen calibrated layer required')
        if torch.backends.quantized.engine not in ('x86', 'fbgemm'):
            raise ValueError('validated CPU x86/fbgemm engine required')
        self.engine = torch.backends.quantized.engine
        self.kind = calibrated.kind
        layer = calibrated.layer
        self.in_features = layer.weight.shape[1]
        self.register_buffer('input_scale', calibrated.input_quant.scale.detach().clone())
        self.register_buffer('input_zero_point', calibrated.input_quant.zero_point.detach().clone())
        wq = calibrated.weight_quant
        # FakeQuantize clamps to -127..127 before quantize_per_channel's storage range.
        w = wq(layer.weight).detach()
        qw = torch.quantize_per_channel(w, wq.scale.double(), wq.zero_point.long(), 0, torch.qint8)
        bias = None if layer.bias is None else layer.bias.detach().clone()
        m, n = layer.weight.shape[:2]
        self.kernel = (torch.ao.nn.quantized.Linear(n, m, bias is not None) if self.kind == 'linear'
                       else torch.ao.nn.quantized.Conv2d(n, m, 1, bias=bias is not None))
        self.kernel.set_weight_bias(qw, bias)
        self.kernel.scale = float(calibrated.output_quant.scale.item())
        self.kernel.zero_point = int(calibrated.output_quant.zero_point.item())
        self.eval()

    def forward(self, x):
        _finite_cpu(x)
        if self.training or x.requires_grad:
            raise ValueError('integer kernel is inference-only')
        if torch.backends.quantized.engine != self.engine:
            raise ValueError('quantized engine changed after construction')
        if self.kind == 'linear':
            if x.ndim < 2 or x.shape[-1] != self.in_features:
                raise ValueError('invalid linear input shape')
        elif x.ndim != 4 or x.shape[1] != self.in_features:
            raise ValueError('invalid convolution input shape')
        scale, zero = float(self.input_scale.item()), int(self.input_zero_point.item())
        bounded = x.clamp(-zero * scale, (127-zero) * scale)
        qx = torch.quantize_per_tensor(bounded, scale, zero, torch.quint8)
        return self.kernel(qx).dequantize()
