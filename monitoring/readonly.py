"""Explicit block-level measurements. No training hooks are installed by default."""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import threading
import weakref
import torch

LAYOUTS = {'HWC': (3, 2), 'NCHW': (4, 1), 'NC': (2, 1)}
REQUIRED_CONTEXT = {'run_id', 'commit', 'checkpoint_hash', 'data_hash', 'split_hash',
                    'dataset', 'partition', 'step', 'sample_id', 'condition'}
_ACTIVE = weakref.WeakSet()
_LOCK = threading.Lock()


def threshold(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError('Threshold must be finite and nonnegative')
    return float(value)


class ChannelStats:
    """Mergeable float64 channel moments; one update is one spatial block, not a pixel sample."""
    def __init__(self, channels, epsilon):
        if type(channels) is not int or channels <= 0: raise ValueError('Positive channel count required')
        self.channels, self.epsilon = channels, threshold(epsilon)
        self.count = self.blocks = 0
        self.mean = torch.zeros(channels, dtype=torch.float64)
        self.m2 = self.mean.clone(); self.square = self.mean.clone(); self.absolute = self.mean.clone()
        self.positive = self.mean.clone(); self.negative = self.mean.clone(); self.near = self.mean.clone()
        self.block_response = self.mean.clone()
        self.minimum = torch.full_like(self.mean, float('inf')); self.maximum = -self.minimum

    def update(self, value, layout, mask=None):
        if layout not in LAYOUTS or not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError('Floating tensor and explicit HWC/NCHW/NC layout required')
        rank, axis = LAYOUTS[layout]
        if value.ndim != rank or value.shape[axis] != self.channels or (layout == 'NCHW' and value.shape[0] != 1):
            raise ValueError('Shape/layout mismatch; NCHW supports one block (N=1)')
        x = value.detach().movedim(axis, -1)
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != tuple(x.shape[:-1]):
                raise ValueError('Boolean mask must exactly match nonchannel dimensions')
            x = x[mask.to(x.device)]
        x = x.reshape(-1, self.channels).to(torch.float64)
        if not torch.isfinite(x).all(): raise ValueError('Nonfinite activation/weight: measurement rejected')
        if not len(x): return
        n = len(x)
        # Reduce on the source device. Only O(channels) statistics are moved to CPU.
        batch_mean = x.mean(0).cpu(); batch_m2 = ((x-x.mean(0))**2).sum(0).cpu()
        delta = batch_mean-self.mean; total = self.count+n
        self.m2 += batch_m2 + delta.square() * (self.count*n/total)
        self.mean += delta*(n/total)
        self.count = total; self.blocks += 1
        self.square += x.square().sum(0).cpu(); self.absolute += x.abs().sum(0).cpu()
        self.positive += (x > self.epsilon).sum(0).cpu()
        self.negative += (x < -self.epsilon).sum(0).cpu()
        self.near += (x.abs() <= self.epsilon).sum(0).cpu()
        self.block_response += (x.abs().mean(0) > self.epsilon).cpu()
        self.minimum = torch.minimum(self.minimum, x.amin(0).cpu())
        self.maximum = torch.maximum(self.maximum, x.amax(0).cpu())

    def report(self):
        def elements(x): return (x/self.count).tolist() if self.count else [None]*self.channels
        return {'channels': self.channels, 'count_per_channel': self.count, 'valid_blocks': self.blocks,
            'epsilon': self.epsilon, 'mean': self.mean.tolist() if self.count else [None]*self.channels,
            'variance_population': elements(self.m2), 'rms': (self.square/self.count).sqrt().tolist() if self.count else [None]*self.channels,
            'mean_absolute': elements(self.absolute), 'positive_rate': elements(self.positive),
            'negative_rate': elements(self.negative), 'near_zero_rate': elements(self.near),
            'response_rate': elements(self.positive+self.negative),
            'block_response_rate': (self.block_response/self.blocks).tolist() if self.blocks else [None]*self.channels,
            'minimum': self.minimum.tolist() if self.count else [None]*self.channels,
            'maximum': self.maximum.tolist() if self.count else [None]*self.channels}


@dataclass(frozen=True)
class LayerSpec:
    path: str
    layout: str
    epsilon: float
    position: str
    unit: str
    selector: tuple = ()

    def __post_init__(self):
        if self.layout not in LAYOUTS or not self.path or not self.position or not self.unit:
            raise ValueError('Explicit module path, layout, position and unit required')
        threshold(self.epsilon)
        if not isinstance(self.selector, tuple) or any(type(k) not in (str, int) for k in self.selector):
            raise ValueError('Selector is a tuple of dictionary keys / sequence indices')


def weight_statistics(model, paths, epsilon):
    """Static direct parameters only; axis 0 is explicit and not a global importance ranking."""
    threshold(epsilon)
    modules = dict(model.named_modules()); results = []
    for path in paths:
        module = modules[path]
        for name, parameter in module.named_parameters(recurse=False):
            if parameter.ndim == 0: raise ValueError('Scalar parameters require a separate measurement policy')
            stats = ChannelStats(parameter.shape[0], epsilon)
            x = parameter.detach().reshape(parameter.shape[0], -1).T.to(torch.float64)
            stats.update(x, 'NC'); record = stats.report()
            record.update(module=path, parameter=name, shape=list(parameter.shape), dtype=str(parameter.dtype),
                          output_axis=0, requires_grad=parameter.requires_grad,
                          parameter_kind='weight' if name == 'weight' else name,
                          l1=(x.abs().sum(0)).cpu().tolist(), l2=(x.square().sum(0).sqrt()).cpu().tolist())
            results.append(record)
    return results


class ReadOnlyMonitor:
    """Install hooks only inside sample(). Stream JSON-safe records; retain no activations/history."""
    def __init__(self, model, specs, sink, max_calls=64):
        self.model, self.specs, self.sink = model, tuple(specs), sink
        if not self.specs or len({s.path for s in self.specs}) != len(self.specs):
            raise ValueError('Nonempty, unique layer paths required')
        if not callable(sink) or type(max_calls) is not int or max_calls <= 0:
            raise ValueError('Callable sink and positive call bound required')
        modules = dict(model.named_modules())
        if any(s.path not in modules for s in self.specs): raise ValueError('Unknown module path')
        self.max_calls, self.emitted = max_calls, 0

    @contextmanager
    def sample(self, context, masks=None):
        """One synchronous forward block; backward/checkpoint replay must run outside this context."""
        if not REQUIRED_CONTEXT <= context.keys() or context['partition'] not in ('train', 'val'):
            raise ValueError('Full identity context and train/val partition required; test selection is excluded')
        if type(context['step']) is not int or context['step'] < 0: raise ValueError('Actual update index required')
        if any(not isinstance(context[k], str) or not context[k] for k in REQUIRED_CONTEXT-{'step'}):
            raise ValueError('Identity fields must be nonempty strings')
        # Freeze caller metadata and reject NaN / unserializable identities before hook registration.
        context = json.loads(json.dumps(context, allow_nan=False))
        if any(getattr(m, 'checkpoint_chunks', False) for m in self.model.modules()):
            raise ValueError('Checkpoint recomputation is not supported; use an explicitly noncheckpointed path')
        masks = {} if masks is None else dict(masks)
        if set(masks)-{s.path for s in self.specs}: raise ValueError('Unknown mask path')
        with _LOCK:
            if self.model in _ACTIVE: raise RuntimeError('Overlapping monitor on this model')
            _ACTIVE.add(self.model)
        handles = []; counts = {}; total_calls = 0
        def hook(spec):
            def measure(module, args, output):
                nonlocal total_calls
                total_calls += 1
                if total_calls > self.max_calls: raise RuntimeError('Sample call bound exceeded')
                selected = output
                for key in spec.selector: selected = selected[key]
                if not isinstance(selected, torch.Tensor): raise ValueError('Selector must resolve to one tensor')
                rank, axis = LAYOUTS[spec.layout]
                if selected.ndim != rank: raise ValueError('Selected tensor rank/layout mismatch')
                stats = ChannelStats(selected.shape[axis], spec.epsilon)
                stats.update(selected, spec.layout, masks.get(spec.path))
                call = counts.get(spec.path, 0); counts[spec.path] = call+1
                record = {'schema_version': 1, 'kind': 'activation', 'context': context,
                          'layer': spec.path, 'call_index': call, 'layout': spec.layout,
                          'position': spec.position, 'unit': spec.unit, 'dtype': str(selected.dtype),
                          'model_mode': 'train' if self.model.training else 'eval', 'statistics': stats.report()}
                # No tensor, computation graph or mutable internal record escapes the collector.
                self.sink(json.loads(json.dumps(record, allow_nan=False))); self.emitted += 1
            return measure
        try:
            for spec in self.specs: handles.append(self.model.get_submodule(spec.path).register_forward_hook(hook(spec)))
            yield self
            missing = [s.path for s in self.specs if counts.get(s.path, 0) == 0]
            if missing: raise RuntimeError('Selected modules were not called: ' + ', '.join(missing))
        finally:
            for handle in handles: handle.remove()
            with _LOCK: _ACTIVE.discard(self.model)


