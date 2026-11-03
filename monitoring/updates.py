"""Explicit, sampled FP32/FP64 gradient and single optimizer-update telemetry.

Independent implementation of the RM-inspired MON-2 measurement contract.
No sampler, parameter ranking, optimizer policy or training default is installed.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import torch

from .readonly import LAYOUTS, REQUIRED_CONTEXT, _ACTIVE, _LOCK


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    axis: int
    unit: str

    def __post_init__(self):
        if not self.name or type(self.axis) is not int or not self.unit:
            raise ValueError('Explicit parameter name, channel axis and unit required')


def _rows(tensor, axis):
    if tensor.layout != torch.strided or tensor.dtype not in (torch.float32, torch.float64):
        raise ValueError('Only dense FP32/FP64 measurements are supported')
    if not torch.isfinite(tensor).all():
        raise ValueError('Nonfinite measurement rejected')
    return tensor.detach().movedim(axis, 0).reshape(tensor.shape[axis], -1).double()


def _norms(tensor, axis):
    rows = _rows(tensor, axis)
    return rows.square().sum(1).sqrt(), rows.square().mean(1).sqrt()


class UpdateMonitor:
    """Wrap one existing eager step, without replacing forward/backward/step.

    sample(..., contract='fp32_single_backward') is the caller's declaration:
    one unscaled backward, no accumulation/AMP/closure/checkpoint recomputation.
    Undeclared scaling cannot be inferred from numerical gradients. Errors are
    fail-fast, not transactional: a completed optimizer step is never rolled back.
    """
    def __init__(self, model, optimizer, parameters, sink, *, activations=(),
                 epsilon=1e-12, max_snapshot_bytes=16*1024*1024, max_calls=32):
        self.model, self.optimizer = model, optimizer
        self.specs, self.activations, self.sink = tuple(parameters), tuple(activations), sink
        if not self.specs or len({s.name for s in self.specs}) != len(self.specs):
            raise ValueError('Nonempty unique parameter names required')
        if len({s.path for s in self.activations}) != len(self.activations):
            raise ValueError('Unique activation paths required')
        if not callable(sink) or type(max_calls) is not int or max_calls <= 0:
            raise ValueError('Callable sink and positive call bound required')
        if type(max_snapshot_bytes) is not int or max_snapshot_bytes <= 0:
            raise ValueError('Positive snapshot byte budget required')
        if isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('Positive finite relative-update epsilon required')
        self.epsilon, self.max_bytes, self.max_calls = epsilon, max_snapshot_bytes, max_calls
        self.emitted = 0

    @contextmanager
    def sample(self, context, *, contract, enabled=True, masks=None):
        if not enabled:
            yield self
            return
        if contract != 'fp32_single_backward':
            raise ValueError('AMP, accumulation and closure contracts are not supported')
        if not REQUIRED_CONTEXT <= context.keys() or context['partition'] != 'train':
            raise ValueError('Full train-only identity context required')
        if type(context['step']) is not int or context['step'] < 0:
            raise ValueError('Nonnegative actual update index required')
        if any(not isinstance(context[k], str) or not context[k] for k in REQUIRED_CONTEXT-{'step'}):
            raise ValueError('Nonempty identity strings required')
        context = json.loads(json.dumps(context, allow_nan=False))
        opt = self.optimizer
        if type(opt) not in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
            raise ValueError('Only ordinary SGD/Adam/AdamW steps are supported')
        if opt._optimizer_step_pre_hooks or opt._optimizer_step_post_hooks:
            raise ValueError('Existing optimizer hooks make measurement order ambiguous')
        if any(g.get('differentiable', False) or g.get('capturable', False) for g in opt.param_groups):
            raise ValueError('Differentiable/captured steps are not supported')
        if any(getattr(m, 'checkpoint_chunks', False) for m in self.model.modules()):
            raise ValueError('Checkpoint recomputation is not supported')
        all_params = dict(self.model.named_parameters())
        selected = {}
        storage = set()
        for spec in self.specs:
            if spec.name not in all_params:
                raise ValueError('Unknown parameter: '+spec.name)
            param = all_params[spec.name]
            if not 0 <= spec.axis < param.ndim or param.numel() == 0:
                raise ValueError('Explicit valid nonempty channel axis required')
            _rows(param, spec.axis)
            key = (param.device, param.untyped_storage().data_ptr())
            if key in storage:
                raise ValueError('Selected parameters must not share storage')
            storage.add(key)
            selected[spec.name] = param
        snapshot_bytes = sum(p.numel()*p.element_size() for p in selected.values())
        if snapshot_bytes > self.max_bytes:
            raise ValueError('Snapshot budget exceeded')
        masks = {} if masks is None else dict(masks)
        if set(masks)-{s.path for s in self.activations}:
            raise ValueError('Unknown activation mask path')
        if any(not isinstance(m, torch.Tensor) or m.dtype != torch.bool for m in masks.values()):
            raise ValueError('Boolean activation masks required')
        mask_bytes = sum(m.numel()*m.element_size() for m in masks.values())
        if mask_bytes > self.max_bytes:
            raise ValueError('Mask snapshot budget exceeded')
        masks = {k:v.detach().clone() for k,v in masks.items()}
        modules = dict(self.model.named_modules())
        if any(s.path not in modules for s in self.activations):
            raise ValueError('Unknown activation module')
        with _LOCK:
            if self.model in _ACTIVE:
                raise RuntimeError('Overlapping monitor on this model')
            _ACTIVE.add(self.model)
        handles, activation_records, records, before = [], [], [], {}
        state = {'pre':0, 'post':0}

        def pre(optimizer, args, kwargs):
            if state['pre'] or state['post']:
                raise RuntimeError('Exactly one optimizer step per sample')
            if len(args) > 1 or kwargs:
                raise ValueError('Optimizer step arguments/closures are unsupported')
            if (torch.is_autocast_enabled() or torch.is_autocast_enabled('cpu')
                    or hasattr(optimizer, 'grad_scale') or hasattr(optimizer, 'found_inf')):
                raise ValueError('AMP/scaled steps are unsupported')
            group_ids = {id(p):i for i,g in enumerate(optimizer.param_groups) for p in g['params']}
            for spec in self.specs:
                p = selected[spec.name]
                if all_params[spec.name] is not dict(self.model.named_parameters()).get(spec.name):
                    raise ValueError('Parameter identity changed during sample')
                before[spec.name] = p.detach().clone()
                norm, rms = _norms(p, spec.axis)
                grad = p.grad
                grad_state = 'frozen' if not p.requires_grad else 'missing' if grad is None else 'zero' if not torch.count_nonzero(grad) else 'present'
                grad_rms = None if grad is None else _norms(grad, spec.axis)[1].cpu().tolist()
                group = group_ids.get(id(p))
                records.append({'parameter':spec.name,'axis':spec.axis,'unit':spec.unit,
                    'shape':list(p.shape),'dtype':str(p.dtype),'requires_grad':p.requires_grad,
                    'optimizer_group':group,'learning_rate':None if group is None else float(optimizer.param_groups[group]['lr']),
                    'gradient_state':grad_state,'gradient_present':grad is not None,'gradient_rms':grad_rms,
                    'weight_l2_before':norm.cpu().tolist(),'weight_rms_before':rms.cpu().tolist()})
            state['pre'] = 1

        def post(optimizer, args, kwargs):
            if state != {'pre':1,'post':0}:
                raise RuntimeError('Invalid optimizer hook order')
            for spec, record in zip(self.specs, records):
                delta = selected[spec.name].detach()-before[spec.name]
                norm, rms = _norms(delta, spec.axis)
                old_norm = _norms(before[spec.name], spec.axis)[0]
                record.update(update_l2=norm.cpu().tolist(), update_rms=rms.cpu().tolist(),
                    relative_update=(norm/(old_norm+self.epsilon)).cpu().tolist(),
                    changed_elements=int(torch.count_nonzero(delta)),
                    update_state='zero' if not torch.count_nonzero(delta) else 'changed')
            before.clear()
            state['post'] = 1

        def forward_hook(spec):
            def capture(module, args, output):
                if state['pre']:
                    raise RuntimeError('Forward after the sampled optimizer step is unsupported')
                if len(activation_records) >= self.max_calls:
                    raise RuntimeError('Activation call budget exceeded')
                value = output
                for key in spec.selector:
                    value = value[key]
                rank, axis = LAYOUTS[spec.layout]
                if not isinstance(value, torch.Tensor) or value.ndim != rank:
                    raise ValueError('Activation selector/layout mismatch')
                if spec.layout == 'NCHW' and value.shape[0] != 1:
                    raise ValueError('Activation NCHW supports one block')
                _rows(value, axis)
                mask = masks.get(spec.path)
                expected = tuple(d for i,d in enumerate(value.shape) if i != axis)
                if mask is not None and tuple(mask.shape) != expected:
                    raise ValueError('Mask shape must match nonchannel dimensions')
                row = {'layer':spec.path,'call_index':sum(r['layer']==spec.path for r in activation_records),
                    'layout':spec.layout,'position':spec.position,'unit':spec.unit,
                    'gradient_state':'missing' if value.requires_grad else 'detached',
                    'gradient_rms':None,'count_per_channel':None,'backward_calls':0}
                activation_records.append(row)
                if value.requires_grad:
                    def gradient_hook(grad):
                        if row['backward_calls'] or state['pre']:
                            raise RuntimeError('Repeated/late backward unsupported')
                        x = grad.detach().movedim(axis,-1)
                        if mask is not None:
                            x = x[mask.to(x.device)]
                        x = x.reshape(-1,grad.shape[axis])
                        _rows(grad,axis)
                        row.update(backward_calls=1,count_per_channel=len(x),
                            gradient_state='empty' if not len(x) else 'zero' if not torch.count_nonzero(x) else 'present',
                            gradient_rms=x.double().square().mean(0).sqrt().cpu().tolist() if len(x) else [None]*grad.shape[axis])
                        # Returning None preserves the original gradient.
                    handles.append(value.register_hook(gradient_hook))
            return capture
        try:
            handles.append(opt.register_step_pre_hook(pre))
            handles.append(opt.register_step_post_hook(post))
            for spec in self.activations:
                handles.append(modules[spec.path].register_forward_hook(forward_hook(spec)))
            yield self
            if state != {'pre':1,'post':1}:
                raise RuntimeError('Sample did not complete one optimizer step')
            missing = {s.path for s in self.activations}-{r['layer'] for r in activation_records}
            if missing:
                raise RuntimeError('Selected activation modules not called: '+','.join(sorted(missing)))
            report = {'schema_version':1,'kind':'optimizer_update','context':context,'contract':contract,
                'epsilon':self.epsilon,'snapshot_bytes':snapshot_bytes,'mask_snapshot_bytes':mask_bytes,
                'optimizer_steps':1,'parameters':records,'activation_gradients':activation_records}
            self.sink(json.loads(json.dumps(report,allow_nan=False)))
            self.emitted += 1
        finally:
            for handle in handles:
                handle.remove()
            before.clear()
            with _LOCK:
                _ACTIVE.discard(self.model)
