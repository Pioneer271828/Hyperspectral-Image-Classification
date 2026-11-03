"""Synchronous isolated eval probes and explicit output-channel zeroing.

Trusted factory/forward callbacks run in an isolated diagnostic process. No live
optimizer is accepted. Factory models must be fresh, independently stored copies
of the declared reference state; arbitrary callback side effects are not sandboxed.
"""
from dataclasses import dataclass
import random
import math
import numpy as np
import torch
from joint_training import _tree_fingerprint
from .probes import FrozenProbe, _input_hashes
from .readonly import LAYOUTS


@dataclass(frozen=True)
class ChannelIntervention:
    observation: str
    channels: tuple

    def __post_init__(self):
        if (not self.observation or not isinstance(self.channels,tuple) or not self.channels
                or any(type(i)!=int or i<0 for i in self.channels)
                or len(set(self.channels))!=len(self.channels)):
            raise ValueError('Explicit observation and nonempty unique channel indices required')


def _snapshot(model):
    return {'state':_tree_fingerprint(model.state_dict()),
            'gradients':_tree_fingerprint({n:p.grad for n,p in model.named_parameters()}),
            'modes':{n:m.training for n,m in model.named_modules()},
            'metadata':_tree_fingerprint(getattr(model,'training_config',{}))}


def _storage(model):
    result=set()
    for x in list(model.parameters())+list(model.buffers()):
        if x.layout!=torch.strided:raise ValueError('Dense model state required')
        if x.numel():result.add((str(x.device),x.untyped_storage().data_ptr()))
    return result


def _tensor_bytes(value):
    if value.layout==torch.strided:return value.numel()*value.element_size()
    if value.layout==torch.sparse_coo:
        return value._values().numel()*value.element_size()+value._indices().numel()*value._indices().element_size()
    raise ValueError('Only dense and COO input/output tensors supported')


def _copy_tree(value, budget, *, cpu=False):
    if torch.is_tensor(value):
        budget[0]-=_tensor_bytes(value)
        if budget[0]<0:raise ValueError('Tensor copy budget exceeded')
        x=value.detach()
        values=x if x.layout==torch.strided else x._values()
        if not torch.isfinite(values).all():raise ValueError('Nonfinite probe tensor')
        return x.cpu().clone() if cpu else x.clone()
    if isinstance(value,dict):return {k:_copy_tree(v,budget,cpu=cpu) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(_copy_tree(v,budget,cpu=cpu) for v in value)
    if isinstance(value,list):return [_copy_tree(v,budget,cpu=cpu) for v in value]
    if type(value)==float and not math.isfinite(value):raise ValueError('Nonfinite output scalar')
    if value is None or type(value) in (str,int,float,bool):return value
    raise ValueError('Only plain tensor trees supported')


def isolated_probe(reference, factory, forward, inputs, plan, *, interventions=(),
                   max_input_bytes=1024**3, max_output_bytes=128*1024**2,
                   max_activation_bytes=1024**3):
    """Return frozen conditional statistics and detached outputs, never updated weights.

    Plan/channel candidates must be fixed before results are read. Observed module
    outputs are tensors only (no selector), and each declared call must occur.
    A fresh factory is invoked on every call, including the no-intervention control.
    This is not a deterministic CUDA policy or a generic untrusted-code sandbox.
    """
    if not isinstance(plan,FrozenProbe) or not callable(factory) or not callable(forward):
        raise ValueError('Frozen probe and trusted factory/forward callbacks required')
    for bound in (max_input_bytes,max_output_bytes,max_activation_bytes):
        if type(bound)!=int or bound<=0:raise ValueError('Positive copy budgets required')
    manifest=plan.manifest
    if _input_hashes(inputs)!=manifest['input_hashes']:
        raise ValueError('Frozen input identity differs')
    interventions=tuple(interventions)
    if len({i.observation for i in interventions})!=len(interventions):
        raise ValueError('Combine channel groups explicitly per observation')
    targets={i.observation:i.channels for i in interventions}
    if set(targets)-set(manifest['calls']):raise ValueError('Unknown intervention observation')
    if any(c['spec']['selector'] for c in manifest['calls'].values()):
        raise ValueError('Tensor-only observations; nested output selectors unsupported')
    initial=_snapshot(reference);reference_storage=_storage(reference)
    reference_modules={id(m) for m in reference.modules()}
    py_rng=random.getstate();np_rng=np.random.get_state();cpu_rng=torch.get_rng_state()
    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    handles=[];captured={};used=set();model=None
    try:
        cloned_inputs=_copy_tree(inputs,[max_input_bytes])
        model=factory()
        if not isinstance(model,torch.nn.Module) or reference_modules & {id(m) for m in model.modules()}:
            raise ValueError('Factory must return a fresh independent model')
        if _storage(model)&reference_storage:raise ValueError('Factory state shares storage with reference')
        if ([(n,str(p.device)) for n,p in model.named_parameters()]!=[(n,str(p.device)) for n,p in reference.named_parameters()]
                or [(n,str(p.device)) for n,p in model.named_buffers()]!=[(n,str(p.device)) for n,p in reference.named_buffers()]):
            raise ValueError('Factory model device differs')
        if [(n,type(m)) for n,m in model.named_modules()]!=[(n,type(m)) for n,m in reference.named_modules()]:
            raise ValueError('Factory structure differs')
        copied=_snapshot(model)
        if copied['state']!=initial['state'] or copied['metadata']!=initial['metadata']:
            raise ValueError('Factory state/checkpoint metadata differs')
        if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in model.modules()):
            raise ValueError('Existing model hooks unsupported')
        model.eval();model.zero_grad(set_to_none=True)
        before=_snapshot(model);modules=dict(model.named_modules());by_path={};counts={};retained=[0]
        for key,c in manifest['calls'].items():
            path=c['spec']['path']
            if path not in modules:raise ValueError('Observation module missing')
            by_path.setdefault(path,{})[c['call_index']]=key;counts[path]=0
        def hook(path):
            def observe(module,args,output):
                call=counts[path];counts[path]+=1
                if call not in by_path[path]:return None
                key=by_path[path][call];spec=manifest['calls'][key]['spec'];rank,axis=LAYOUTS[spec['layout']]
                if not torch.is_tensor(output) or output.ndim!=rank:
                    raise ValueError('Declared tensor output/layout mismatch')
                retained[0]+=_tensor_bytes(output)
                if retained[0]>max_activation_bytes:raise ValueError('Retained activation budget exceeded')
                value=output
                if key in targets:
                    channels=targets[key]
                    if max(channels)>=value.shape[axis]:raise ValueError('Intervention channel out of range')
                    value=output.clone();value.index_fill_(axis,torch.tensor(channels,device=value.device),0)
                    used.add(key)
                # Freeze this call before downstream in-place operations can
                # modify either the original output or the intervention result.
                captured[key]=value.detach().clone()
                return value if key in targets else None
            return observe
        for path in by_path:handles.append(modules[path].register_forward_hook(hook(path)))
        with torch.no_grad():outputs=forward(model,cloned_inputs)
        if used!=set(targets):raise ValueError('Intervention call not executed')
        if _input_hashes(cloned_inputs)!=manifest['input_hashes']:
            raise ValueError('Probe forward mutated cloned inputs')
        statistics=plan.observe(cloned_inputs,captured,context=manifest['context'])
        if _snapshot(model)!=before:raise ValueError('Probe forward changed model state/modes/gradients')
        detached=_copy_tree(outputs,[max_output_bytes],cpu=True)
        return {'outputs':detached,'statistics':statistics,
                'audit':{'reference_state':initial['state'],'probe_identity':plan.identity,
                         'interventions':[{'observation':i.observation,'channels':list(i.channels)} for i in interventions],
                         'optimizer_updates':0,'execution':'isolated_eval_no_grad',
                         'declared_calls_executed':sorted(captured),'state_unchanged':True}}
    finally:
        for h in handles:h.remove()
        captured.clear()
        random.setstate(py_rng);np.random.set_state(np_rng);torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        if _snapshot(reference)!=initial or _input_hashes(inputs)!=manifest['input_hashes']:
            raise RuntimeError('Trusted callback mutated caller state/input; no automatic rollback')
