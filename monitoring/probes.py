"""Frozen, train/validation-only conditional observations; no model runner or pruning.

The caller supplies activations from the declared checkpoint in an isolated probe
process. This layer authenticates the input/measurement contract, not arbitrary
caller computation. No hooks, optimizer, RNG or automatic channel ranking exist.
"""
from dataclasses import asdict
import json
import torch
from model.research_config import digest
from research_runtime import tensor_hash
from .readonly import ChannelStats, LAYOUTS, REQUIRED_CONTEXT


def class_masks(labels, eligible, classes, *, unit):
    """Pixel labels cannot be silently used as region identities; zero is unlabelled."""
    if unit != 'pixel':
        raise ValueError('Pixel class masks require pixel units; declare region mapping separately')
    if (not isinstance(labels, torch.Tensor) or labels.dtype not in (torch.int16, torch.int32, torch.int64)
            or not isinstance(eligible, torch.Tensor) or eligible.dtype != torch.bool
            or labels.shape != eligible.shape or labels.device != eligible.device):
        raise ValueError('Integer labels and matching boolean eligibility required')
    classes=tuple(classes)
    if not classes or len(set(classes))!=len(classes) or any(type(c)!=int or c<=0 for c in classes):
        raise ValueError('Unique positive class IDs required; unlabelled zero is not background')
    if (labels < 0).any() or not set(labels[eligible].unique().tolist()) <= {0,*classes}:
        raise ValueError('Eligible labels outside declared class space')
    return {str(c):(eligible & (labels==c)).detach().clone() for c in classes}


def _input_hashes(inputs):
    if not inputs or any(not isinstance(k,str) or not k or not isinstance(v,torch.Tensor)
                         for k,v in inputs.items()):
        raise ValueError('Nonempty named tensor inputs required')
    return {k:tensor_hash(v) for k,v in sorted(inputs.items())}


class FrozenProbe:
    """One immutable case, explicit layer calls and fixed conditional masks.

    observation keys name (layer path, call index) pairs through the calls mapping.
    Each observation is one declared tensor call; each bucket uses its own
    element count rather than a mean of percentages from other buckets.
    No aggregation across plans, layouts, datasets or checkpoint identities occurs.
    """
    def __init__(self, context, inputs, calls, buckets, *, threshold_source,
                 allowed_datasets, max_mask_bytes=16*1024*1024, max_calls=64):
        if not REQUIRED_CONTEXT <= context.keys() or context['partition'] not in ('train','val'):
            raise ValueError('Complete train/val identity required; test probes are forbidden')
        if type(context['step'])!=int or context['step']<0:
            raise ValueError('Nonnegative step required')
        if any(not isinstance(context[k],str) or not context[k] for k in REQUIRED_CONTEXT-{'step'}):
            raise ValueError('Nonempty identity strings required')
        if not isinstance(allowed_datasets,(tuple,list)):
            raise ValueError('Explicit permitted dataset sequence required')
        allowed=tuple(allowed_datasets)
        if not allowed or context['dataset'] not in allowed or any(not isinstance(n,str) or not n for n in allowed):
            raise ValueError('Dataset outside frozen permitted domain set')
        if not isinstance(threshold_source,str) or not threshold_source:
            raise ValueError('Explicit pre-frozen threshold provenance required')
        if type(max_calls)!=int or max_calls<=0 or not calls or len(calls)>max_calls:
            raise ValueError('Nonempty bounded observation calls required')
        if type(max_mask_bytes)!=int or max_mask_bytes<=0:
            raise ValueError('Positive mask budget required')
        if set(calls)!=set(buckets):
            raise ValueError('Every declared call needs exactly one bucket definition')
        self._specs={};self._masks={};definitions={};total=0;identities=set()
        for key,(spec,call_index) in calls.items():
            if not isinstance(key,str) or not key or type(call_index)!=int or call_index<0:
                raise ValueError('Explicit observation key and nonnegative call index required')
            identity=(spec.path,call_index)
            if identity in identities:raise ValueError('Duplicate layer/call identity')
            identities.add(identity)
            if spec.unit not in ('pixel','region','latent_position'):
                raise ValueError('Explicit pixel/region/latent_position unit required')
            if not buckets[key]:raise ValueError('Nonempty condition bucket definitions required')
            self._specs[key]=spec;self._masks[key]={};hashes={}
            for name,mask in buckets[key].items():
                if not isinstance(name,str) or not name or not isinstance(mask,torch.Tensor) or mask.dtype!=torch.bool:
                    raise ValueError('Named boolean condition masks required')
                total+=mask.numel()*mask.element_size()
                if total>max_mask_bytes:raise ValueError('Mask budget exceeded')
                self._masks[key][name]=mask.detach().clone()
                hashes[name]={'hash':tensor_hash(mask),'shape':list(mask.shape),'count':int(mask.sum())}
            definitions[key]={'spec':asdict(spec),'call_index':call_index,'buckets':hashes}
        manifest={'version':1,'context':context,'input_hashes':_input_hashes(inputs),
                  'calls':definitions,'threshold_source':threshold_source,
                  'allowed_datasets':sorted(set(allowed)),'mask_bytes':total,
                  'scope':'caller-supplied activations; isolated model execution separately required'}
        self._serialized=json.dumps(manifest,sort_keys=True,allow_nan=False)
        self.identity=digest(json.loads(self._serialized))

    @property
    def manifest(self):
        return json.loads(self._serialized)

    def observe(self, inputs, activations, *, context):
        plan=self.manifest
        if digest(plan)!=self.identity or context!=plan['context']:
            raise ValueError('Frozen probe identity/context changed')
        if _input_hashes(inputs)!=plan['input_hashes']:
            raise ValueError('Frozen probe inputs changed')
        if set(activations)!=set(self._specs):
            raise ValueError('Missing/extra declared activation calls')
        records=[]
        for key,spec in self._specs.items():
            if digest(asdict(spec))!=digest(plan['calls'][key]['spec']):
                raise ValueError('Frozen measurement threshold/layout/unit changed')
            value=activations[key];rank,axis=LAYOUTS[spec.layout]
            if (not isinstance(value,torch.Tensor) or value.ndim!=rank
                    or value.shape[axis]<=0 or not value.is_floating_point()
                    or not torch.isfinite(value).all()):
                raise ValueError('Finite declared activation tensor required')
            for name,mask in self._masks[key].items():
                if tensor_hash(mask)!=plan['calls'][key]['buckets'][name]['hash']:
                    raise ValueError('Frozen condition mask changed')
                stats=ChannelStats(value.shape[axis],spec.epsilon)
                stats.update(value,spec.layout,mask)
                records.append({'observation':key,'layer':spec.path,
                    'call_index':plan['calls'][key]['call_index'],'unit':spec.unit,
                    'layout':spec.layout,'position':spec.position,'bucket':name,
                    'statistics':stats.report()})
        return {'version':1,'probe_identity':self.identity,'context':plan['context'],
                'threshold_source':plan['threshold_source'],'records':records,
                'automatic_pruning':False,'model_execution_verified':False}
