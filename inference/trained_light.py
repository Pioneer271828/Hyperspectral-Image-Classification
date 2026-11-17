"""Authenticated, selected head-only light artifact; explicit trained cascade."""
from copy import deepcopy
import hashlib
import io
import os
from pathlib import Path
import tempfile
import torch
from inference.light_cnn import LightCNN, validate_source
from inference.tile_cascade import _execute
from model.research_config import digest
from transfer.g3_training import state_hash
from transfer.light_head_training import LightHeadRun


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _validate(full,light,spec):
    validate_source(full)
    keys={'method','training_spec','training_sha256','contract_hash','history_hash','completed',
          'selected_step','selected_validation','selected_head_state','light_state'}
    if not isinstance(spec,dict) or set(spec)!=keys or spec['method']!='selected_head_only_light_v1':
        raise ValueError('Invalid trained-light specification')
    train=spec['training_spec']
    if (not isinstance(train,dict) or digest(train)!=spec['contract_hash']
            or any(not _sha(spec[k]) for k in ('training_sha256','contract_hash','history_hash','selected_head_state','light_state'))
            or train.get('method')!='independent_frozen_prefix_head_only'
            or train.get('selection')!='first_min_full_devval_CNN_NLL_each_update'
            or train.get('source_state')!=state_hash(full) or train.get('architecture')!=full.export_config()
            or type(spec['completed']) is not int or spec['completed']!=train.get('updates')
            or type(spec['selected_step']) is not int or not 1<=spec['selected_step']<=spec['completed']):
        raise ValueError('Invalid source, training contract or completed selection')
    dataset=train['identity']['dataset']
    if (type(light) is not LightCNN or dataset not in light.heads or light.architecture!=full.export_config()
            or light.source_state!=train['source_state'] or any(m.training for m in light.modules())
            or any(p.requires_grad for p in light.parameters()) or light.adapter.cache_attention
            or state_hash(light)!=spec['light_state'] or state_hash(light.heads[dataset])!=spec['selected_head_state']):
        raise ValueError('Frozen selected light state mismatch')
    source=full.state_dict();prefixes={'adapter':'backbone.adapter','denoise':'backbone.CNN_denoise','cnn':'backbone.CNN_Branch','heads':'cnn_heads'}
    for key,v in light.state_dict().items():
        if v.device.type!='cpu' or not torch.isfinite(v).all() or (v.is_floating_point() and v.dtype!=torch.float32):
            raise ValueError('Finite CPU FP32 light required')
        if key.startswith('heads.'+dataset+'.'):continue
        prefix,suffix=key.split('.',1)
        if not torch.equal(v,source[prefixes[prefix]+'.'+suffix]):raise ValueError('Frozen prefix or another task head changed')
    return dataset


class TrainedLight:
    """Owned frozen light plus provenance; SHA must come from trusted registration."""
    def __init__(self,full,light,spec):
        _validate(full,light,spec)
        self.model=deepcopy(light);self._spec=deepcopy(spec);self._spec_hash=digest(spec)

    @property
    def specification(self):return deepcopy(self._spec)

    def bind_fallback(self, source, fallback, *, expected_fallback_state):
        """Keep the trained prefix bound to its source while pinning a matched full route."""
        self.validate(source)
        validate_source(fallback)
        if state_hash(fallback) != expected_fallback_state or fallback.export_config() != source.export_config():
            raise ValueError("Matched full fallback differs from its frozen state or source architecture")
        self._training_source = source
        self._fallback_state = expected_fallback_state
        return self

    def validate(self,full):
        if digest(self._spec)!=self._spec_hash:raise ValueError('Trained-light provenance changed')
        if not hasattr(self, "_training_source"):
            return _validate(full,self.model,self._spec)
        validate_source(full)
        if state_hash(full) != self._fallback_state or full.export_config() != self._training_source.export_config():
            raise ValueError("Matched full fallback state or architecture changed")
        return _validate(self._training_source,self.model,self._spec)


def export_trained_light(full,scene,record,checkpoint,path,*,expected_training_sha256,expected_contract_hash):
    """Reauthenticate a completed run and recheck selected fixed-validation output."""
    run=LightHeadRun.load(full,scene,record,checkpoint,expected_sha256=expected_training_sha256,
                         expected_contract_hash=expected_contract_hash)
    if run.completed!=run.spec['updates']:raise ValueError('Cannot publish incomplete training')
    light=deepcopy(run.path);light.heads[scene.dataset].load_state_dict(run.selected_head,strict=True)
    light.requires_grad_(False);light.eval()
    spec=dict(method='selected_head_only_light_v1',training_spec=deepcopy(run.spec),training_sha256=expected_training_sha256,
              contract_hash=expected_contract_hash,history_hash=digest(run.history),completed=run.completed,
              selected_step=run.selected_step,selected_validation=deepcopy(run.history[run.selected_step-1]['validation']),
              selected_head_state=state_hash(light.heads[scene.dataset]),light_state=state_hash(light))
    _validate(full,light,spec)
    packet=dict(format='vca-trained-light-cpu-v1',spec=spec,spec_hash=digest(spec),state=deepcopy(light.state_dict()))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
            temp=Path(f.name);torch.save(packet,f);f.flush();os.fsync(f.fileno())
        os.link(temp,path)
    finally:
        if temp is not None:temp.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trained_light(full,path,*,expected_sha256,expected_training_sha256,expected_contract_hash):
    """No labels at inference; external artifact/run/contract pins are all required."""
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Trained-light file hash mismatch')
    packet=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if (not isinstance(packet,dict) or set(packet)!={'format','spec','spec_hash','state'}
            or packet['format']!='vca-trained-light-cpu-v1'):raise ValueError('Invalid trained-light packet')
    spec=packet['spec']
    if (not isinstance(spec,dict) or digest(spec)!=packet['spec_hash'] or spec.get('training_sha256')!=expected_training_sha256
            or spec.get('contract_hash')!=expected_contract_hash):raise ValueError('Wrong registered training provenance')
    light=LightCNN(full);ref=light.state_dict();state=packet['state']
    if (not isinstance(state,dict) or set(state)!=set(ref) or any(not torch.is_tensor(v) or v.shape!=ref[k].shape
            or v.dtype!=ref[k].dtype or not torch.isfinite(v).all() for k,v in state.items())):raise ValueError('Invalid trained-light state schema')
    light.load_state_dict(state,strict=True)
    return TrainedLight(full,light,spec)


def trained_cascade(full,published,requests,*,mode='full',criterion=None,threshold=None,min_coverage=1.0):
    if type(published) is not TrainedLight:raise ValueError('Authenticated trained-light pair required')
    dataset=published.validate(full)
    if not isinstance(requests,(list,tuple)) or any(not isinstance(row,dict) or row.get('dataset')!=dataset for row in requests):
        raise ValueError('Only the registered trained task is supported')
    result=_execute(full,published.model,requests,mode=mode,criterion=criterion,threshold=threshold,min_coverage=min_coverage)
    result['training_contract']=published.specification['contract_hash']
    result['selected_step']=published.specification['selected_step']
    return result
