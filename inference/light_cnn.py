"""Independent frozen CNN path copied from an uncoupled CPU VCA model."""
import copy
import hashlib
import io
import os
from pathlib import Path
import tempfile
import torch
from torch import nn
from torch.nn import functional as F
from compression.int8_model import _dense
from compression.int8_checkpoint import _check_state
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import digest
from transfer.g3_training import state_hash


def validate_source(model):
    _dense(model)
    b=model.backbone
    if (any(m.training for m in model.modules()) or b.graph_mode!='learned' or b.adapter.cache_attention
            or any(getattr(b,k) is not None for k in ('interaction','region_residual','coverage_modulation','graph_router','edge_updater'))):
        raise ValueError('Frozen uncoupled learned-graph CPU source with attention cache disabled required')
    if any(type(m.point_conv) is not nn.Conv2d for m in b.CNN_Branch):
        raise ValueError('Original float CNN layers required')


class LightCNN(nn.Module):
    def __init__(self,source):
        super().__init__();validate_source(source);b=source.backbone
        self.architecture=copy.deepcopy(source.export_config());self.source_state=state_hash(source)
        self.extra_channels=b.extra_channels
        self.adapter=copy.deepcopy(b.adapter);self.adapter.cache_attention=False;self.adapter.clear_attention_cache()
        self.denoise=copy.deepcopy(b.CNN_denoise);self.cnn=copy.deepcopy(b.CNN_Branch)
        self.heads=copy.deepcopy(source.cnn_heads)
        self.requires_grad_(False);self.eval()

    def frozen_features(self,dataset,x,*,wavelengths):
        """Detached CNN features; preserve eval normalization and never build a graph."""
        if any(m.training for m in self.modules()):raise ValueError('LightCNN is inference-only')
        if dataset not in self.heads:raise ValueError('Unknown task')
        if (not torch.is_tensor(x) or x.device.type!='cpu' or x.dtype!=torch.float32 or x.requires_grad
                or x.ndim!=3 or min(x.shape[:2])<1 or not torch.isfinite(x).all()):raise ValueError('Finite CPU FP32 block required')
        if (not torch.is_tensor(wavelengths) or wavelengths.device.type!='cpu' or wavelengths.dtype!=torch.float32
                or wavelengths.requires_grad or wavelengths.ndim!=1 or wavelengths.numel()<1
                or not torch.isfinite(wavelengths).all() or (wavelengths<=0).any()
                or x.shape[2]!=wavelengths.numel()+self.extra_channels):raise ValueError('Explicit finite wavelengths required')
        h,w,_=x.shape;bands=wavelengths.numel()
        with torch.no_grad():
            latent=self.adapter(x[:,:,:bands],wavelengths=wavelengths)
            noise=torch.cat((latent,x[:,:,bands:]),2).permute(2,0,1).unsqueeze(0)
            for block in self.denoise:noise=block(noise,dataset)
            clean=noise.squeeze(0).permute(1,2,0)
            stage1=self.cnn[0](clean.permute(2,0,1).unsqueeze(0),dataset)
            pixels=stage1.squeeze(0).permute(1,2,0).reshape(h*w,128)
            stage2=self.cnn[1](pixels.reshape(h,w,128).permute(2,0,1).unsqueeze(0),dataset)
            features=stage2.squeeze(0).permute(1,2,0).reshape(h*w,64)
        if not torch.isfinite(features).all():raise ValueError('Nonfinite light features')
        return features

    def forward(self,dataset,x,*,wavelengths):
        features=self.frozen_features(dataset,x,wavelengths=wavelengths)
        with torch.no_grad():result=F.softmax(self.heads[dataset](features),-1)
        if not torch.isfinite(result).all():raise ValueError('Nonfinite light output')
        return result


def save_light(model,path):
    if type(model) is not LightCNN or any(m.training for m in model.modules()):raise ValueError('Frozen LightCNN required')
    state=model.state_dict();_check_state(state)
    spec=dict(architecture=model.architecture,source_state=model.source_state,method='independent_cnn_copy_v1')
    packet=dict(format='vca-light-cnn-cpu-v1',spec=spec,spec_hash=digest(spec),state=state)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
            temp=Path(f.name);torch.save(packet,f);f.flush();os.fsync(f.fileno())
        os.link(temp,path)
    finally:
        if temp is not None:temp.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_light(path,expected_sha256):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError('Light file hash mismatch')
    packet=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if not isinstance(packet,dict) or set(packet)!={'format','spec','spec_hash','state'} or packet['format']!='vca-light-cnn-cpu-v1':raise ValueError('Invalid light packet')
    spec=packet['spec']
    if (not isinstance(spec,dict) or set(spec)!={'architecture','source_state','method'} or digest(spec)!=packet['spec_hash']
            or spec['method']!='independent_cnn_copy_v1' or not isinstance(spec['source_state'],str)
            or len(spec['source_state'])!=64 or any(c not in '0123456789abcdef' for c in spec['source_state'])):raise ValueError('Invalid light specification')
    cfg=spec['architecture']
    with torch.random.fork_rng(devices=[]):
        source=MultiDatasetVCASAHGCN(cfg['class_counts'],**cfg['backbone_kwargs']).eval();source.backbone.adapter.cache_attention=False
        model=LightCNN(source)
    if model.architecture!=cfg:raise ValueError('Noncanonical architecture')
    _check_state(packet['state']);ref=model.state_dict()
    if (not isinstance(packet['state'],dict) or set(packet['state'])!=set(ref)
            or any(not torch.is_tensor(v) or v.shape!=ref[k].shape or v.dtype!=ref[k].dtype for k,v in packet['state'].items())):raise ValueError('Invalid light state schema')
    model.load_state_dict(packet['state'],strict=True);model.source_state=spec['source_state']
    return model.eval()
