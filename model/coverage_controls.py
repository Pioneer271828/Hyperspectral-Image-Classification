"""E1-C generator-input controls; never change the physical descriptor or gate."""
import hashlib
import math
import torch


def generator_features(coverage, condition, seed, anchor_valid=None):
    f=coverage.features.detach()
    if condition=='continuous':return f
    if condition=='anchor_only':
        p=f[:,0]
        return torch.stack([p,torch.sin(math.pi*p),torch.cos(math.pi*p),
            torch.sin(2*math.pi*p),torch.cos(2*math.pi*p),
            torch.sin(4*math.pi*p),torch.cos(4*math.pi*p),torch.ones_like(p),torch.ones_like(p)],1)
    if condition=='binary':
        if (not isinstance(anchor_valid,torch.Tensor) or anchor_valid.shape!=coverage.usable.shape
                or anchor_valid.dtype!=torch.bool or anchor_valid.device!=f.device):
            raise ValueError('binary coverage control requires actual VCA boolean anchor_valid')
        result=f.clone();result[:,1:5]=anchor_valid[:,None].to(f.dtype)
        return result
    if condition=='shuffled':
        identity=coverage.diagnostics.get('metadata_hash')
        if not isinstance(identity,str) or not identity:raise ValueError('shuffled control requires observation identity')
        k=len(f);value=int.from_bytes(hashlib.sha256(f'{seed}:{identity}:{k}'.encode()).digest()[:8],'big')%(2**63)
        permutation=torch.randperm(k,generator=torch.Generator(device='cpu').manual_seed(value))
        if k>1 and torch.equal(permutation,torch.arange(k)):permutation=permutation.roll(1)
        result=f.clone();result[:,1:5]=f[permutation.to(f.device),1:5]
        return result
    raise ValueError('Unknown coverage condition')
