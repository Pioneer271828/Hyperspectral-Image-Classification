"""Separate one-file compressed model format; legacy checkpoints stay unchanged."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile

import torch

from compression.low_rank import LowRankMatrix, factorize_model
from model.VCASAHGCN import MultiDatasetVCASAHGCN


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_low_rank(model, path):
    if type(model) is not MultiDatasetVCASAHGCN or not hasattr(model, '_low_rank_plan'):
        raise ValueError('explicitly factorized VCA model required')
    plan = copy.deepcopy(model._low_rank_plan)
    actual = {name: module.rank for name, module in model.named_modules()
              if isinstance(module, LowRankMatrix)}
    if actual != plan['ranks']:
        raise ValueError('factor topology differs from plan')
    state = model.state_dict()
    dtypes = {v.dtype for v in state.values() if torch.is_tensor(v) and v.is_floating_point()}
    if len(dtypes) != 1:
        raise ValueError('mixed floating dtype checkpoint unsupported')
    if any(not torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v)):
        raise ValueError('nonfinite checkpoint')
    packet = dict(format='vca-matrix-low-rank-v1', architecture=model.export_config(),
                  plan=plan, state=state, dtype=str(next(iter(dtypes))).split('.')[-1],
                  trainable={k: p.requires_grad for k, p in model.named_parameters()},
                  training_config=copy.deepcopy(getattr(model, 'training_config', {})))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temp = Path(handle.name)
            torch.save(packet, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)  # atomic no-clobber publication on the same filesystem
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return sha256(path)


def load_low_rank(path, expected_sha256, *, device='cpu'):
    if sha256(path) != expected_sha256:
        raise ValueError('checkpoint hash mismatch')
    packet = torch.load(path, map_location='cpu', weights_only=True)
    if (not isinstance(packet, dict) or set(packet) !=
            {'format', 'architecture', 'plan', 'state', 'dtype', 'trainable', 'training_config'}
            or packet['format'] != 'vca-matrix-low-rank-v1'):
        raise ValueError('not a low-rank checkpoint')
    if packet['dtype'] not in ('float64', 'float32', 'float16', 'bfloat16'):
        raise ValueError('unsupported dtype')
    cfg = packet['architecture']
    # Reconstructing a model must not consume the caller's training RNG stream.
    with torch.random.fork_rng(devices=[]):
        dense = MultiDatasetVCASAHGCN(cfg['class_counts'], **cfg['backbone_kwargs'])
        dense = dense.to(dtype=getattr(torch, packet['dtype']))
        plan = packet['plan']
        if (not isinstance(plan, dict) or set(plan) != {'ranks', 'init', 'seed', 'allow_expansion'}
                or not plan['ranks']):
            raise ValueError('invalid low-rank plan')
        # Factor shapes only: saved factors replace this private initialization.
        model = factorize_model(dense, plan['ranks'], init='random', seed=plan['seed'],
                                allow_expansion=plan['allow_expansion'])
    if plan['init'] not in ('svd', 'random'):
        raise ValueError('invalid factor initialization identity')
    model._low_rank_plan = copy.deepcopy(plan)
    model.load_state_dict(packet['state'], strict=True)
    params = dict(model.named_parameters())
    if set(params) != set(packet['trainable']) or any(type(v) is not bool for v in packet['trainable'].values()):
        raise ValueError('trainable parameter contract mismatch')
    for name, param in params.items():
        param.requires_grad_(packet['trainable'][name])
    if any(not torch.isfinite(v).all() for v in model.state_dict().values() if torch.is_tensor(v)):
        raise ValueError('nonfinite checkpoint')
    model.training_config = packet['training_config']
    return model.to(device).eval()
