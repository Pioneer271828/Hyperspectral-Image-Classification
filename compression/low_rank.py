"""Explicit matrix low-rank compression, independent implementation using torch SVD.

Only Linear and ungrouped unit-stride 1x1 Conv2d are supported. The caller
chooses each layer/rank; no automatic pruning or change of model defaults.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def matrix(layer):
    if type(layer) is nn.Linear:
        kind = 'linear'
    elif (type(layer) is nn.Conv2d and layer.kernel_size == (1, 1)
          and layer.groups == 1 and layer.stride == (1, 1)
          and layer.padding == (0, 0) and layer.dilation == (1, 1)
          and layer.padding_mode == 'zeros'):
        kind = 'conv1x1'
    else:
        raise ValueError('only plain Linear or unit-stride ungrouped 1x1 Conv2d')
    if layer._forward_hooks or layer._forward_pre_hooks or layer._backward_hooks:
        raise ValueError('remove layer hooks before factorization')
    if not layer.weight.is_floating_point() or not torch.isfinite(layer.weight).all():
        raise ValueError('finite real floating weight required')
    if layer.bias is not None and not torch.isfinite(layer.bias).all():
        raise ValueError('finite bias required')
    return kind, layer.weight.detach().flatten(1)


def spectrum(layer):
    """Weight-only diagnostics; not a channel-removal or accuracy criterion."""
    kind, w = matrix(layer)
    s = torch.linalg.svdvals(w.cpu().double())
    energy = s.square()
    total = float(energy.sum())
    fractions = energy / total if total else torch.zeros_like(energy)
    return dict(kind=kind, shape=list(w.shape), singular_values=s.tolist(),
                cumulative_energy=fractions.cumsum(0).tolist(),
                participation_rank=float(1 / fractions.square().sum()) if total else 0.,
                stable_rank=total / float(energy[0]) if total else 0.)


class LowRankMatrix(nn.Module):
    """Two actual linear operations; never materialize dense W in forward."""
    def __init__(self, layer, rank, *, init='svd', seed=64, allow_expansion=False):
        super().__init__()
        kind, w = matrix(layer)
        m, n = w.shape
        if type(rank) is not int or not 1 <= rank <= min(m, n):
            raise ValueError('rank must be an integer within matrix dimensions')
        if type(allow_expansion) is not bool:
            raise ValueError('allow_expansion must be boolean')
        if not allow_expansion and rank * (m + n) >= m * n:
            raise ValueError('rank does not reduce parameter count')
        if init not in ('svd', 'random') or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('invalid initialization or seed')
        self.kind, self.rank = kind, rank
        self.in_features, self.out_features = n, m
        if init == 'svd':
            u, s, vh = torch.linalg.svd(w.cpu().double(), full_matrices=False)
            root = s[:rank].sqrt()
            left, right = u[:, :rank] * root, root[:, None] * vh[:rank]
        else:
            generator = torch.Generator(device='cpu').manual_seed(seed)
            right = torch.randn(rank, n, generator=generator, dtype=torch.float64) / math.sqrt(n)
            left = torch.randn(m, rank, generator=generator, dtype=torch.float64) / math.sqrt(rank)
        self.left = nn.Parameter(left.to(w).clone(), requires_grad=layer.weight.requires_grad)
        self.right = nn.Parameter(right.to(w).clone(), requires_grad=layer.weight.requires_grad)
        self.bias = (None if layer.bias is None else
                     nn.Parameter(layer.bias.detach().clone(), requires_grad=layer.bias.requires_grad))
        self.train(layer.training)
        if not all(torch.isfinite(p).all() for p in self.parameters()):
            raise ValueError('factor conversion produced nonfinite parameters')

    def forward(self, x):
        if self.kind == 'linear':
            return F.linear(F.linear(x, self.right), self.left, self.bias)
        return F.conv2d(F.conv2d(x, self.right[:, :, None, None]),
                        self.left[:, :, None, None], self.bias)


def factorize_model(model, ranks, *, init='svd', seed=64, allow_expansion=False):
    """Return an owned copy. Create optimizers *after* topology replacement.

    Shared modules/parameters are rejected rather than silently untied.
    An empty plan is an exact deepcopy, and leaves checkpoint APIs unchanged.
    """
    if hasattr(model, '_low_rank_plan'):
        raise ValueError('already factorized')
    if not isinstance(ranks, dict) or any(not isinstance(k, str) or not k for k in ranks):
        raise ValueError('explicit nonempty layer paths required')
    modules = dict(model.named_modules())
    all_modules = list(model.named_modules(remove_duplicate=False))
    all_params = list(model.named_parameters(remove_duplicate=False))
    replacements = {}
    for path, rank in sorted(ranks.items()):
        if path not in modules:
            raise ValueError('unknown layer: ' + path)
        layer = modules[path]
        if sum(other is layer for _, other in all_modules) != 1:
            raise ValueError('shared layer is unsupported')
        for param in layer.parameters():
            if sum(other is param for _, other in all_params) != 1:
                raise ValueError('shared parameter is unsupported')
        replacements[path] = LowRankMatrix(layer, rank, init=init, seed=seed,
                                            allow_expansion=allow_expansion)
    result = copy.deepcopy(model)
    for path, replacement in replacements.items():
        parent_path, _, leaf = path.rpartition('.')
        parent = result.get_submodule(parent_path) if parent_path else result
        setattr(parent, leaf, replacement)
    if ranks:
        result._low_rank_plan = dict(ranks=copy.deepcopy(ranks), init=init, seed=seed,
                                    allow_expansion=allow_expansion)
    return result
