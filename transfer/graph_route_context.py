"""E3 development-only, per-tile fixed support with explicit node semantics.

No model arithmetic changes. This bank is a required training input, not a
training checkpoint or authorization to evaluate query. Whole-model recovery
must separately bind weights, optimizer, sampler, budget and selection.
"""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import numpy as np
import torch

from loadData.spatial_tiles import tile_slice
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.budgeted_graph_router import BudgetedGraphRouter
from model.learnable_superpixel_graph import LearnableSuperpixelGraph
from model.research_config import canonical_json, digest
from research_runtime import tensor_hash
from transfer.g3_training import _forward
from transfer.observation_policy import scene_identity


def route_state_hash(model):
    """Unlike tensor-only legacy hashes, include the router's JSON extra state."""
    return digest({k: {'tensor': tensor_hash(v)} if torch.is_tensor(v)
                   else {'json': v} for k, v in model.state_dict().items()})


def _model(model, scene):
    if type(model) is not MultiDatasetVCASAHGCN:
        raise ValueError('Original multi-dataset model required')
    b = model.backbone
    if (b.graph_mode != 'learned' or type(b.graph_router) is not BudgetedGraphRouter
            or b.adapter.cache_attention or b.graph_router.fixed is not None
            or any(getattr(b, k) is not None for k in ('interaction', 'region_residual', 'coverage_modulation'))
            or list(model.heads) != [scene.dataset] or model.heads[scene.dataset].out_features != scene.classes
            or any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
            or any(p.device.type != 'cpu' or p.dtype != torch.float32 or not torch.isfinite(p).all()
                   for p in model.parameters())):
        raise ValueError('Uncoupled learned-graph CPU FP32 GroupNorm model with unbound router required')
    r = b.graph_router
    if r._forward_hooks or r._forward_pre_hooks or getattr(r, '_context_busy', False):
        raise ValueError('Router already intercepted or active')
    return r


def _context(scene, record, tile_id):
    identity = scene_identity(scene, record)
    if type(tile_id) is not int or not 0 <= tile_id < len(record['roles']):
        raise ValueError('Integer development tile id required')
    role = record['roles'][tile_id]['role']
    tile = scene.plan['tiles'][tile_id]
    if role not in ('devtrain', 'devval'):
        raise ValueError('Only development training/validation contexts are allowed')
    raw = np.ascontiguousarray(scene.cube[tile_slice(tile)])
    if raw.dtype != np.float32 or not np.isfinite(raw).all():
        raise ValueError('Finite FP32 development block required')
    scale = 50 if scene.dataset == 'paviaU' else 100
    grid = list(LearnableSuperpixelGraph._grid_shape(*raw.shape[:2], scale))
    return dict(identity=identity, tile_id=tile_id, role=role, box=tile['box'],
                shape=list(raw.shape), raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                graph_scale=scale, grid_shape=grid,
                node_order='row_major_seed_grid_v1', node_position='grid_seed_centers',
                observation='full_physical_bands')


def _keys(graph):
    graph = graph.coalesce()
    return (graph.indices()[0] * graph.shape[0] + graph.indices()[1]).cpu().tolist()


class RouteContextBank:
    """Authenticated per-context masks; values and learned assignments may train.

    The fixed masks are chosen once from the registered evaluation source. No
    labels are read for selection. Validation never backpropagates. Calls are
    sequential; the original router state is restored even when forward fails.
    """
    def __init__(self, value, *, expected_hash):
        self._value = json.loads(canonical_json(value))
        v = self._value
        if (set(v) != {'version', 'source_state', 'architecture', 'freeze_step', 'contexts', 'bank_hash'}
                or v['version'] != 1 or type(v['freeze_step']) is not int or v['freeze_step'] < 0
                or v['bank_hash'] != expected_hash
                or digest({k: x for k, x in v.items() if k != 'bank_hash'}) != expected_hash
                or not isinstance(v['source_state'], str) or len(v['source_state']) != 64
                or not isinstance(v['contexts'], dict) or not v['contexts']):
            raise ValueError('Registered support bank identity/schema mismatch')
        config = v['architecture']['backbone_kwargs']['graph_route']
        for key, row in v['contexts'].items():
            if (set(row) != {'context', 'fixed'} or digest(row['context']) != key
                    or row['context']['role'] not in ('devtrain', 'devval')
                    or row['context']['node_order'] != 'row_major_seed_grid_v1'):
                raise ValueError('Invalid bound context')
            fixed = row['fixed']; n = np.prod(row['context']['grid_shape']).item()
            if (set(fixed) != {'nodes', 'candidates', 'selected'} or type(fixed['nodes']) is not int
                    or fixed['nodes'] != n or n < 1
                    or any(type(k) is not int or not 0 <= k < n*n for k in fixed['candidates'])
                    or fixed['candidates'] != sorted(set(fixed['candidates']))):
                raise ValueError('Invalid node/candidate identity')
            keys = torch.tensor(fixed['candidates'], dtype=torch.long)
            a = torch.sparse_coo_tensor(torch.stack((keys//n, keys % n)),
                                        torch.ones(len(keys)), (n, n)).coalesce()
            r = BudgetedGraphRouter(**config); r.fixed = deepcopy(fixed)
            r(a)  # Reuse budget, self-loop, membership and sorted-selection checks.
        self.bank_hash = expected_hash

    def to_dict(self):
        return deepcopy(self._value)

    @classmethod
    def freeze(cls, source, scene, record, *, tile_ids, step, expected_source_state):
        _model(source, scene)
        if (any(m.training for m in source.modules()) or route_state_hash(source) != expected_source_state
                or type(step) is not int or step < 0 or not isinstance(tile_ids, list)
                or not tile_ids or any(type(i) is not int for i in tile_ids)
                or tile_ids != sorted(set(tile_ids))):
            raise ValueError('Frozen source, unique ordered tiles and explicit step required')
        contexts = [_context(scene, record, i) for i in tile_ids]
        model = deepcopy(source); router = model.backbone.graph_router; router.set_step(step)
        rows = {}
        for context in contexts:
            seen = []
            def capture(module, args, result):
                graph, _, _ = module._graph(args[0]); out = result[0]
                seen.append(dict(nodes=graph.shape[0], candidates=_keys(graph), selected=_keys(out)))
            handle = router.register_forward_hook(capture)
            try:
                with torch.no_grad(): _forward(model, scene, scene.plan['tiles'][context['tile_id']])
            finally:
                handle.remove()
            if len(seen) != 1:
                raise ValueError('Exactly one route call per context required')
            rows[digest(context)] = dict(context=context, fixed=seen[0])
        value = dict(version=1, source_state=expected_source_state,
                     architecture=source.export_config(), freeze_step=step, contexts=rows)
        value['bank_hash'] = digest(value)
        if route_state_hash(source) != expected_source_state:
            raise ValueError('Source changed during support freeze')
        return cls(value, expected_hash=value['bank_hash'])

    def forward(self, model, scene, record, tile_id, *, step):
        router = _model(model, scene)
        if (digest({k: v for k, v in self._value.items() if k != 'bank_hash'}) != self.bank_hash
                or digest(model.export_config()) != digest(self._value['architecture'])):
            raise ValueError('Bank or architecture changed')
        context = _context(scene, record, tile_id)
        row = self._value['contexts'].get(digest(context))
        if row is None or row['context'] != context:
            raise ValueError('Unregistered or changed development context')
        if type(step) is not int or step < 0:
            raise ValueError('Explicit nonnegative route step required')
        validation = context['role'] == 'devval'
        if validation and any(m.training for m in model.modules()):
            raise ValueError('Development validation requires evaluation mode')
        before = router.get_extra_state()
        router._context_busy = True
        try:
            router.set_step(step); router.fixed = deepcopy(row['fixed'])
            with torch.no_grad() if validation else nullcontext():
                return _forward(model, scene, scene.plan['tiles'][tile_id])
        finally:
            router.set_extra_state(before)
            del router._context_busy
