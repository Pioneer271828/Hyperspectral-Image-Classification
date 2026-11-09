"""E3 registered development optimization, reusing accepted execution evidence.

Four dynamic route modes or a separately authenticated per-tile support bank.
No observation/sample curriculum, new edges, or automatic training launch.
"""
from copy import deepcopy
import numpy as np
import torch
from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from transfer.curriculum_training import CurriculumRun
from transfer.development_training import sparse_labels
from transfer.g3_training import _forward
from transfer.graph_route_context import RouteContextBank, route_state_hash, _model, _context


class GraphRouteRun(CurriculumRun):
    FORMAT = 'vca-graph-route-run-v1'
    FAILURE_FORMAT = 'vca-graph-route-failure-v1'
    ROUTE_FIELDS = ('route_input_nnz', 'route_output_nnz', 'route_added_self', 'route_max_degree',
                    'route_min_degree', 'route_budget', 'route_mode', 'route_step', 'route_fixed')

    def __init__(self, source, scene, record, *, bank=None, expected_bank_hash=None, **kwargs):
        if (kwargs.get('policy') is not None or kwargs.get('expected_policy_hash') is not None
                or kwargs.get('observation_policy') is not None or kwargs.get('expected_observation_hash') is not None):
            raise ValueError('Routing is a separate experiment with both curricula disabled')
        self.bank = None
        if bank is not None:
            if not isinstance(bank, RouteContextBank): raise ValueError('Registered support bank required')
            self.bank = RouteContextBank(bank.to_dict(), expected_hash=expected_bank_hash)
            v = self.bank.to_dict()
            expected = {_context(scene, record, t['id'])['tile_id']: _context(scene, record, t['id'])
                        for t in record['roles'] if t['role'] in ('devtrain', 'devval')}
            if (digest(v['architecture']) != digest(source.export_config())
                    or v['contexts'].keys() != {digest(c) for c in expected.values()}
                    or any(v['contexts'][digest(c)]['context'] != c for c in expected.values())):
                raise ValueError('Fixed bank must cover every matching development context')
        elif expected_bank_hash is not None:
            raise ValueError('Dynamic route cannot carry a fixed bank hash')
        kwargs.setdefault('policy', None); kwargs.setdefault('expected_policy_hash', None)
        super().__init__(source, scene, record, **kwargs)

    def _validate_model(self, model):
        r = _model(model, self.scene)
        if any(m.training for m in model.modules()) or r.step != 0:
            raise ValueError('Sealed E3 model must be eval with caller-neutral step0')

    def _model_hash(self, model): return route_state_hash(model)

    def _contract_extensions(self):
        return dict(graph_route=dict(kind='dynamic' if self.bank is None else 'fixed_support',
                    bank_hash=None if self.bank is None else self.bank.bank_hash,
                    support_source_state=None if self.bank is None else self.bank.to_dict()['source_state'],
                    training_step='one_based_optimizer_step', validation_step=0,
                    initialization_state=self.spec['source_state']))

    def _check_extensions(self):
        if self.spec['graph_route'] != self._contract_extensions()['graph_route']:
            raise ValueError('Registered route contract changed')
        if self.bank is not None:
            v = self.bank.to_dict()
            if digest({k:x for k,x in v.items() if k != 'bank_hash'}) != self.bank.bank_hash:
                raise ValueError('Support bank changed')

    def _predict(self, model, tile, step):
        if self.bank is not None:
            return self.bank.forward(model, self.scene, self.record, tile['id'], step=step)
        router = _model(model, self.scene); before = router.get_extra_state()
        router._context_busy = True
        try:
            router.set_step(step)
            return _forward(model, self.scene, tile)
        finally:
            router.set_extra_state(before); del router._context_busy

    def _training_forward(self, tile, step, extra):
        if extra: raise ValueError('Observation extension forbidden for route experiment')
        return self._predict(self.model, tile, step)

    def _evaluate(self, model):
        # Same full-validation arithmetic/order as the accepted development runner.
        # Random dynamic routes use step0 at EVERY selection event, including reload.
        labels = sparse_labels(self.record, self.scene.plan, 'validation_support')
        matrix = np.zeros((self.scene.classes, self.scene.classes), dtype=np.int64)
        total = correct = 0; nll = 0.; model.eval()
        with torch.no_grad():
            for tile in self.record['roles']:
                if tile['role'] != 'devval': continue
                target = labels[tile_slice(tile)].reshape(-1); valid = target > 0
                if not valid.any(): continue
                probs = self._predict(model, self.scene.plan['tiles'][tile['id']], 0)[0].detach().numpy()[valid]
                target = target[valid]-1; prediction = probs.argmax(1)
                matrix += np.bincount(target*self.scene.classes+prediction,
                                      minlength=self.scene.classes**2).reshape(matrix.shape)
                correct += int((prediction == target).sum()); total += len(target)
                nll -= float(np.log(np.maximum(probs[np.arange(len(target)), target], 1e-8)).sum())
        if total != 45: raise ValueError('Fixed validation budget changed')
        return dict(nll=nll/total, oa=correct/total, count=total, confusion=matrix.tolist(),
                    support=matrix.sum(1).tolist(), recall=(np.diag(matrix)/matrix.sum(1)).tolist())

    def _history_extensions(self, outputs):
        return dict(route={k: outputs[3]['interaction'][k] for k in self.ROUTE_FIELDS})

    def _history_extension_keys(self): return {'route'}

    def _validate_history_extensions(self, row):
        v = row['route']; r = self.source.backbone.graph_router
        c = _context(self.scene, self.record, row['tile_id']); n = int(np.prod(c['grid_shape']))
        if (set(v) != set(self.ROUTE_FIELDS) or type(v['route_fixed']) is not bool
                or v['route_fixed'] != (self.bank is not None) or v['route_mode'] != r.mode
                or any(type(v[k]) is not int for k in self.ROUTE_FIELDS if k not in ('route_mode', 'route_fixed'))
                or v['route_step'] != row['step'] or v['route_budget'] != r.budget
                or not 1 <= v['route_min_degree'] <= v['route_max_degree'] <= min(n, r.budget)
                or not 0 <= v['route_added_self'] <= n
                or not 0 <= v['route_input_nnz'] <= n*n
                or not n <= v['route_output_nnz'] <= min(n*r.budget, v['route_input_nnz']+v['route_added_self'])):
            raise ValueError('Invalid registered routing execution')
        if self.bank is not None:
            fixed = self.bank.to_dict()['contexts'][digest(c)]['fixed']
            if v['route_output_nnz'] != len(fixed['selected']): raise ValueError('Fixed support count changed')

    def _restore_model(self, state):
        ref = self.model.state_dict()
        if not isinstance(state, dict) or set(state) != set(ref): raise ValueError('Invalid routed model state')
        for k, v in state.items():
            if torch.is_tensor(ref[k]):
                if not torch.is_tensor(v) or v.shape != ref[k].shape or v.dtype != ref[k].dtype or not torch.isfinite(v).all():
                    raise ValueError('Invalid routed tensor')
            elif digest(v) != digest(ref[k]):
                raise ValueError('Router extra state changed outside bound execution')
        self.model.load_state_dict(state)

    @classmethod
    def load(cls, source, scene, record, path, *, plan, expected_sha256, expected_contract_hash,
             bank=None, expected_bank_hash=None):
        return super().load(source, scene, record, path, plan=plan, policy=None,
                            expected_sha256=expected_sha256, expected_contract_hash=expected_contract_hash,
                            constructor_kwargs=dict(bank=bank, expected_bank_hash=expected_bank_hash))
