"""Fixed elementwise weight masks with explicit unmasking semantics."""
import torch
from torch import nn
from torch.nn.utils import parametrize


class _FixedMask(nn.Module):
    def __init__(self, keep):
        super().__init__()
        self.register_buffer('keep', keep.detach().clone())

    def forward(self, weight):
        # masked_fill also yields exact zeros if a hidden masked entry becomes nonfinite.
        return weight.masked_fill(~self.keep, 0)


class MaskedWeights:
    """Explicit opt-in Linear/Conv2d masks, retaining optimizer Parameter identities.

    recovery is mandatory: keep_latent reveals the latest underlying values;
    restore_masked_values restores ONLY masked weight values captured on attach.
    Neither choice restores optimizer states or rolls back training.
    """
    def __init__(self, model, masks, *, recovery):
        if recovery not in ('keep_latent', 'restore_masked_values'):
            raise ValueError('Explicit recovery policy required')
        if not isinstance(masks, dict) or not masks:
            raise ValueError('Nonempty parameter-path -> boolean mask dictionary required')
        self.model, self.recovery = model, recovery
        self._masks = {}
        for path, mask in masks.items():
            if not isinstance(path, str) or not path or not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
                raise ValueError('Named parameters and boolean tensor masks required')
            self._masks[path] = mask.detach().clone()
        self._attached = []
        self._state_guard = None

    @property
    def active(self):
        return bool(self._attached)

    def attach(self):
        if self.active: raise RuntimeError('Masks already active')
        modules = dict(self.model.named_modules())
        references = {}
        for _, parameter in self.model.named_parameters(remove_duplicate=False):
            references[id(parameter)] = references.get(id(parameter), 0)+1
        targets = []
        # Validate the whole plan before touching any module.
        for path, mask in self._masks.items():
            module_path, _, name = path.rpartition('.')
            module = modules.get(module_path)
            if name != 'weight' or not isinstance(module, (nn.Linear, nn.Conv2d)):
                raise ValueError('Only explicit Linear/Conv2d weight parameters are supported: '+path)
            if parametrize.is_parametrized(module):
                raise ValueError('Existing/overlapping parametrizations are not supported: '+path)
            weight = module._parameters.get('weight')
            if (weight is None or not weight.is_floating_point() or tuple(mask.shape) != tuple(weight.shape)
                    or not torch.isfinite(weight).all()):
                raise ValueError('Finite floating weight and exactly matching mask shape required: '+path)
            if references[id(weight)] != 1: raise ValueError('Shared/tied parameters are not supported: '+path)
            keep = mask.to(device=weight.device)
            targets.append((module, weight, keep))
        try:
            for module, weight, keep in targets:
                restore = weight.detach()[~keep].clone()
                mapping = _FixedMask(keep)
                parametrize.register_parametrization(module, 'weight', mapping)
                self._attached.append((module, weight, keep, restore))
                if module.parametrizations.weight.original is not weight:
                    raise RuntimeError('Parametrization replaced the optimizer parameter')
        except BaseException:
            self.detach()
            raise
        def reject_legacy_state(module, prefix, keep_vars):
            if not getattr(self, '_checkpoint_export', False):
                raise RuntimeError('Active masks cannot use legacy state_dict/checkpoints; detach first. mask_plan is not a training checkpoint.')
        self._state_guard = self.model.register_state_dict_pre_hook(reject_legacy_state)
        return self

    def detach(self):
        if self._state_guard is not None:
            self._state_guard.remove()
            self._state_guard = None
        # The underlying parameter keeps all ordinary optimizer updates.
        for module, original, keep, restore in reversed(self._attached):
            parametrize.remove_parametrizations(module, 'weight', leave_parametrized=False)
            if module.weight is not original: raise RuntimeError('Original Parameter identity changed')
            if self.recovery == 'restore_masked_values':
                with torch.no_grad(): original[~keep] = restore
        self._attached.clear()

    def __enter__(self):
        return self.attach()

    def __exit__(self, kind, error, traceback):
        self.detach()
        return False

    def _checkpoint_payload(self):
        """Internal explicit export; the legacy state_dict guard remains active otherwise."""
        if not self.active: raise RuntimeError('Active masks required')
        self._checkpoint_export = True
        try:
            state = {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
        finally:
            self._checkpoint_export = False
        snapshots = {}
        for path, (module, original, keep, restore) in zip(self._masks, self._attached):
            prefix = path[:-len('weight')]
            state[path] = state.pop(prefix+'parametrizations.weight.original')
            exported_mask = state.pop(prefix+'parametrizations.weight.0.keep')
            if not torch.equal(exported_mask, self._masks[path].cpu()):
                raise ValueError('Active mask was externally modified')
            snapshots[path] = restore.detach().cpu().clone()
        if any('parametrizations.' in key for key in state):
            raise ValueError('Unmanaged parametrizations cannot be checkpointed')
        return state, snapshots
    def mask_plan(self):
        """Portable fixed masks only, NOT a model/optimizer training checkpoint."""
        return {'schema_version': 1, 'kind': 'explicit_weight_mask_plan',
                'masks': {path: mask.detach().cpu().clone() for path, mask in self._masks.items()}}

    @classmethod
    def from_plan(cls, model, plan, *, recovery):
        if (not isinstance(plan, dict) or set(plan) != {'schema_version','kind','masks'}
                or type(plan['schema_version']) is not int or plan['schema_version'] != 1
                or plan['kind'] != 'explicit_weight_mask_plan'):
            raise ValueError('Unsupported mask plan')
        return cls(model, plan['masks'], recovery=recovery)


