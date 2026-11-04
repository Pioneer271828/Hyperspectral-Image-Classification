"""Opt-in M4 x auxiliary supervision; original trainer and defaults unchanged."""
import copy
import math
import torch

from joint_baseline import forward
from joint_training import train_step, validation_snapshot
from model import utils
from scripts.train_joint_m4 import M4JointRuntime, m4_settings

MODES = ('none', 'x_residual')
SUPERVISION = ('legacy_three_heads', 'primary_only')


def policy(mode, supervision):
    if mode not in MODES or supervision not in SUPERVISION:
        raise ValueError('Unregistered M4/supervision cell')
    return {'schema_version': 1, 'mode': mode, 'supervision': supervision,
            'auxiliary_parameters': 'retained_initialization_no_gradient_no_update' if supervision == 'primary_only' else 'trained',
            'reported_heads': ['fused'] if supervision == 'primary_only' else ['fused', 'graph', 'cnn']}


def supervision_settings(settings, mode, supervision):
    spec = policy(mode, supervision)
    result = m4_settings(settings, mode)
    if 'supervision_policy' in settings:
        raise ValueError('Expected baseline settings without supervision override')
    for key in ('cnn_loss_weight', 'graph_branch_loss_weight'):
        if result['config']['loss'][key] != 1.0:
            raise ValueError('Expected original unit auxiliary weights')
        result['config']['loss'][key] = 0.0 if supervision == 'primary_only' else 1.0
    result['supervision_policy'] = spec
    return result


class SupervisionRuntime(M4JointRuntime):
    def __init__(self, mode, supervision):
        self.spec = policy(mode, supervision)

    def validate(self, settings):
        super().validate(settings)
        weight = 0.0 if self.spec['supervision'] == 'primary_only' else 1.0
        if (settings.get('supervision_policy') != self.spec
                or settings['config']['model']['region_residual_mode'] != self.spec['mode']
                or any(settings['config']['loss'][k] != weight for k in ('cnn_loss_weight', 'graph_branch_loss_weight'))):
            raise ValueError('Supervision settings/policy mismatch')
        return True

    def build_model(self, config, counts):
        if config.model.region_residual_mode != self.spec['mode']:
            raise ValueError('M4 runtime/config mismatch')
        return super().build_model(config, counts)

    def train_step(self, model, optimizer, name, item, config):
        primary = self.spec['supervision'] == 'primary_only'
        if model.backbone.region_residual_mode != self.spec['mode']:
            raise ValueError('M4 runtime/model mismatch')
        expected = 0.0 if primary else 1.0
        if config.loss.cnn_loss_weight != expected or config.loss.graph_branch_loss_weight != expected:
            raise ValueError('Runtime loss weights mismatch')
        prefixes = ('hgnn_heads.', 'cnn_heads.')
        before_aux = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith(prefixes)} if primary else {}
        before_residual = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith('backbone.region_residual.')}
        if not primary:
            result = train_step(model, optimizer, name, item, config)
        else:
            model.train(); optimizer.zero_grad(set_to_none=True)
            outputs = forward(model, name, item)
            loss = utils.compute_loss(outputs[0], item['train_gt_onehot'], item['train_label_mask'], class_balance=config.loss.balanced)
            if not torch.isfinite(loss): raise ValueError('Nonfinite primary loss')
            loss.backward()
            gradients = {}
            for group, prefix in [('adapter', 'backbone.adapter.'), ('cnn', 'backbone.CNN_Branch.'),
                                  ('graph', 'backbone.HGNN_Branch.'), ('fused_head', 'heads.'+name+'.')]:
                params = [p for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
                if not params or any(not torch.isfinite(p.grad).all() for p in params):
                    raise ValueError('Missing/nonfinite gradient: '+group)
                gradients[group] = math.sqrt(sum(float(p.grad.double().square().sum()) for p in params))
            if any(p.grad is not None for n, p in model.named_parameters() if n in before_aux):
                raise ValueError('Inactive auxiliary head acquired a gradient')
            optimizer.step()
            if any(not torch.equal(p.detach(), before_aux[n]) for n, p in model.named_parameters() if n in before_aux):
                raise ValueError('Inactive auxiliary head changed')
            result = {'dataset': name, 'loss': {'fused_supervised': float(loss.detach()), 'total': float(loss.detach())},
                      'gradient_norms': gradients, 'learning_rates': [float(g['lr']) for g in optimizer.param_groups]}
        residual = [(n, p) for n, p in model.named_parameters() if n in before_residual]
        if residual and any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in residual):
            raise ValueError('Missing/nonfinite residual gradient')
        result.update(supervision_policy=copy.deepcopy(self.spec),
                      inactive_auxiliary_changed_elements=0 if primary else None,
                      residual_gradient_norm=math.sqrt(sum(float(p.grad.double().square().sum()) for _, p in residual)),
                      residual_changed_elements=sum(int(torch.count_nonzero(p.detach()-before_residual[n])) for n, p in residual))
        return result

    def validation_snapshot(self, model, items):
        metric, probabilities = validation_snapshot(model, items)
        if self.spec['supervision'] == 'primary_only':
            for name in items:
                for phase in ('train', 'val'):
                    metric['metrics'][name][phase] = {'fused': metric['metrics'][name][phase]['fused']}
                    probabilities[name][phase] = {'fused': probabilities[name][phase]['fused']}
        return metric, probabilities

    def load_selected(self, path, items, raw_source_path, *, device='cpu'):
        model = super().load_selected(path, items, raw_source_path, device=device)
        recorded = model.training_config.get('joint_runtime', {})
        self.validate(recorded)
        if recorded.get('config') != model.training_config.get('research_config'):
            raise ValueError('Checkpoint research/runtime configuration mismatch')
        if model.backbone.region_residual_mode != self.spec['mode']:
            raise ValueError('Checkpoint residual mode mismatch')
        return model
