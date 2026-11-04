"""Existing VCA modes in the fixed calibrated joint benchmark, opt-in only."""
import copy
from scripts.train_joint_m4 import m4_settings, M4JointRuntime

MODES = ('wavelength', 'index', 'uniform')


def adapter_settings(base, mode):
    if mode not in MODES: raise ValueError('Unknown adapter comparison mode')
    result = m4_settings(base, 'none')
    if (result['config']['model']['adapter_attention_mode'] != 'wavelength'
            or result['config']['model'].get('coverage_mode', 'none') != 'none'
            or 'joint_observation' in result or 'supervision_policy' in result):
        raise ValueError('Expected full-observation original-supervision baseline')
    result['config'] = copy.deepcopy(result['config'])
    result['config']['model']['adapter_attention_mode'] = mode
    result['adapter_comparison'] = {'version': 1, 'mode': mode,
        'registered_parameter_capacity': 'unchanged',
        'attention_parameters_active': mode != 'uniform',
        'metadata_claim': 'provided_quality_only_not_assumed_measured'}
    return result


class AdapterModeRuntime(M4JointRuntime):
    def __init__(self, mode):
        if mode not in MODES: raise ValueError('Unknown adapter mode')
        self.mode = mode

    def validate(self, settings):
        super().validate(settings)
        cfg = settings['config']
        spec = settings.get('adapter_comparison', {})
        if (spec != {'version': 1, 'mode': self.mode, 'registered_parameter_capacity': 'unchanged',
                     'attention_parameters_active': self.mode != 'uniform',
                     'metadata_claim': 'provided_quality_only_not_assumed_measured'}
                or cfg['model']['adapter_attention_mode'] != self.mode
                or cfg['model']['region_residual_mode'] != 'none'
                or cfg['model'].get('coverage_mode', 'none') != 'none'
                or any(cfg['loss'][k] != 1.0 for k in ('cnn_loss_weight', 'graph_branch_loss_weight'))
                or 'joint_observation' in settings or 'supervision_policy' in settings):
            raise ValueError('Adapter comparison policy mismatch')
        return True

    def build_model(self, config, counts):
        if config.model.adapter_attention_mode != self.mode: raise ValueError('Adapter build mode mismatch')
        return super().build_model(config, counts)

    def load_selected(self, path, items, raw_source_path, *, device='cpu'):
        model = super().load_selected(path, items, raw_source_path, device=device)
        recorded = model.training_config['joint_runtime']; self.validate(recorded)
        if (recorded['config'] != model.training_config['research_config']
                or model.backbone.adapter.attention_mode != self.mode):
            raise ValueError('Checkpoint adapter mode/config mismatch')
        return model
