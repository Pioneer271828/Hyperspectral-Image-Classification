"""Registered full source-only M4 study; short transfer contracts stay unchanged."""
import copy,json
from pathlib import Path
from joint_baseline import NAMES
from m4_transfer import M4SourceRuntime,source_options
from target_initialization import state_id

MODES=('capacity_control','x_residual')
EXPECTED={'protocol':'g4_m4_source_pretrain_full_v1','held_targets':list(NAMES),
    'training_seeds':[64,512,2048],'modes':list(MODES),'epochs':500,
    'source_draws_per_epoch':2,'updates_per_case':1002,'new_optimizer_updates':18036,
    'selection':'calibrated_source_macro_fused_validation_oa_first_max',
    'target_payload_access':False,'test_scoring':False,'reuse_joint_checkpoint':False,
    'observation':'full','supervision':'legacy_three_heads','attention':'wavelength',
    'graph_context':'selected_source_train_label_lda_slic',
    'source_context':'full_source_scene_transductive','finalization':'original_strict_stop_on_failure',
    'none_source_reuse_commit':'9c66e20f85d26e88ecb9e6790d8c1e2c62ae8ed1'}


def load_study(path='configs/experiments/m4_source_full.json'):
    value=json.loads(Path(path).read_text())
    if value!=EXPECTED:raise ValueError('Modified full M4 source protocol')
    return value


def full_settings(held_out,seed,mode):
    if mode not in MODES:raise ValueError('Only registered active source modes')
    value=source_options(held_out,mode,seed=seed)
    value.update(protocol=EXPECTED['protocol'],profile='full',epochs=500,val_interval=10,optimizer_steps=1002)
    value['config']['name']='m4_source_full_'+held_out+'_'+mode
    value['config']['training'].update(epochs=500,val_interval=10)
    value['m4_source_study']=copy.deepcopy(EXPECTED)
    return value


class M4SourceFullRuntime(M4SourceRuntime):
    def __init__(self,initial_state_sha256=None):self.expected_initial_state=initial_state_sha256

    def validate(self,settings):
        expected=full_settings(settings.get('held_out'),settings.get('model_seed'),settings.get('m4_mode'))
        if {k:v for k,v in settings.items() if k not in ('executed_optimizer_steps','selected_step')}!=expected:
            raise ValueError('Full M4 source settings differ')
        return True

    def build_model(self,config,counts):
        model=super().build_model(config,counts)
        if self.expected_initial_state is not None and state_id(model)!=self.expected_initial_state:
            raise ValueError('Frozen full M4 source initialization differs')
        return model


def case_name(held_out,seed,mode):return held_out+'-s'+str(seed)+'-'+mode


def select_case(launch,source,held_out,seed,mode):
    if (launch['code']!=source or launch['study']!=EXPECTED or launch['test_scoring'] is not False
            or launch['expected_updates']!=18036):raise ValueError('M4 source launch identity differs')
    expected={(h,s,m) for h in NAMES for s in EXPECTED['training_seeds'] for m in MODES}
    cases=launch['cases'];keys=[(c['held_out'],c['seed'],c['mode']) for c in cases]
    if (len(cases)!=18 or set(keys)!=expected or any(type(c['seed'])is not int for c in cases)
            or type(seed)is not int or (held_out,seed,mode) not in expected):
        raise ValueError('Eighteen distinct M4 source cases required')
    root=Path(launch['out_dir']).resolve()
    for c in cases:
        if (c['settings']!=full_settings(c['held_out'],c['seed'],c['mode'])
                or c['name']!=case_name(c['held_out'],c['seed'],c['mode'])
                or Path(c['out']).resolve()!=root/'cases'/c['name']):
            raise ValueError('M4 source case configuration/output differs')
        prov=c['provenance'];ctx=c['source_context']
        if (prov!={**ctx,'m4_mode':c['mode'],'m4_graph_binding':c['graph_binding']}
                or ctx.get('held_out')!=c['held_out'] or ctx.get('target_payload_accessed') is not False
                or ctx.get('sources')!=c['settings']['source_tasks']
                or set(c['graph_binding'])!=set(c['settings']['source_tasks'])):
            raise ValueError('M4 source roles/provenance/graph differ')
    return next(c for c in cases if (c['held_out'],c['seed'],c['mode'])==(held_out,seed,mode))
