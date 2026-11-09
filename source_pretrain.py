"""Frozen full source pretraining, independent of target observation preparation."""
import copy,json
from pathlib import Path
from joint_baseline import NAMES
from source_training import source_settings,SourceRuntime

EXPECTED={'protocol':'g4_source_only_pretrain_full_v1','held_targets':list(NAMES),
    'training_seeds':[64,512,2048],'epochs':500,'source_draws_per_epoch':2,
    'updates_per_case':1002,'new_optimizer_updates':9018,
    'selection':'calibrated_source_macro_fused_validation_oa_first_max',
    'target_payload_access':False,'test_scoring':False,'reuse_joint_checkpoint':False,
    'residual':'none','observation':'full','supervision':'legacy_three_heads',
    'attention':'wavelength','graph_context':'selected_source_train_label_lda_slic',
    'source_context':'full_source_scene_transductive','finalization':'original_strict_stop_on_failure'}


def load_study(path='configs/experiments/source_pretrain_full.json'):
    value=json.loads(Path(path).read_text())
    if value!=EXPECTED:raise ValueError('Modified source pretraining protocol')
    return value


def full_settings(held_out,seed):
    value=source_settings(held_out,seed=seed)
    value.update(protocol=EXPECTED['protocol'],profile='full',epochs=500,val_interval=10,optimizer_steps=1002)
    value['config']['name']='source_full_'+'_'.join(value['source_tasks'])
    value['config']['training'].update(epochs=500,val_interval=10)
    value['pretraining_study']=copy.deepcopy(EXPECTED)
    return value


class SourcePretrainRuntime(SourceRuntime):
    def validate(self,settings):
        expected=full_settings(settings.get('held_out'),settings.get('model_seed'))
        value={k:v for k,v in settings.items() if k not in ('executed_optimizer_steps','selected_step')}
        if value!=expected:raise ValueError('Full source pretraining configuration mismatch')
        return True


def select_case(launch,source,held_out,seed):
    if launch['code']!=source or launch['study']!=EXPECTED or launch['test_scoring'] is not False:
        raise ValueError('Source pretraining source/study mismatch')
    expected={(h,s) for h in NAMES for s in EXPECTED['training_seeds']}
    cases=launch['cases'];keys=[(c['held_out'],c['seed']) for c in cases]
    if (len(cases)!=9 or set(keys)!=expected or len({c['out'] for c in cases})!=9
            or any(type(c['seed'])is not int for c in cases)
            or type(seed)is not int or (held_out,seed) not in expected):
        raise ValueError('Nine distinct registered source cases required')
    for case in cases:
        if case['settings']!=full_settings(case['held_out'],case['seed']):raise ValueError('Frozen source case changed')
    return next(c for c in cases if (c['held_out'],c['seed'])==(held_out,seed))
