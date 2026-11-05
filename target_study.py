"""Frozen full G4 target study; train/validation only, no query reader."""
import copy,json
from pathlib import Path
from joint_baseline import NAMES
from loadData.dataset_registry import file_hash
from target_initialization import METHODS
from target_training import TargetRuntime,target_settings

EXPECTED={'protocol':'g4_target_adaptation_full_v1','targets':list(NAMES),'training_seeds':[64,512,2048],
    'methods':list(METHODS),'support_seed':64,'support_per_class':10,'validation_per_class':5,
    'epochs':1499,'val_interval':30,'updates_per_case':1500,'new_optimizer_updates':40500,
    'source_producer_commit':'9c66e20f85d26e88ecb9e6790d8c1e2c62ae8ed1','source_declared_updates':1002,
    'selection':'minimum_calibrated_fused_validation_nll_first_tie','query_labels_during_training':False,
    'test_scoring_during_training':False,'reuse_joint_checkpoint':False,
    'target_graph_policy':'g4_target_label_free_static_v1','target_context':'g4_full_target_scene_transductive',
    'residual':'none','observation':'full','supervision':'legacy_three_heads','attention':'wavelength',
    'finalization':'original_strict_stop_on_failure'}


def load_study(path='configs/experiments/target_full_study.json'):
    value=json.loads(Path(path).read_text())
    if value!=EXPECTED:raise ValueError('Modified full target protocol; register another version')
    return value


def case_name(target,seed,method):return target+'-s'+str(seed)+'-'+method


def full_settings(target,method,seed,*,device='cuda'):
    value=target_settings(target,method,seed=seed,support_seed=64,per_class=10,device=device)
    value.update(protocol=EXPECTED['protocol'],profile='full',epochs=1499,val_interval=30,optimizer_steps=1500)
    value['config']['name']='g4_target_full_'+target
    value['config']['training'].update(epochs=1499,val_interval=30)
    value['target_study']=copy.deepcopy(EXPECTED)
    return value


class TargetStudyRuntime(TargetRuntime):
    def __init__(self,*args,initialization_state_sha256=None,**kwargs):
        self.expected_initialization=initialization_state_sha256
        super().__init__(*args,**kwargs)

    def build_model(self,config,counts):
        model=super().build_model(config,counts)
        if (self.expected_initialization is not None
                and self.initialization['initial_state_sha256']!=self.expected_initialization):
            raise ValueError('Frozen target initialization changed')
        return model

    def validate(self,settings):
        expected=full_settings(settings.get('target'),settings.get('method'),settings.get('model_seed'),
            device=settings['config']['device'])
        if {k:v for k,v in settings.items() if k not in ('executed_optimizer_steps','selected_step')}!=expected:
            raise ValueError('Full target runtime configuration differs')
        return True


def validate_source_case(case):
    """Match the accepted full source run to this held target and paired seed."""
    if case['method']==METHODS[0]:
        if case.get('source_spec') or case.get('source_result'):raise ValueError('Scratch source access forbidden')
        return
    binding=case['source_result'];spec=case['source_spec'];path=Path(binding['path'])
    if file_hash(path)!=binding['sha256']:raise ValueError('Full source result changed')
    value=json.loads(path.read_text());settings=value['settings']
    if (value['status']!='completed' or value['optimizer_steps']!=EXPECTED['source_declared_updates']
            or value['start_code']['commit']!=EXPECTED['source_producer_commit']
            or settings['protocol']!='g4_source_only_pretrain_full_v1'
            or settings['held_out']!=case['target'] or settings['model_seed']!=case['seed']
            or settings['source_tasks']!=[n for n in NAMES if n!=case['target']]
            or Path(spec['source_path']).resolve()!= (path.parent/'primary.pt').resolve()
            or spec['source_weight_sha256']!=value['artifacts']['primary.pt']
            or spec['source_sidecar_sha256']!=value['artifacts']['primary.pt.json']
            or file_hash(spec['source_path'])!=spec['source_weight_sha256']
            or file_hash(spec['source_path']+'.json')!=spec['source_sidecar_sha256']):
        raise ValueError('Source full producer/target/seed/pair mismatch')


def select_case(launch,source,target,seed,method):
    if (launch['code']!=source or launch['study']!=EXPECTED or launch['query_labels_accessed'] is not False
            or launch['test_scoring'] is not False or launch['expected_updates']!=40500):
        raise ValueError('Full target launch source/protocol differs')
    keys=[(c['target'],c['seed'],c['method']) for c in launch['cases']]
    expected={(t,s,m) for t in NAMES for s in EXPECTED['training_seeds'] for m in METHODS}
    if (len(keys)!=27 or set(keys)!=expected or type(seed)is not int or (target,seed,method) not in expected
            or len({c['out'] for c in launch['cases']})!=27):raise ValueError('Expected27 distinct registered target cases')
    root=Path(launch['out_dir']).resolve()
    target_inputs={}
    for case in launch['cases']:
        name=case_name(case['target'],case['seed'],case['method'])
        if (type(case['seed'])is not int or case['name']!=name
                or Path(case['out']).resolve()!=root/'cases'/name
                or case['settings']!=full_settings(case['target'],case['method'],case['seed'])):
            raise ValueError('Frozen target settings/name/path changed')
        if (case['method']==METHODS[0]) != (not case['source_spec'] and case['source_result'] is None):
            raise ValueError('Frozen source role differs')
        if case['target'] in target_inputs and target_inputs[case['target']]!=case['inputs']:
            raise ValueError('Paired target observation/label family differs')
        target_inputs[case['target']]=case['inputs']
    return next(c for c in launch['cases'] if (c['target'],c['seed'],c['method'])==(target,seed,method))
