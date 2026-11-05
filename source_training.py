"""Source-only short training contract; no target adaptation or full budget."""
import copy
from dataclasses import replace
from joint_baseline import load_config
from joint_calibrated_selection import VALIDATION_POLICY,load_selected
from joint_training import train_step,validation_snapshot
from source_data_context import roles

PROTOCOL='source_only_engineering_v1'


def source_settings(held_out,sources=None,seed=64):
    selected=roles(held_out,sources)
    if type(seed) is not int or seed not in (64,512,2048):raise ValueError('Unregistered source seed')
    cfg=load_config('configs/experiments/base97_joint.json')
    cfg=replace(cfg,name='source_engineering_'+'_'.join(selected),
        data=replace(cfg.data,datasets=list(selected)),
        training=replace(cfg.training,seed=seed,epochs=3,steps_per_epoch=len(selected),val_interval=1))
    cfg.validate()
    return {'protocol':PROTOCOL,'profile':'engineering','held_out':held_out,'source_tasks':list(selected),
        'model_seed':seed,'schedule_seed':seed,'split_root_seed':64,'epochs':3,'val_interval':1,
        'steps_per_epoch':len(selected),'optimizer_steps':4*len(selected),'test_scoring':False,
        'scheduler':{'factor':.5,'patience':20,'threshold':.0001},'config':cfg.to_dict(),
        'validation_policy':copy.deepcopy(VALIDATION_POLICY),
        'source_policy':{'target_payload_access':False,'initialization':'fresh_source_only',
            'source_context':'source_full_scene_train_label_lda_slic','target_graph_reuse':False}}


class SourceRuntime:
    def validate(self,settings):
        expected=source_settings(settings.get('held_out'),settings.get('source_tasks'),settings.get('model_seed'))
        # Final checkpoint metadata adds these two execution fields only.
        recorded={k:v for k,v in settings.items() if k not in ('executed_optimizer_steps','selected_step')}
        if recorded!=expected:raise ValueError('Source training policy/configuration mismatch')
        return True

    def task_names(self,settings):
        self.validate(settings)
        return tuple(settings['source_tasks'])

    def build_model(self,config,counts):
        if tuple(counts)!=tuple(config.data.datasets):raise ValueError('Source model task mismatch')
        return config.build_model(counts)

    train_step=staticmethod(train_step)
    validation_snapshot=staticmethod(validation_snapshot)

    def load_selected(self,path,items,raw_source_path,*,device='cpu'):
        model=load_selected(path,items,raw_source_path,device=device)
        settings=model.training_config['joint_runtime'];self.validate(settings)
        tasks=self.task_names(settings)
        if (tuple(items)!=tasks or tuple(model.heads)!=tasks
                or model.training_config['research_config']!=settings['config']
                or any(settings['held_out'] in n.split('.') for n in model.state_dict())):
            raise ValueError('Selected checkpoint source/target/configuration mismatch')
        return model
