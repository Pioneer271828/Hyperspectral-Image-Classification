"""Spatially isolated G3 support manifests and explicit target-label release."""
import copy
import re
import numpy as np
from model.research_config import digest
from loadData.spatial_tiles import validate_tile_plan,tile_slice


def _pool(plan,partition):
    validate_tile_plan(plan)
    pieces=[]
    for tile in plan['tiles']:
        if tile['partition']==partition:
            y0,y1,x0,x1=tile['box'];yy,xx=np.mgrid[y0:y1,x0:x1]
            pieces.append((yy*plan['shape'][1]+xx).reshape(-1))
    return np.sort(np.concatenate(pieces)) if pieces else np.array([],dtype=np.int64)


def _classes(classes):
    classes=tuple(classes)
    if not classes or any(type(c) is not int or c<=0 for c in classes) or list(classes)!=sorted(set(classes)):
        raise ValueError('Explicit sorted unique positive target classes required')
    return classes


def make_support(labels,plan,classes,per_class=10,seed=2027):
    """Stratify only the target train pool; never inspect val/test label values."""
    classes=_classes(classes)
    if type(per_class) is not int or per_class<=0 or type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('Positive per-class budget and uint32 seed required')
    labels=np.asarray(labels)
    if list(labels.shape)!=plan['shape']: raise ValueError('Target label shape mismatch')
    pool=_pool(plan,'train');values=labels.reshape(-1)[pool]
    if not np.isfinite(values).all() or not np.equal(values,np.floor(values)).all() or not np.isin(values,(0,*classes)).all():
        raise ValueError('Invalid labels inside target support pool')
    rng=np.random.default_rng(seed);selected=[];counts={}
    for cls in classes:
        candidates=pool[values==cls];counts[str(cls)]=len(candidates)
        if len(candidates)<per_class: raise ValueError('Insufficient target train support for class '+str(cls)+'; no split redraw')
        selected.extend(rng.permutation(candidates)[:per_class].tolist())
    selected=sorted(selected);chosen=labels.reshape(-1)[selected].astype(np.int64).tolist()
    record={'schema_version':1,'kind':'spatial_target_support','shape':list(labels.shape),'plan_hash':plan['plan_hash'],
            'classes':list(classes),'per_class':per_class,'seed':seed,'candidate_partition':'train',
            'indices':selected,'labels':chosen,'candidate_counts':counts,'target_validation_labels':0,
            'selected_label_hash':digest({'indices':selected,'labels':chosen})}
    record['support_hash']=digest(record)
    return record


def validate_support(record,labels,plan):
    if record!=make_support(labels,plan,record['classes'],record['per_class'],record['seed']):
        raise ValueError('Support record differs from deterministic target train selection')


def _validate_record(record,plan):
    if (record.get('schema_version')!=1 or record.get('kind')!='spatial_target_support'
            or record.get('support_hash')!=digest({k:v for k,v in record.items() if k!='support_hash'})
            or record.get('plan_hash')!=plan['plan_hash'] or record.get('shape')!=plan['shape']
            or record.get('candidate_partition')!='train' or record.get('target_validation_labels')!=0):
        raise ValueError('Support identity/hash mismatch')
    classes=_classes(record['classes']);ids=record['indices'];values=record['labels']
    if (any(type(i) is not int for i in ids) or ids!=sorted(set(ids)) or len(ids)!=len(values)
            or not np.isin(ids,_pool(plan,'train')).all() or any(type(v) is not int or v not in classes for v in values)
            or type(record['per_class']) is not int or record['per_class']<=0
            or any(values.count(c)!=record['per_class'] for c in classes)
            or record['selected_label_hash']!=digest({'indices':ids,'labels':values})):
        raise ValueError('Support indices/labels/budget invalid')


def target_labels_for_tile(labels,plan,support,tile_id,*,phase,final_checkpoint_sha256=None):
    """Guard against accidental phase mixing; this Python API is not a security sandbox."""
    _validate_record(support,plan)
    if list(np.shape(labels))!=plan['shape'] or type(tile_id) is not int or not 0<=tile_id<len(plan['tiles']):
        raise ValueError('Target labels/tile identity mismatch')
    tile=plan['tiles'][tile_id]
    if phase=='adapt' and tile['partition']=='train':
        y0,y1,x0,x1=tile['box'];result=np.zeros((y1-y0,x1-x0),dtype=np.int64)
        for index,value in zip(support['indices'],support['labels']):
            y,x=divmod(index,plan['shape'][1])
            if y0<=y<y1 and x0<=x<x1:
                if labels[y,x]!=value: raise ValueError('Selected support label changed')
                result[y-y0,x-x0]=value
        return result
    if (phase=='final_test' and tile['partition']=='test' and isinstance(final_checkpoint_sha256,str)
            and re.fullmatch('[0-9a-f]{64}',final_checkpoint_sha256)):
        return np.array(labels[tile_slice(tile)],copy=True)
    raise ValueError('Target labels unavailable for this phase/tile; validation budget is zero and test needs frozen checkpoint identity')


def validate_config(config):
    required={'schema_version','task_id','task_type','source_dataset','target_dataset','source_sensor','target_sensor',
              'class_policy','support_budgets_per_class','primary_support_per_class','target_validation_labels',
              'tile_size','gap','split_seed','support_seed','model_seed','target_head_seed','source_sampler_seed',
              'target_sampler_seed','source_updates','source_selection','target_updates','target_selection',
              'normalization','graph_mode','norm_type','methods','spectral_policy','status'}
    if set(config)!=required: raise ValueError('Unknown/missing transfer configuration fields')
    expected={'schema_version':1,'task_type':'G3_few_shot_transfer','source_dataset':'paviaU','target_dataset':'xuzhou',
              'source_sensor':'ROSIS','target_sensor':'HYSPEX_bundled_readme',
              'class_policy':'new_target_head_original_classes','support_budgets_per_class':[1,5,10],
              'primary_support_per_class':10,'target_validation_labels':0,'source_selection':'source_val_nll',
              'target_selection':'fixed_final_step','normalization':'separate_source_target_train_region_minmax',
              'graph_mode':'learned_label_free','norm_type':'group_norm',
              'methods':['target_scratch_full','source_pretrained_head_only','source_pretrained_full_finetune'],
              'spectral_policy':'full_delivered_bands_with_explicit_M1_metadata','status':'protocol_only_no_training'}
    if any(config[k]!=v for k,v in expected.items()): raise ValueError('First-task transfer contract changed')
    if not isinstance(config['task_id'],str) or not config['task_id']: raise ValueError('Task ID required')
    for key in ('split_seed','support_seed','model_seed','target_head_seed','source_sampler_seed','target_sampler_seed'):
        if type(config[key]) is not int or not 0<=config[key]<2**32: raise ValueError('Invalid task seed')
    if any(type(config[k]) is not int or config[k]<=0 for k in ('source_updates','target_updates','tile_size')) or type(config['gap']) is not int or config['gap']<0:
        raise ValueError('Invalid update/geometry budget')

def freeze_task(config,source_identity,target_identity,source_plan,target_plan,support,evidence):
    """Freeze one explicit G3 task; source and target class numbers are NOT a semantic mapping."""
    validate_config(config)
    if config['source_dataset']!=source_identity['name'] or config['target_dataset']!=target_identity['name'] or source_identity['name']==target_identity['name']:
        raise ValueError('Source/target identity mismatch')
    validate_tile_plan(source_plan);validate_tile_plan(target_plan);_validate_record(support,target_plan)
    if (support['per_class']!=config['primary_support_per_class'] or support['seed']!=config['support_seed']
            or config['target_validation_labels']!=0 or config['target_selection']!='fixed_final_step'
            or config['class_policy']!='new_target_head_original_classes'
            or support['classes']!=list(range(1,target_identity['class_count']+1))):
        raise ValueError('G3 label/selection contract mismatch')
    for name,plan in [('source',source_plan),('target',target_plan)]:
        identity=source_identity if name=='source' else target_identity
        if plan['shape']!=identity['shape'][:2] or any(plan[k]!=config[k] for k in ('tile_size','gap')) or plan['seed']!=config['split_seed']:
            raise ValueError('Spatial geometry differs from frozen task')
    if any(type(config[k]) is not int or config[k]<=0 for k in ('source_updates','target_updates')):
        raise ValueError('Explicit positive source/target update budgets required')
    if not re.fullmatch('[0-9a-f]{64}',evidence.get('target_sensor_readme_sha256','')) or evidence.get('target_sensor_readme_mentions_hyspex') is not True:
        raise ValueError('First task needs its bundled HYSPEX readme evidence')
    task={'schema_version':1,'kind':'frozen_G3_task','config':copy.deepcopy(config),
          'source_identity':copy.deepcopy(source_identity),'target_identity':copy.deepcopy(target_identity),
          'source_plan_hash':source_plan['plan_hash'],'target_plan_hash':target_plan['plan_hash'],
          'support_hash':support['support_hash'],'target_support_labels':len(support['indices']),
          'target_validation_labels':0,'target_total_training_selection_labels':len(support['indices']),
          'sensor_evidence':copy.deepcopy(evidence),'claims':{'zero_shot':False,'G2_unseen_family':False,
          'shared_semantic_classes':False,'target_train_unlabeled_context_allowed':True},
          'phase_access':{'source_pretrain':['source_train_inputs_labels'], 'source_selection':['source_val_inputs_labels'],
          'target_adapt':['target_train_inputs','target_support_labels'], 'target_selection':['fixed_step_no_target_validation'],
          'final_test':['target_test_tiles_only_after_frozen_checkpoint']}}
    task['task_hash']=digest(task);return task

