"""Role-isolated development adaptation; does not evaluate external test."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from loadData.spatial_tiles import validate_tile_plan,tile_slice
from model.research_config import digest
from model.coverage_config import coverage_trace
from model.checkpoint_io import save_model_checkpoint
from research_runtime import write_json,hash_file
from transfer.g3_training import Scene,build_target,_optimizer,_update,_forward,state_hash,head_hash
from transfer.tile_sampling import SupportTileSampler
from transfer.run_provenance import resolve_budget,run_fields,verify_run_artifacts
from transfer.multiscale_training import MultiscaleTrial


def development_counts(record):
    """Fixed per-class 10/5 contract; totals vary only with declared classes."""
    classes=record['classes']
    if (not isinstance(classes,list) or not 1<=len(classes)<=256 or
            any(type(c) is not int for c in classes) or classes!=list(range(1,len(classes)+1))):
        raise ValueError('Bounded contiguous positive classes required')
    training,validation=10*len(classes),5*len(classes)
    for key,expected in (('training_labels',training),('validation_labels',validation),
                         ('total_training_selection_labels',training+validation)):
        if type(record.get(key)) is not int or record[key]!=expected:
            raise ValueError('Declared total differs from fixed per-class development budget')
    return training,validation


def validate(record,plan):
    validate_tile_plan(plan)
    if record['validation_hash']!=digest({k:v for k,v in record.items() if k!='validation_hash'}) or record['parent_plan_hash']!=plan['plan_hash']:
        raise ValueError('Development identity mismatch')
    if len(record['roles'])!=len(plan['tiles']):raise ValueError('Role count mismatch')
    for role,tile in zip(record['roles'],plan['tiles']):
        if {k:v for k,v in role.items() if k!='role'}!=tile:raise ValueError('Role geometry altered')
        allowed=('devtrain','devval') if tile['partition']=='train' else ('test',) if tile['partition']=='test' else ('unused',)
        if role['role'] not in allowed:raise ValueError('Invalid role')
    development_counts(record)
    width=plan['shape'][1];seen=set()
    for name,role,budget in [('train_support','devtrain',10),('validation_support','devval',5)]:
        support=record[name];ids=support['indices'];labels=support['labels']
        if ids!=sorted(set(ids)) or len(ids)!=len(labels) or seen.intersection(ids):raise ValueError('Support overlap/order')
        seen.update(ids)
        if (any(type(c) is not int or c not in record['classes'] for c in labels) or
                any(labels.count(c)!=budget for c in record['classes']) or len(ids)!=budget*len(record['classes'])):
            raise ValueError('Invalid label budget')
        for index in ids:
            if type(index) is not int or not 0<=index<np.prod(plan['shape']):raise ValueError('Invalid index')
            y,x=divmod(index,width)
            if not any(t['role']==role and t['box'][0]<=y<t['box'][1] and t['box'][2]<=x<t['box'][3] for t in record['roles']):
                raise ValueError('Support escapes allowed role')


def prepare_scene(cube,bands,plan,record):
    validate(record,plan)
    if list(cube.shape[:2])!=plan['shape'] or cube.shape[-1]!=len(bands.band_ids):raise ValueError('Cube shape mismatch')
    chunks=[np.asarray(cube[tile_slice(t)]) for t in record['roles'] if t['role']=='devtrain']
    if not chunks or not all(np.isfinite(x).all() for x in chunks):raise ValueError('Invalid training inputs')
    low=np.min([x.min(axis=(0,1)) for x in chunks],axis=0)
    high=np.max([x.max(axis=(0,1)) for x in chunks],axis=0)
    stats={'schema_version':1,'plan_hash':plan['plan_hash'],'fit_partition':'train',
           'development_hash':record['validation_hash'],'fit_role':'devtrain','minimum':low.tolist(),'maximum':high.tolist()}
    stats['statistics_hash']=digest(stats)
    return Scene('xuzhou',cube,plan,bands,len(record['classes']),stats)


def sparse_labels(record,plan,name):
    result=np.zeros(plan['shape'],dtype=np.int64)
    support=record[name];result.reshape(-1)[support['indices']]=support['labels']
    return result


def evaluate_validation(model,scene,record,*,observation_policy=None,condition='full'):
    validate(record,scene.plan)
    _,validation_count=development_counts(record)
    if scene.classes!=len(record['classes']):raise ValueError('Scene/record class counts differ')
    if observation_policy is None and condition!='full':raise ValueError('Perturbation condition requires frozen policy')
    settings = getattr(model, 'training_config', {})
    if 'multiscale_metadata_hash' in settings and settings['multiscale_metadata_hash'] != scene.bands.fingerprint:
        raise ValueError('Multiscale validation requires the registered band metadata')
    expected=settings.get('observation_policy_hash')
    if expected is not None and (observation_policy is None or observation_policy.policy_hash!=expected):
        raise ValueError('Reload the matching frozen observation policy for this checkpoint')
    if observation_policy is not None:observation_policy.validate(scene,record)
    labels=sparse_labels(record,scene.plan,'validation_support');total=correct=0;nll=0.
    matrix=np.zeros((scene.classes,scene.classes),dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for tile in record['roles']:
            if tile['role']!='devval':continue
            target=labels[tile_slice(tile)].reshape(-1);valid=target>0
            if not valid.any():continue
            original=scene.plan['tiles'][tile['id']]
            extra={} if observation_policy is None else {'observation_view':observation_policy.view(scene,record,original,phase='validation',condition=condition)}
            output=_forward(model,scene,original,**extra)[0].detach().cpu().numpy()
            target=target[valid]-1;probs=output[valid]
            prediction=probs.argmax(1)
            matrix+=np.bincount(target*scene.classes+prediction,minlength=scene.classes**2).reshape(scene.classes,scene.classes)
            correct+=int((prediction==target).sum());total+=len(target)
            nll-=float(np.log(np.maximum(probs[np.arange(len(target)),target],1e-8)).sum())
    if total!=validation_count:raise ValueError('Fixed per-class validation label budget changed')
    return {'nll':nll/total,'oa':correct/total,'count':total,'confusion':matrix.tolist(),
            'support':matrix.sum(1).tolist(),'recall':(np.diag(matrix)/matrix.sum(1)).tolist()}


def adapt(scene,record,base,folder,*,method='target_scratch_full',source=None,steps=2,val_every=1,device='cpu',seed=2036,progress=None,sampling='uniform',observation_policy=None,multiscale=None):
    validate(record,scene.plan)
    if type(steps) is not int or steps<=0 or type(val_every) is not int or val_every<=0:raise ValueError('Positive budgets required')
    if observation_policy is not None:observation_policy.validate(scene,record,steps)
    if multiscale is not None and (not isinstance(multiscale, MultiscaleTrial) or method == 'source_pretrained_head_only'):
        raise ValueError('Multiscale trial requires explicit controls and a trainable backbone')
    model,config=build_target(base,scene,method,source,model_seed=seed,head_seed=seed+2,device=device)
    if multiscale is not None:
        model = multiscale.initialize(model, seed)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    config=resolve_budget(config,steps,val_every)
    initial_backbone=state_hash(model.backbone);initial_heads=head_hash(model)
    optimizer=_optimizer(model,config);labels=sparse_labels(record,scene.plan,'train_support')
    sampler=SupportTileSampler(scene.plan,record['roles'],labels,record['classes'],sampling)
    rng=np.random.default_rng(seed+4);torch.manual_seed(seed);history=[];trace=[];best=float('inf')
    model.training_config={'phase':'development','development_hash':record['validation_hash'],'statistics_hash':scene.stats['statistics_hash'],
                           'method':method,'selection':record['selection'],'training_labels':record['training_labels'],'validation_labels':record['validation_labels'],
                           'optimizer_budget':steps,'model_seed':seed,'head_seed':seed+2,'sampler_seed':seed+4,
                           'sampler':sampler.metadata}
    records=run_fields(config,phase='development',method=method,head_seed=seed+2,sampler_seed=seed+4,
                       val_every=val_every,selection=record['selection'],sampling=sampler.metadata,optimizer=optimizer,
                       coverage_metadata=scene.bands,model=model)
    model.training_config.update(records)
    policy_fields={} if observation_policy is None else {'observation_policy_hash':observation_policy.policy_hash,
                                                        'observation_selection_condition':'full'}
    model.training_config.update(policy_fields)
    trial_fields = {} if multiscale is None else {'multiscale': asdict(multiscale),
                                                'multiscale_metadata_hash': scene.bands.fingerprint}
    model.training_config.update(trial_fields)
    if observation_policy is not None:write_json(folder/'observation-policy.json',observation_policy.to_dict())
    for step in range(1,steps+1):
        tile,anchor=sampler.draw(rng)
        extra={} if observation_policy is None else {'observation_view':observation_policy.view(scene,record,tile,phase='training',step=step)}
        loss=_update(model,scene,tile,labels[tile_slice(tile)],config,optimizer,head_only=method=='source_pretrained_head_only',**extra)
        trace.append({'step':step,'tile_id':tile['id'],'loss':loss,**({'anchor_class':anchor} if anchor is not None else {}),
                      **({} if observation_policy is None else {'observation':observation_policy.training_record(step)}),
                      **coverage_trace(model,extra['observation_view'].bands if observation_policy is not None else scene.bands)})
        if step%val_every==0 or step==steps:
            val=evaluate_validation(model,scene,record,**({} if observation_policy is None else {'observation_policy':observation_policy}))
            if not np.isfinite(val['nll']):raise ValueError('Nonfinite validation')
            history.append({'step':step,**val})
            if progress:progress({'method':method,'budget':steps,'step':step,'validation':val})
            if val['nll']<best:
                best=val['nll'];model.training_config['selected_step']=step
                save_model_checkpoint(model,folder/'model.pt')
    result={'status':'completed','method':method,'selection':record['selection'],'steps':steps,'trace':trace,'validation':history,
            'selected_step':min(history,key=lambda x:x['nll'])['step'],'test_evaluated':False,
            'initial_backbone_hash':initial_backbone,'initial_head_hash':initial_heads,
            'final_backbone_hash':state_hash(model.backbone),'final_head_hash':head_hash(model),
            'development_hash':record['validation_hash'],'checkpoint_sha256':hash_file(folder/'model.pt'),
            'sidecar_sha256':hash_file(folder/'model.pt.json'),'sampler':sampler.metadata,
            **records,**policy_fields,**trial_fields}
    write_json(folder/'result.json',result);write_json(folder/'statistics.json',scene.stats)
    verify_run_artifacts(folder)
    return result
