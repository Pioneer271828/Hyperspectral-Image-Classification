"""Role-isolated development adaptation; does not evaluate external test."""
from pathlib import Path
import numpy as np
import torch
from loadData.spatial_tiles import validate_tile_plan,tile_slice
from model.research_config import digest
from model.checkpoint_io import save_model_checkpoint
from research_runtime import write_json,hash_file
from transfer.g3_training import Scene,build_target,_optimizer,_update,_forward


def validate(record,plan):
    validate_tile_plan(plan)
    if record['validation_hash']!=digest({k:v for k,v in record.items() if k!='validation_hash'}) or record['parent_plan_hash']!=plan['plan_hash']:
        raise ValueError('Development identity mismatch')
    if len(record['roles'])!=len(plan['tiles']):raise ValueError('Role count mismatch')
    for role,tile in zip(record['roles'],plan['tiles']):
        if {k:v for k,v in role.items() if k!='role'}!=tile:raise ValueError('Role geometry altered')
        allowed=('devtrain','devval') if tile['partition']=='train' else ('test',) if tile['partition']=='test' else ('unused',)
        if role['role'] not in allowed:raise ValueError('Invalid role')
    if record['classes']!=list(range(1,len(record['classes'])+1)):raise ValueError('Invalid classes')
    width=plan['shape'][1];seen=set()
    for name,role,budget in [('train_support','devtrain',10),('validation_support','devval',5)]:
        support=record[name];ids=support['indices'];labels=support['labels']
        if ids!=sorted(set(ids)) or len(ids)!=len(labels) or seen.intersection(ids):raise ValueError('Support overlap/order')
        seen.update(ids)
        if any(labels.count(c)!=budget for c in record['classes']) or len(ids)!=budget*len(record['classes']):raise ValueError('Invalid label budget')
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


def evaluate_validation(model,scene,record):
    labels=sparse_labels(record,scene.plan,'validation_support');total=correct=0;nll=0.
    model.eval()
    with torch.no_grad():
        for tile in record['roles']:
            if tile['role']!='devval':continue
            target=labels[tile_slice(tile)].reshape(-1);valid=target>0
            if not valid.any():continue
            original=scene.plan['tiles'][tile['id']]
            output=_forward(model,scene,original)[0].detach().cpu().numpy()
            target=target[valid]-1;probs=output[valid]
            correct+=int((probs.argmax(1)==target).sum());total+=len(target)
            nll-=float(np.log(np.maximum(probs[np.arange(len(target)),target],1e-8)).sum())
    if total!=45:raise ValueError('Validation label budget differs from45')
    return {'nll':nll/total,'oa':correct/total,'count':total}


def adapt(scene,record,base,folder,*,method='target_scratch_full',source=None,steps=2,val_every=1,device='cpu',seed=2036):
    validate(record,scene.plan)
    if type(steps) is not int or steps<=0 or type(val_every) is not int or val_every<=0:raise ValueError('Positive budgets required')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    model,config=build_target(base,scene,method,source,model_seed=seed,head_seed=seed+2,device=device)
    optimizer=_optimizer(model,config);labels=sparse_labels(record,scene.plan,'train_support')
    tiles=[scene.plan['tiles'][t['id']] for t in record['roles'] if t['role']=='devtrain' and (labels[tile_slice(t)]>0).any()]
    rng=np.random.default_rng(seed+4);torch.manual_seed(seed);history=[];trace=[];best=float('inf')
    model.training_config={'development_hash':record['validation_hash'],'statistics_hash':scene.stats['statistics_hash'],
                           'method':method,'selection':record['selection'],'training_labels':90,'validation_labels':45}
    for step in range(1,steps+1):
        tile=tiles[int(rng.integers(len(tiles)))];loss=_update(model,scene,tile,labels[tile_slice(tile)],config,optimizer,head_only=method=='source_pretrained_head_only')
        trace.append({'step':step,'tile_id':tile['id'],'loss':loss})
        if step%val_every==0 or step==steps:
            val=evaluate_validation(model,scene,record)
            if not np.isfinite(val['nll']):raise ValueError('Nonfinite validation')
            history.append({'step':step,**val})
            if val['nll']<best:
                best=val['nll'];model.training_config['selected_step']=step
                save_model_checkpoint(model,folder/'model.pt')
    result={'status':'completed','steps':steps,'trace':trace,'validation':history,
            'selected_step':min(history,key=lambda x:x['nll'])['step'],'test_evaluated':False,
            'development_hash':record['validation_hash'],'checkpoint_sha256':hash_file(folder/'model.pt'),
            'sidecar_sha256':hash_file(folder/'model.pt.json')}
    write_json(folder/'result.json',result);write_json(folder/'statistics.json',scene.stats)
    return result
