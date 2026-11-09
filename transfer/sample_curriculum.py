"""CUR-S: frozen train-support scores and positive class-protected soft weights."""
from copy import deepcopy
import math
import time
import numpy as np
import torch
from loadData.spatial_tiles import tile_slice
from model.research_config import digest
from model.task_losses import probability_loss_components
from transfer.observation_policy import scene_identity
from transfer.development_training import sparse_labels
from transfer.g3_training import _forward,state_hash


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _scores(value):
    keys={'schema_version','identity','support','classes','teacher_state','score_kind','losses','forward_tiles','seconds','score_hash'}
    if not isinstance(value,dict) or set(value)!=keys or value['schema_version']!=1:raise ValueError('Invalid teacher record')
    if value['score_hash']!=digest({k:v for k,v in value.items() if k!='score_hash'}):raise ValueError('Teacher hash mismatch')
    if not isinstance(value['support'],dict) or set(value['support'])!={'indices','labels'}:raise ValueError('Invalid support record')
    ids=value['support']['indices'];labels=value['support']['labels'];classes=value['classes'];losses=value['losses']
    if any(not isinstance(v,list) for v in (ids,labels,classes,losses)):raise ValueError('List-valued support/scores required')
    if (not ids or ids!=sorted(set(ids)) or any(type(i) is not int or i<0 for i in ids)
            or len(ids)!=len(labels) or len(ids)!=len(losses) or not classes or classes!=list(range(1,len(classes)+1))
            or any(type(c) is not int or c not in classes for c in labels) or set(labels)!=set(classes)
            or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in losses)
            or not _sha(value['teacher_state']) or value['score_kind']!='frozen_teacher_train_support_nll'
            or type(value['forward_tiles']) is not int or value['forward_tiles']<1
            or type(value['seconds']) not in (int,float) or not math.isfinite(value['seconds']) or value['seconds']<0):
        raise ValueError('Invalid training support/scores/cost')
    return deepcopy(value)


def score_training_support(model,scene,record,*,expected_teacher_state):
    identity=scene_identity(scene,record)
    if (not _sha(expected_teacher_state) or state_hash(model)!=expected_teacher_state
            or any(m.training for m in model.modules()) or model.backbone.adapter.cache_attention
            or any(v.device.type!='cpu' or not torch.isfinite(v).all() or (v.is_floating_point() and v.dtype!=torch.float32) for v in model.state_dict().values())):
        raise ValueError('Frozen CPU FP32 teacher with cache disabled required')
    support=record['train_support'];lookup=dict(zip(support['indices'],support['labels']));losses={};calls=0;t=time.perf_counter()
    with torch.no_grad():
        for tile in scene.plan['tiles']:
            if record['roles'][tile['id']]['role']!='devtrain':continue
            y0,y1,x0,x1=tile['box'];yy,xx=np.mgrid[y0:y1,x0:x1];ids=(yy*scene.plan['shape'][1]+xx).ravel()
            selected=[(j,int(i)) for j,i in enumerate(ids) if i in lookup]
            if not selected:continue
            p=_forward(model,scene,tile)[0];calls+=1
            if not torch.isfinite(p).all():raise ValueError('Nonfinite teacher probabilities')
            for j,i in selected:losses[i]=float(-p[j,lookup[i]-1].clamp(min=1e-8,max=1.).log())
    if set(losses)!=set(lookup) or state_hash(model)!=expected_teacher_state:raise ValueError('Teacher support/state changed')
    value=dict(schema_version=1,identity=identity,support=deepcopy(support),classes=record['classes'].copy(),teacher_state=expected_teacher_state,
        score_kind='frozen_teacher_train_support_nll',losses=[losses[i] for i in support['indices']],forward_tiles=calls,seconds=time.perf_counter()-t)
    value['score_hash']=digest(value)
    return _scores(value)


def _ranks(scores,order,seed):
    if order not in ('balanced','random','easy_to_hard','hard_to_easy') or type(seed) is not int or not 0<=seed<2**32:raise ValueError('Invalid curriculum order/seed')
    ranks={};ids=scores['support']['indices'];labels=scores['support']['labels'];losses=scores['losses']
    for cls in scores['classes']:
        indices=[j for j,c in enumerate(labels) if c==cls]
        if order in ('easy_to_hard','hard_to_easy'):
            indices.sort(key=lambda j:((1 if order=='easy_to_hard' else -1)*losses[j],ids[j]))
        if order=='random':
            rng=np.random.default_rng(int(digest({'seed':seed,'class':cls})[:16],16));indices=rng.permutation(indices).tolist()
        for rank,j in enumerate(indices):ranks[str(ids[j])]=rank
    return ranks


def _value(scores,updates,order,seed,start_fraction,full_at,minimum_weight):
    scores=_scores(scores)
    if type(updates) is not int or updates<1 or type(full_at) is not int or not 1<=full_at<=updates:raise ValueError('Invalid update/pacing budget')
    for v in (start_fraction,minimum_weight):
        if type(v) not in (int,float) or not math.isfinite(v) or not 0<v<=1:raise ValueError('Positive fraction and weight floor required')
    value=dict(schema_version=1,scores=scores,updates=updates,order=order,seed=seed,start_fraction=start_fraction,full_at=full_at,
        minimum_weight=minimum_weight,ranks=_ranks(scores,order,seed),selection='unchanged_full_validation_nll',sampler='class_first')
    value['policy_hash']=digest(value)
    return value


class SampleCurriculum:
    def __init__(self,value):
        keys={'schema_version','scores','updates','order','seed','start_fraction','full_at','minimum_weight','ranks','selection','sampler','policy_hash'}
        if not isinstance(value,dict) or set(value)!=keys:raise ValueError('Invalid sample curriculum record')
        expected=_value(*(value[k] for k in ('scores','updates','order','seed','start_fraction','full_at','minimum_weight')))
        if value!=expected:raise ValueError('Sample curriculum identity/ranks mismatch')
        self._value=deepcopy(expected)

    @classmethod
    def freeze(cls,scores,*,updates,order,seed,start_fraction,full_at,minimum_weight):
        return cls(_value(scores,updates,order,seed,start_fraction,full_at,minimum_weight))

    def to_dict(self):return deepcopy(self._value)
    @property
    def policy_hash(self):return self._value['policy_hash']

    def validate(self,scene,record):
        s=self._value['scores']
        if scene_identity(scene,record)!=s['identity'] or record['train_support']!=s['support'] or record['classes']!=s['classes']:
            raise ValueError('Training support/scene identity mismatch')

    def weights(self,ids,*,step):
        v=self._value;s=v['scores'];known=dict(zip(s['support']['indices'],s['support']['labels']))
        if type(step) is not int or not 1<=step<=v['updates']:raise ValueError('Step outside frozen budget')
        if not isinstance(ids,(list,tuple)) or not ids or any(type(i) is not int or i not in known for i in ids) or len(ids)!=len(set(ids)):
            raise ValueError('Unique released training IDs required')
        fraction=1. if v['full_at']==1 else min(1.,v['start_fraction']+(1-v['start_fraction'])*(step-1)/(v['full_at']-1))
        counts={c:s['support']['labels'].count(c) for c in s['classes']}
        raw=np.array([1. if v['order']=='balanced' or v['ranks'][str(i)]<math.ceil(fraction*counts[known[i]]) else v['minimum_weight'] for i in ids],dtype=np.float64)
        # Preserve each present class's original count, including a one-pixel rare
        # class: compute_loss clamps class counts at one, so raw .1 is insufficient.
        result=raw.copy()
        for cls in sorted({known[i] for i in ids}):
            selected=np.array([known[i]==cls for i in ids]);result[selected]*=selected.sum()/raw[selected].sum()
        return result


def curriculum_loss(outputs,scene,record,tile,*,policy=None,step=None,class_balance=False,**loss_kwargs):
    scene_identity(scene,record)
    if (type(tile.get('id')) is not int or not 0<=tile['id']<len(scene.plan['tiles'])
            or tile!=scene.plan['tiles'][tile['id']] or record['roles'][tile['id']]['role']!='devtrain'):
        raise ValueError('Only original devtrain tile allowed')
    y0,y1,x0,x1=tile['box'];yy,xx=np.mgrid[y0:y1,x0:x1];ids=(yy*scene.plan['shape'][1]+xx).ravel()
    target=sparse_labels(record,scene.plan,'train_support')[tile_slice(tile)].ravel();selected=target>0
    if not selected.any():raise ValueError('Released training support required')
    if not isinstance(outputs,(tuple,list)) or len(outputs)<3 or any(not isinstance(p,torch.Tensor) for p in outputs[:3]):raise ValueError('Three heads required')
    device=outputs[0].device
    if device.type!='cpu' or any(p.device!=device or p.dtype!=torch.float32 or p.shape!=(len(target),scene.classes) or not torch.isfinite(p).all() for p in outputs[:3]):
        raise ValueError('Finite CPU FP32 three-head probabilities required')
    labels=torch.tensor(target,dtype=torch.long,device=device);mask=(labels>0)[:,None].expand(-1,scene.classes).float()
    onehot=F_onehot(labels,scene.classes)*mask
    if policy is not None:
        if not isinstance(policy,SampleCurriculum) or class_balance is not True:raise ValueError('Curriculum requires explicit class-balanced control')
        policy.validate(scene,record);weights=policy.weights(ids[selected].tolist(),step=step)
        allweights=torch.zeros(len(target),dtype=torch.float32,device=device);allweights[selected]=torch.tensor(weights,dtype=torch.float32)
        mask=mask*allweights[:,None]
    elif step is not None:raise ValueError('Disabled curriculum takes no progress')
    return probability_loss_components(outputs,onehot,mask,class_balance=class_balance,**loss_kwargs)


def F_onehot(labels,classes):
    return torch.nn.functional.one_hot((labels-1).clamp_min(0),classes).float()
