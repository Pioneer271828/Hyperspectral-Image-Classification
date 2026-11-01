"""Oracle-assisted deterministic whole-tile development split, protocol preparation only."""
import numpy as np
from loadData.spatial_tiles import validate_tile_plan, tile_slice
from model.research_config import digest


def make_development(labels,plan,classes,train_per_class=10,val_per_class=5):
    validate_tile_plan(plan)
    if classes!=list(range(1,len(classes)+1)) or not classes or any(type(n) is not int or n<=0 for n in (train_per_class,val_per_class)):
        raise ValueError('Explicit contiguous classes and positive budgets required')
    labels=np.asarray(labels)
    if list(labels.shape)!=plan['shape']:raise ValueError('Shape mismatch')
    tiles=[t for t in plan['tiles'] if t['partition']=='train']
    counts={}
    for tile in tiles:
        values=labels[tile_slice(tile)]
        if not np.isfinite(values).all() or not np.equal(values,np.floor(values)).all() or not np.isin(values,[0,*classes]).all():
            raise ValueError('Invalid original train labels')
        counts[tile['id']]=np.bincount(values.astype(int).ravel(),minlength=len(classes)+1)[1:]
    if not counts:raise ValueError('No training tiles')
    remaining=sum(counts.values());selected=np.zeros(len(classes),dtype=int);held=[]
    while (selected<val_per_class).any():
        deficit=np.maximum(val_per_class-selected,0)
        choices=[(int(np.minimum(v,deficit).sum()),-i,i) for i,v in counts.items()
                 if i not in held and (remaining-v>=train_per_class).all()]
        if not choices or max(choices)[0]<=0:
            raise ValueError('Cannot cover validation while preserving training budget; no fallback')
        index=max(choices)[2];held.append(index);selected+=counts[index];remaining-=counts[index]
    held=sorted(held)
    roles=[{**t,'role':('devval' if t['id'] in held else 'devtrain') if t['partition']=='train'
            else 'test' if t['partition']=='test' else 'unused'} for t in plan['tiles']]
    width=plan['shape'][1]
    def support(role,budget,seed):
        pieces=[]
        for t in roles:
            if t['role']!=role:continue
            y0,y1,x0,x1=t['box'];yy,xx=np.mgrid[y0:y1,x0:x1]
            pieces.extend((yy*width+xx).ravel().tolist())
        pool=np.array(sorted(pieces));values=labels.reshape(-1)[pool]
        rng=np.random.default_rng(seed);ids=[]
        for c in classes:
            candidates=pool[values==c]
            if len(candidates)<budget:raise ValueError('Insufficient support')
            ids.extend(rng.permutation(candidates)[:budget].tolist())
        ids=sorted(ids)
        return {'indices':ids,'labels':labels.reshape(-1)[ids].astype(int).tolist(),'seed':seed}
    train=support('devtrain',train_per_class,2027);val=support('devval',val_per_class,3031)
    assert not set(train['indices'])&set(val['indices'])
    result={'schema_version':1,'kind':'oracle_spatial_development_v1','parent_plan_hash':plan['plan_hash'],
        'shape':plan['shape'],'classes':classes,'roles':roles,'heldout_tile_ids':held,
        'training_candidates':remaining.tolist(),'validation_candidates':selected.tolist(),
        'train_support':train,'validation_support':val,'training_labels':len(train['indices']),
        'validation_labels':len(val['indices']),'total_training_selection_labels':len(train['indices'])+len(val['indices']),
        'selection':'minimum_validation_nll_then_earliest_step','test_unchanged':True,
        'rule':'greedy maximum capped validation deficit gain; preserve train minimum; tie lowest tile ID',
        'oracle_disclosure':'full original train class counts used for split construction; not label-blind',
        'normalization_contract':'fit devtrain tiles only; no devval/test context in adaptation',
        'runner_compatible':False}
    result['validation_hash']=digest(result)
    return result
