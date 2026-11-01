"""Separate validation-only release; leaves the zero-validation G3 contract unchanged."""
import numpy as np
from loadData.transfer_protocol import _pool, _validate_record
from model.research_config import digest


def make_validation(labels, plan, support, per_class=5, seed=3031):
    _validate_record(support, plan)
    if type(per_class) is not int or per_class<=0 or type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('Positive budget and uint32 seed required')
    if list(np.shape(labels))!=plan['shape']:
        raise ValueError('Label shape mismatch')
    pool=_pool(plan,'val')
    values=np.asarray(labels).reshape(-1)[pool]
    if not np.isfinite(values).all() or not np.equal(values,np.floor(values)).all() or not np.isin(values,[0,*support['classes']]).all():
        raise ValueError('Invalid validation labels')
    rng=np.random.default_rng(seed)
    chosen=[];counts={}
    for cls in support['classes']:
        candidates=pool[values==cls];counts[str(cls)]=len(candidates)
        if len(candidates)<per_class:
            raise ValueError(f'Insufficient validation candidates for class {cls}: {len(candidates)}; no redraw')
        chosen.extend(rng.permutation(candidates)[:per_class].tolist())
    chosen=sorted(chosen)
    result={'schema_version':1,'kind':'independent_target_validation','partition':'val',
            'plan_hash':plan['plan_hash'],'support_hash':support['support_hash'],
            'classes':support['classes'],'per_class':per_class,'seed':seed,'indices':chosen,
            'labels':np.asarray(labels).reshape(-1)[chosen].astype(int).tolist(),'candidate_counts':counts,
            'training_labels':len(support['indices']),'validation_labels':len(chosen),
            'total_training_selection_labels':len(support['indices'])+len(chosen),
            'selection':'minimum_validation_nll_then_earliest_step',
            'access':'selection_only_never_optimizer_or_train_statistics'}
    if set(chosen)&set(support['indices']):
        raise ValueError('Training and validation overlap')
    result['validation_hash']=digest(result)
    return result
