"""CUR-O: reorder already frozen observations without changing their multiset."""
from collections import Counter
from copy import deepcopy
import math
import json
import numpy as np
from model.research_config import digest
from transfer.observation_policy import ObservationPolicy

ORDERS=('fixed','random','easy_to_hard','hard_to_easy')


def _permutation(source,severity,order,seed):
    if order not in ORDERS or type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('Explicit order and uint32 seed required')
    keys=[digest(r['config']) for r in source['training']]
    if (not isinstance(severity,dict) or set(severity)!=set(keys)
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in severity.values())):
        raise ValueError('Exact finite preregistered config severity map required')
    indices=list(range(source['updates']))
    if order=='random':indices=np.random.default_rng(seed).permutation(indices).tolist()
    if order in ('easy_to_hard','hard_to_easy'):
        indices.sort(key=lambda i:severity[keys[i]],reverse=order=='hard_to_easy')
    return indices


def _value(source,severity,order,seed):
    if 'curriculum' in source or source.get('selection_condition')!='full':raise ValueError('Unmodified full-selection source required')
    # Metadata dataclasses contain tuples; freeze a canonical JSON representation
    # so disk recovery is structurally identical, not merely hash-equivalent.
    source=json.loads(json.dumps(ObservationPolicy(source).to_dict()))
    if source.get('training_application','real_subset')!='real_subset':raise ValueError('Real-band subsets only')
    permutation=_permutation(source,severity,order,seed)
    value=deepcopy(source);value.pop('policy_hash')
    value['training']=[deepcopy(source['training'][i]) for i in permutation]
    value['schedule_rule']='CUR-O frozen observation permutation; sampler order is independent'
    value['curriculum']=dict(schema_version=1,source_policy=source,severity=deepcopy(severity),order=order,seed=seed,permutation=permutation)
    value['policy_hash']=digest(value)
    return value


class CurriculumPolicy(ObservationPolicy):
    """Serialized schedules are reconstructed from the source before acceptance.

    Severity means a declared observation condition, not learned sample difficulty.
    Equal marginal exposure does not guarantee equal condition-by-class/tile exposure.
    """
    def __init__(self,value):
        if not isinstance(value,dict) or not isinstance(value.get('curriculum'),dict):raise ValueError('Curriculum record required')
        c=value['curriculum']
        if set(c)!={'schema_version','source_policy','severity','order','seed','permutation'} or c['schema_version']!=1:
            raise ValueError('Invalid curriculum schema')
        expected=_value(c['source_policy'],c['severity'],c['order'],c['seed'])
        if value!=expected:raise ValueError('Curriculum differs from frozen permutation/validation/budget')
        super().__init__(expected)

    @classmethod
    def freeze(cls,source,*,order='off',severity=None,seed=0):
        if not isinstance(source,ObservationPolicy):raise ValueError('Frozen M1 source policy required')
        if order=='off':
            if severity is not None or seed!=0 or 'curriculum' in source.to_dict():raise ValueError('Disabled policy takes no schedule controls')
            return ObservationPolicy(source.to_dict())
        return cls(_value(source.to_dict(),severity,order,seed))

    def exposure(self,completed=None):
        n=self._value['updates'] if completed is None else completed
        if type(n) is not int or not 0<=n<=self._value['updates']:raise ValueError('Invalid completed updates')
        rows=self._value['training'][:n];severity=self._value['curriculum']['severity']
        config=Counter(digest(r['config']) for r in rows)
        return dict(updates=n,by_config={key:config[key] for key in sorted(severity)},
                    by_severity={str(v):sum(config[k] for k in config if severity[k]==v) for v in sorted(set(severity.values()))},
                    total_retained_band_exposures=sum(len(r['selected_positions']) for r in rows),
                    record_multiset_hash=digest(sorted(r['record_hash'] for r in rows)))

    def progress(self,completed):
        exposure=self.exposure(completed)
        value=dict(schema_version=1,policy_hash=self.policy_hash,completed_updates=completed,
                   prefix_hash=digest(self._value['training'][:completed]),exposure=exposure)
        value['progress_hash']=digest(value)
        return value

    def resume_step(self,progress):
        if not isinstance(progress,dict) or type(progress.get('completed_updates')) is not int:raise ValueError('Invalid progress')
        if progress!=self.progress(progress['completed_updates']):raise ValueError('Progress/policy/exposure mismatch')
        return None if progress['completed_updates']==self._value['updates'] else progress['completed_updates']+1
