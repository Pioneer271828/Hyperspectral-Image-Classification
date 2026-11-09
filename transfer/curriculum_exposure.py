"""Frozen class-first exposure gates; never redraw or extend an unlucky budget."""
from collections import Counter
from copy import deepcopy
import numpy as np
from model.research_config import digest
from transfer.observation_policy import scene_identity
from transfer.development_training import sparse_labels
from transfer.tile_sampling import SupportTileSampler


def _value(scene,record,updates,seed,minimum_class_blocks,minimum_support_visits):
    identity=scene_identity(scene,record)
    if type(updates) is not int or updates<1 or type(seed) is not int or not 0<=seed<2**32:
        raise ValueError('Positive update budget and explicit uint32 sampler seed required')
    classes=record['classes'];keys={str(c) for c in classes}
    if (not isinstance(minimum_class_blocks,dict) or set(minimum_class_blocks)!=keys
            or any(type(n) is not int or n<1 for n in minimum_class_blocks.values())
            or type(minimum_support_visits) is not int or minimum_support_visits<1):
        raise ValueError('Explicit positive exposure requirements for every class/support required')
    sampler=SupportTileSampler(scene.plan,record['roles'],sparse_labels(record,scene.plan,'train_support'),classes,'class_first')
    support=record['train_support'];width=scene.plan['shape'][1];members={}
    for tile in sampler.tiles:
        y0,y1,x0,x1=tile['box']
        members[tile['id']]=[i for i in support['indices'] if y0<=i//width<y1 and x0<=i%width<x1]
    rng=np.random.default_rng(seed);events=[]
    for step in range(1,updates+1):
        tile,anchor=sampler.draw(rng)
        events.append(dict(step=step,tile_id=tile['id'],anchor=anchor,support_ids=members[tile['id']].copy()))
    value=dict(schema_version=1,identity=identity,support=deepcopy(support),classes=classes.copy(),updates=updates,seed=seed,
        minimum_class_blocks=deepcopy(minimum_class_blocks),minimum_support_visits=minimum_support_visits,
        sampler=deepcopy(sampler.metadata),events=events,exposure_kind='raw_support_visits_not_weighted_loss',
        selection='unchanged_full_validation_nll',resampling='forbidden')
    value['plan_hash']=digest(value)
    return value


class ExposurePlan:
    """Reconstruct the exact original sampler sequence at load, before use.

    The expected hash must come from the caller's frozen registration, not the
    untrusted loaded payload. This gate checks exposure, not optimizer execution.
    """
    def __init__(self,scene,record,value,*,expected_plan_hash):
        keys={'schema_version','identity','support','classes','updates','seed','minimum_class_blocks','minimum_support_visits',
              'sampler','events','exposure_kind','selection','resampling','plan_hash'}
        if not isinstance(value,dict) or set(value)!=keys or value['plan_hash']!=expected_plan_hash:
            raise ValueError('Registered exposure plan hash/schema mismatch')
        expected=_value(scene,record,*(value[k] for k in ('updates','seed','minimum_class_blocks','minimum_support_visits')))
        if value!=expected:raise ValueError('Exposure plan differs from original support/sampler sequence')
        self._value=deepcopy(expected)

    @classmethod
    def freeze(cls,scene,record,*,updates,seed,minimum_class_blocks,minimum_support_visits):
        value=_value(scene,record,updates,seed,minimum_class_blocks,minimum_support_visits)
        return cls(scene,record,value,expected_plan_hash=value['plan_hash'])

    @property
    def plan_hash(self):return self._value['plan_hash']
    @property
    def updates(self):return self._value['updates']
    def to_dict(self):return deepcopy(self._value)

    def event(self,step):
        if type(step) is not int or not 1<=step<=self.updates:raise ValueError('Step outside frozen exposure budget')
        return deepcopy(self._value['events'][step-1])

    def exposure(self,completed=None):
        n=self.updates if completed is None else completed
        if type(n) is not int or not 0<=n<=self.updates:raise ValueError('Invalid completed exposure steps')
        v=self._value;lookup=dict(zip(v['support']['indices'],v['support']['labels']))
        visits=Counter();blocks=Counter();anchors=Counter();tiles=Counter()
        for event in v['events'][:n]:
            visits.update(event['support_ids']);blocks.update(set(lookup[i] for i in event['support_ids']))
            anchors[event['anchor']]+=1;tiles[event['tile_id']]+=1
        by_class={str(c):sum(visits[i] for i,l in lookup.items() if l==c) for c in v['classes']}
        class_deficits={str(c):max(0,v['minimum_class_blocks'][str(c)]-blocks[c]) for c in v['classes']}
        pixel_deficits={str(i):max(0,v['minimum_support_visits']-visits[i]) for i in lookup}
        return dict(completed_updates=n,class_blocks={str(c):blocks[c] for c in v['classes']},class_label_visits=by_class,
            support_visits={str(i):visits[i] for i in lookup},anchor_counts={str(c):anchors[c] for c in v['classes']},
            tile_counts={str(t):tiles[t] for t in v['sampler']['tile_ids']},class_deficits=class_deficits,support_deficits=pixel_deficits,
            meets_minimum=not any(class_deficits.values()) and not any(pixel_deficits.values()),
            exposure_kind=v['exposure_kind'])

    def require_eligible(self):
        result=self.exposure()
        if not result['meets_minimum']:raise ValueError('Frozen budget misses preregistered exposure; no automatic redraw/extension')
        return result

    def progress(self,completed):
        result=self.exposure(completed)
        value=dict(schema_version=1,plan_hash=self.plan_hash,completed_updates=completed,
            prefix_hash=digest(self._value['events'][:completed]),exposure=result)
        value['progress_hash']=digest(value)
        return value

    def resume_step(self,progress):
        if not isinstance(progress,dict) or type(progress.get('completed_updates')) is not int:raise ValueError('Invalid exposure progress')
        n=progress['completed_updates']
        if progress!=self.progress(n):raise ValueError('Exposure progress/prefix/identity mismatch')
        return None if n==self.updates else n+1


class ExposureLedger:
    """Record caller-reported completed events, after successful optimizer steps.

    Does not inspect the optimizer and cannot by itself prove actual training.
    Independent model/optimizer/gradient acceptance is still required.
    """
    def __init__(self,plan,progress=None):
        if not isinstance(plan,ExposurePlan):raise ValueError('Frozen exposure plan required')
        self.plan=plan;self.completed=0
        if progress is not None:
            plan.resume_step(progress);self.completed=progress['completed_updates']

    def append(self,event):
        if self.completed==self.plan.updates:raise ValueError('Completed budget cannot receive more events')
        expected=self.plan.event(self.completed+1)
        if not isinstance(event,dict) or event!=expected or digest(event)!=digest(expected):raise ValueError('Exposure event reordered, repeated or mismatched')
        self.completed+=1
        return self.snapshot()

    def snapshot(self):return self.plan.progress(self.completed)

    def finish(self):
        if self.completed!=self.plan.updates:raise ValueError('Incomplete update budget')
        self.plan.require_eligible()
        return dict(status='passed_exposure_only',optimizer_execution_verified=False,progress=self.snapshot())
