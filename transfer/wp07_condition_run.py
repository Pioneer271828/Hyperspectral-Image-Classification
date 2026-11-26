"""Private provider/run adapter; use the existing fine16 phase executor and native lifecycle."""
from contextlib import contextmanager
from copy import deepcopy
from itertools import product
import numpy as np
import torch
from model.research_config import digest
from research_runtime import tensor_hash
from transfer import wp07_graph_range as graph
from transfer import wp07_fine16_training as parent
from transfer import native_released_training as synthetic
from transfer.native_scale_training import Run as NativeRun, baseline
from transfer.native_scale_fixture import Fixture, registration
from transfer.development_blocks import array_hash
from transfer.development_bank_training import _initial
from transfer.g3_training import state_hash
from transfer.wp07_observed_conditions import initialize, bind_condition, policies, selection_policy

FORMAT='wp07-observed-index-condition-integration-v1'
CASES=tuple(scope+'.'+mode for scope,mode in product(('both','cnn','graph'),('mean','content','metadata','joint')))


def key(role,event):
    return f'{role}/{event["tile"]}/{event["band"]}/{event["profile"]}'


class FixtureProvider(parent.FixtureProvider):
    def __init__(self, classes):
        graph.require(classes in (9,22),'Only the existing synthetic class catalogs are admitted')
        self.classes=classes;self.fixtures={r:Fixture(classes,r) for r in ('devtrain','devval')}
        self.catalog=digest(registration(classes));self.override='matched'

    def baseline(self,seed):return baseline(self.classes,seed)

    def events(self):
        train=[]
        for j,tile in enumerate(self.fixtures['devtrain'].tiles):
            for band in ('full','even_retained_positions'):
                train.append(dict(step=len(train)+1,tile=tile,factor=1,band=band,profile=('a','b','c','d')[j%4],condition='matched'))
        val=[dict(tile=tile,factor=1,band=band,profile=p,condition='matched') for tile,band,p in
            product(self.fixtures['devval'].tiles,('full','even_retained_positions'),('a','b','c','d'))]
        return train,val,[0,6,12]

    def selection_condition(self,actual,spec):
        return bind_condition(actual,self.classes+2,self.catalog,self.override)

    def evidence(self,actual,labels,targets):
        return dict(observation=actual.receipt['observation'],labels=array_hash(labels),released=targets.summary())

    def validate_observation(self,spec,actual,labels,targets,event):
        parent.FixtureProvider.validate_observation(self,spec,actual,labels,targets,event)
        graph.require(spec['inputs'][key(actual.receipt['role'],event)]==self.evidence(actual,labels,targets),'Registered synthetic input or released point changed')

    @contextmanager
    def intervention(self,value):
        graph.require(self.override=='matched','No nested condition intervention')
        self.override=value
        try:yield
        finally:self.override='matched'


def cache(provider,producer):
    model=graph.initialize(provider.baseline(64),7385);entries={}
    train,val,_=provider.events()
    for role,events in (('devtrain',train),('devval',val)):
        for event in events:
            actual,_,_=provider.materialize(role,event,producer)
            entry=graph.capture_geometry(model,actual,producer,role,16,condition=provider.selection_condition(actual,None))
            graph.compare_supports(entry['near'],entry['wide'],require_distinct=True)
            k=key(role,event);graph.require(k not in entries,'Repeated source geometry');entries[k]=entry
    return entries


def contract(provider,seed,arm,producer,numeric,cache_pin,authority):
    graph.require_checks();graph.require(seed in (64,512,2048) and arm in CASES and provider.synthetic,'Only registered synthetic integration may update')
    source=provider.baseline(seed);model=initialize(source,seed+7321);train,val,events=provider.events();inputs={}
    for role,roster in (('devtrain',train),('devval',val)):
        for event in roster:
            actual,labels,points=provider.materialize(role,event,producer);inputs[key(role,event)]=provider.evidence(actual,labels,points)
    return dict(format=FORMAT,producer=producer,classes=provider.classes,seed=seed,arm=arm,
        supervision=parent.MODE,objective=parent.LOSS,training=train,validation=val,steps=12,prefix=5,events=events,
        numeric=deepcopy(numeric),module_seed=seed+7321,rng_seed=seed+15000,learning_rate=.0001,
        initialization=_initial(model,source),source_state=state_hash(source),parameters=sum(p.numel() for p in model.parameters()),
        registered_graph_scale=16,cache=cache_pin,authority=authority,synthetic=True,inputs=inputs,
        condition_catalog=provider.catalog,selection='first_min_released_point_fused_nll',
        common_candidates={'cnn':[1,2,3],'graph':[1,2]},condition_semantics='observed_processed_band_indices_separate_from_verified_physical')


class ConditionRun(parent.RangeRun):
    """Only construction and input binding differ; start/update/evaluate/pack/restore remain native."""
    def __init__(self,spec,provider,entries,producer,execution):
        graph.require_checks()
        expected=contract(provider,spec['seed'],spec['arm'],producer,execution.numeric,spec['cache'],spec['authority'])
        graph.require(spec==expected,'Condition integration registration, input, budget or arithmetic differs')
        self._contract,self.producer,self.execution=deepcopy(spec),producer,execution
        self._released,self._entries=provider,entries;self.device=execution.device
        self.model=initialize(provider.baseline(spec['seed']),spec['module_seed']).to(self.device)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=spec['learning_rate'],foreach=False,fused=False)
        self._train,self._val='devtrain','devval';self._history,self._validation,self._selected=[],[],None
        self._fresh=tuple(spec['initialization']['fresh']);self._capacity,self._region_action=None,None
        self._selection_override=None;self._policy_override=None
        self.timing=dict(train_update_seconds=0.,validation_seconds=0.,transform_seconds=0.,forward_seconds=0.,graph_seconds=0.)

    def _materialize(self,role,event):
        result=NativeRun._materialize(self,role,event);entry=self._entries[key(role,event)]
        graph.require(graph.identity(result[0])==entry['identity'],'Geometry does not match this profile/band/input')
        self._current_entry=entry;return result

    def _forward(self,actual,role,intervention):
        if self._released.override!='matched' or self._policy_override is not None:
            graph.require(not self.model.training and not torch.is_grad_enabled(),'Condition intervention is read-only inference')
        entry=self._current_entry;graph.require(graph.identity(actual)==entry['identity'],'Frozen input differs')
        with selection_policy(self.model,*policies(self._policy_override or self._contract['arm'])),graph.frozen_support(self.model,entry,'content'):
            output,trace,condition=NativeRun._forward(self,actual,role,intervention)
        graph.require(trace['q']==tensor_hash(entry['q']) and trace['a0']==tensor_hash(entry['near']),'Frozen Q/A differs')
        graph.require(trace['range_nnz']==[entry['near']._nnz(),entry['wide']._nnz()],'Candidate graph pool differs')
        trace['range_identity']={k:graph.sparse_digest(entry[k]) for k in ('q','near','wide')}
        trace['selection_override']=None
        trace['condition_semantics']='index_observed_physical_verified_only'
        return output,trace,condition

    @contextmanager
    def policy_intervention(self,policy):
        graph.require(policy in CASES and self._policy_override is None,'Unknown or nested policy intervention')
        self._policy_override=policy
        try:yield
        finally:self._policy_override=None

    def restore(self,value,cursor):
        NativeRun.restore(self,value,cursor)
        for role,rows in (('devtrain',value['history']),('devval',[r for v in value['validation'] for r in v['rows']])):
            for row in rows:
                entry=self._entries[key(role,row)]
                graph.require(row['graph']['range_identity']=={k:graph.sparse_digest(entry[k]) for k in ('q','near','wide')},'Checkpoint frozen geometry changed')

    def update(self,event):
        before={n:p.detach().clone() for n,p in self.model.named_parameters()}
        NativeRun.update(self,event)
        self._history[-1]['parameter_participation']=[dict(tensor=n,numel=p.numel(),
            gradient_present=p.grad is not None,
            nonzero_gradient_elements=0 if p.grad is None else int(torch.count_nonzero(p.grad)),
            changed_elements=int(torch.count_nonzero(p.detach()-before[n])))
            for n,p in self.model.named_parameters()]
