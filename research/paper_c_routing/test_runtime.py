"""Zero-update synthetic integration/fault tests using the frozen WP11 runtime."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from inference.native_deployment import Request, Runtime, RouteFailure
from evaluation.wp11_deployment_synthetic import models
from policy import ARMS, execute, plan, metrics
from audit import independent_plans
from audit import audit_prepared_inputs
from policy import digest as input_digest
from model.research_config import digest as canonical_digest
from research_runtime import tensor_hash
import hashlib
import json


@pytest.fixture
def actual():
    torch.set_num_threads(1)
    full,light=models(64,'xuzhou')
    engine=Runtime(device='cpu',full=full,light=light,graph_policy='rebuild')
    gen=torch.Generator().manual_seed(512)
    rows=[]
    for parent in range(2):
        for stride in (1,2,4):
            n=12//stride
            rows.append(Request(f'{parent}-stride{stride}','xuzhou',f'obs{parent}-{stride}',str(parent),
                        f'stride{stride}',torch.rand(8,8,n+2,generator=gen),
                        torch.linspace(415,2508,n),None,None,20,1/stride,True))
    return engine,rows


def test_six_arms_skip_actual_graph_and_preserve_endpoints(actual):
    engine,rows=actual
    light=execute(engine,rows,'light');full=execute(engine,rows,'full')
    reference=execute(engine,rows,'content_coverage')['full_plan']
    assert light['counts']['graph_builds']==0 and full['counts']['graph_builds']==6
    for arm in ARMS[2:]:
        routed=execute(engine,rows,arm,reference=reference)
        assert routed['counts']['light_calls']==6
        assert routed['counts']['full_calls']==routed['counts']['graph_builds']==3
        assert routed['counts']['graph_reuses']==0
        for i,row in enumerate(rows):
            endpoint=full if row.key in routed['full_plan'] else light
            assert torch.equal(routed['predictions'][i][0],endpoint['predictions'][i][0])
    assert 'build_graph' not in vars(engine.full.backbone.learnable_graph)


def test_bad_metadata_rejected_before_any_neural_call(actual,monkeypatch):
    engine,rows=actual;calls=[]
    monkeypatch.setattr(engine,'run',lambda *a,**kw:calls.append(a))
    cases=[(rows,'unknown',{}),(rows,'content',{'seed':-1}),
           ([rows[0],rows[0]],'light',{}),([replace(rows[0],known=False)],'full',{}),
           ([replace(rows[0],coverage=float('nan'))],'content',{}),
           (rows,'matched_random',{'reference':['foreign']}),(rows,'content',{'full_fraction':float('nan')})]
    for reqs,arm,kwargs in cases:
        with pytest.raises(ValueError):execute(engine,reqs,arm,**kwargs)
    assert not calls


def test_full_failure_cleans_graph_override_and_weight_drift_rejects(actual,monkeypatch):
    engine,rows=actual
    builder=engine.full.backbone.learnable_graph
    original=builder.build_graph
    def broken(*args,**kwargs):raise RuntimeError('registered graph failure')
    monkeypatch.setattr(type(builder),'build_graph',broken)
    with pytest.raises(RouteFailure):execute(engine,rows,'content',full_fraction=1)
    assert 'build_graph' not in vars(builder)
    monkeypatch.setattr(type(builder),'build_graph',original.__func__)
    assert execute(engine,rows,'light')['counts']['full_calls']==0
    with torch.no_grad():next(engine.light.parameters()).add_(.1)
    with pytest.raises(ValueError,match='model changed'):execute(engine,rows,'light')


def test_cheap_failure_aborts_without_spending_full_budget(actual,monkeypatch):
    engine,rows=actual;calls=[]
    original=engine.run
    def run(reqs,mode,**kwargs):
        calls.append(mode)
        if mode=='light':raise RuntimeError('registered cheap failure')
        return original(reqs,mode,**kwargs)
    monkeypatch.setattr(engine,'run',run)
    with pytest.raises(RuntimeError,match='cheap failure'):execute(engine,rows,'content_coverage')
    assert calls==['light']


def test_independent_reconstruction_has_no_producer_plan_dependency(actual):
    engine,rows=actual
    light=execute(engine,rows,'light')
    arrays={f'p{i}':p[0].numpy() for i,p in enumerate(light['predictions'])}
    roster=[{'key':r.key,'parent':r.parent,'band':r.band,'coverage':r.coverage} for r in rows]
    expected,margins,_,_=independent_plans(roster,arrays,{'seed':64,'full_fraction':.5})
    reference=execute(engine,rows,'content_coverage')['full_plan']
    for arm in ARMS:
        got=execute(engine,rows,arm,reference=reference)
        assert got['full_plan']==expected[arm]
    with pytest.raises(ValueError):metrics([1.5],[[.5,.5]],2)
    with pytest.raises(ValueError):metrics([1],[[1.2,-.2]],2)


def test_numpy_rebuild_matches_native_request_bytes_and_detects_normalization_drift(tmp_path):
    raw=np.random.default_rng(64).random((8,8,12),dtype=np.float32)
    path=tmp_path/'tile-0.npy';np.save(path,raw,allow_pickle=False)
    waves=np.linspace(415,2508,12)
    stats={'schema_version':1,'plan_hash':'synthetic-plan','fit_partition':'train',
           'minimum':[0.]*12,'maximum':[1.]*12}
    stats['statistics_hash']=canonical_digest(stats)
    dev={'statistics':stats,'plan':{'shape':[8,8],'plan_hash':'synthetic-plan'},
         'record':{'roles':[{'id':0,'role':'devval','box':[0,8,0,8]}]},
         'blocks':[{'id':0,'name':'tile-0.npy','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}],
         'bands':{'wavelengths':waves.tolist()}}
    yy,xx=np.mgrid[0:8,0:8];coords=np.stack((yy/7,xx/7),axis=-1).astype(np.float32)
    tensor=torch.from_numpy(np.concatenate((raw,coords),axis=-1));wave=torch.tensor(waves,dtype=torch.float32)
    requests=[]
    for stride in (1,2,4):
        pos=torch.arange(0,12,stride);x=torch.cat((tensor[:,:,pos],tensor[:,:,12:]),-1)
        requests.append(Request(f'0-stride{stride}','xuzhou',tensor_hash(x),'0',f'stride{stride}',x,wave[pos],None,None,100,len(pos)/12,True))
    setup={'input_hash':input_digest([r.identity for r in requests]),
           'requests':[{'key':r.key,'coverage':r.coverage,'bands':len(r.waves)} for r in requests]}
    contract={'model_pins':{},'descriptor':'retained_fraction','normalization':'released_development'}
    result=audit_prepared_inputs(tmp_path,tmp_path/'development.json',dev,contract,setup)
    assert result['input_hash_exact'] and result['request_count']==3
    stats['minimum'][0]=.1
    stats['statistics_hash']=canonical_digest({k:v for k,v in stats.items() if k!='statistics_hash'})
    with pytest.raises(ValueError,match='normalized request bytes'):
        audit_prepared_inputs(tmp_path,tmp_path/'development.json',dev,contract,setup)
