from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import torch
from loadData.observation_sampling import BandMetadata
from scripts.run_native_released_points import arithmetic
from transfer import wp07_condition_run as run
from transfer import wp07_observed_conditions as cond
from transfer import wp07_graph_range as graph
from transfer.native_scale_observation import forward
from transfer.xuzhou_multiscale_audit import _exact


class ObservedConditionsTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.provider=run.FixtureProvider(9);cls.entries=run.cache(cls.provider,'fixture')
        cls.event=cls.provider.events()[1][0]
        cls.actual,_,_=cls.provider.materialize('devval',cls.event,'fixture');cls.entry=cls.entries[run.key('devval',cls.event)]

    def model(self):return cond.initialize(self.provider.baseline(64),7385)

    def call(self,model,cnn='joint',gnn='joint',intervention='matched',actual=None,entry=None):
        actual=self.actual if actual is None else actual;entry=self.entry if entry is None else entry
        binding=cond.bind_condition(actual,11,self.provider.catalog,intervention)
        with cond.selection_policy(model,cnn,gnn),graph.frozen_support(model,entry,'content'):
            return forward(model,actual,'fixture','devval',condition=binding,graph_scale=16)

    @torch.no_grad()
    def test_shared_capacity_and_missing_fallback_on_same_trained_tensors(self):
        model=self.model();before={n:v.clone() for n,v in model.state_dict().items()}
        for left,right in [('metadata','mean'),('joint','content')]:
            a,_=self.call(model,left,left,'missing');b,_=self.call(model,right,right)
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(a[:3],b[:3])))
        for scope in cond.SCOPES:
            for mode in cond.MODES:
                other=self.model();self.assertEqual(set(other.state_dict()),set(before))
                self.assertTrue(all(torch.equal(v,before[n]) for n,v in other.state_dict().items()))
        self.assertTrue(all(torch.equal(v,before[n]) for n,v in model.state_dict().items()))

    @torch.no_grad()
    def test_same_content_condition_swap_is_branch_separated_and_QA_fixed(self):
        model=self.model()
        for cnn,gnn,invariant_head in [('joint','mean',1),('mean','joint',2)]:
            a,ta=self.call(model,cnn,gnn);b,tb=self.call(model,cnn,gnn,'swap_band')
            self.assertTrue(torch.equal(a[invariant_head],b[invariant_head]))
            self.assertFalse(torch.equal(a[0],b[0]))
            self.assertEqual((ta['q'],ta['a0']),(tb['q'],tb['a0']))
        for mode in ('mean','content'):
            a,_=self.call(model,mode,mode);b,_=self.call(model,mode,mode,'swap_band')
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(a[:3],b[:3])))

    def test_index_positions_have_signal_beyond_count_and_never_promote_wavelengths(self):
        a=cond.index_features((0,2,4,6,8,10),11);b=cond.index_features((0,1,4,6,8,10),11)
        self.assertEqual(a[0],b[0]);self.assertEqual(a[1:3],b[1:3]);self.assertNotEqual(a,b)
        encoder=cond.IndexConditionEncoder();ref=encoder.indices[0].weight
        approx=BandMetadata('approx',(0,2,4),(500.,600.,700.),'nm','approximate',(True,)*3)
        first=encoder(cond.IndexPacket(approx,(0,2,4),5),None,None,ref)
        second=encoder(cond.IndexPacket(None,(0,2,4),5),None,None,ref)
        self.assertTrue(torch.equal(first,second));self.assertIsNone(encoder(None,None,None,ref))
        with self.assertRaisesRegex(ValueError,'GSD'):encoder(cond.IndexPacket(approx,(0,2,4),5),None,0.,ref)
        with self.assertRaisesRegex(ValueError,'duplicate'):cond.index_features((0,0),5)

    def test_registered_condition_and_source_budget_tampering_rejected(self):
        binding=cond.bind_condition(self.actual,11,self.provider.catalog)
        with self.assertRaisesRegex(ValueError,'changed'):replace(binding,gsd_m=999.).validate(self.actual,self.actual.receipt)
        with TemporaryDirectory() as name,arithmetic(Path(name),None) as execution:
            spec=run.contract(self.provider,64,'both.joint','fixture',execution.numeric,{'fixture':'cache'},{'synthetic':True})
            engine=run.ConditionRun(spec,self.provider,self.entries,'fixture',execution);engine.start();engine.update(spec['training'][0]);value=engine.pack()
            active=[n for n,e in value['history'][0]['fresh'].items() if '.indices.' in n and e['gradient_l1'] is not None and e['gradient_l1']>0 and e['changed']]
            self.assertTrue(active)
            replay=run.ConditionRun(spec,self.provider,self.entries,'fixture',execution);replay.restore(value,1);self.assertTrue(_exact(replay.pack(),value))
            with self.assertRaisesRegex(ValueError,'registration'):run.ConditionRun({**spec,'steps':13},self.provider,self.entries,'fixture',execution)
            altered=dict(self.entries);k=run.key('devtrain',spec['training'][0]);e=dict(altered[k]);q=e['q'].coalesce();e['q']=torch.sparse_coo_tensor(q.indices(),q.values()*.99,q.shape).coalesce();altered[k]=e
            with self.assertRaisesRegex(ValueError,'geometry changed'):run.ConditionRun(spec,self.provider,altered,'fixture',execution).restore(value,1)

    @torch.no_grad()
    def test_same_candidates_execute_under_all_four_policies(self):
        model=self.model();first=model.backbone.CNN_Branch[0]
        for mode in cond.MODES:
            with patch.object(first,'_filter',wraps=first._filter) as filters:
                output,trace=self.call(model,mode,mode)
                self.assertEqual([c.args[1] for c in filters.call_args_list],[1,2,3])
                self.assertEqual(len(trace['range_nnz']),2)
                self.assertEqual(trace['weights']['cnn']['scales'],[1,2,3])
                self.assertEqual(trace['weights']['graph']['scales'],[1,2])
                self.assertEqual(len(output[:3]),3)
        self.assertFalse(first.content_selector._forward_hooks)
        self.assertFalse(model.backbone.range_metadata._forward_hooks)
