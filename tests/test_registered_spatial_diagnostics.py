"""Actual freeze/export CLI, registered identity and retained failures."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from model.research_config import digest
from research_runtime import ROOT
from transfer.registered_development import execute_route,_json_new,_read,_load_registration
from evaluation.registered_spatial_diagnostics import freeze_diagnostics,export_diagnostics,_load
from evaluation.block_input_diagnostics import BlockInputPlan,review_block
from evaluation.graph_edge_intervention import _selected
from tests.test_registered_graph_route import prepared
from tests.test_block_input_diagnostics import spec


class RegisteredDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);cls.cases=[]
        for fixed in (False,True):
            root=cls.root/str(fixed);root.mkdir();_,_,_,_,reg,sha=prepared(root,fixed)
            done=execute_route(reg,out=root/'training',to_step=24,expected_commit=sha)
            _,run,_=_load(reg,done['checkpoint']);s=spec(run);s.update(small_area=4,thin_depth=1)
            (root/'requests').mkdir();requests=_json_new(root/'requests/requests.json',dict(version=1,blocks=[s]))
            cls.cases.append(dict(root=root,reg=reg,sha=sha,done=done['checkpoint'],requests=requests))
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def folder(self,c,name):return c['root']/(self._testMethodName+'-'+name)

    def frozen(self,c,name='freeze'):
        return freeze_diagnostics(c['reg'],c['done'],c['requests'],out=self.folder(c,name),expected_commit=c['sha'])

    def test_actual_cli_two_modes_and_exact_direct_input_review(self):
        for c in self.cases:
            before={p:p.read_bytes() for p in c['root'].rglob('*') if p.is_file()}
            base=[sys.executable,str(ROOT/'scripts/export_registered_spatial_diagnostics.py'),'--expected-commit',c['sha']]
            cmd=base+['freeze','--out',str(self.folder(c,'f'))]
            for name,pin in [('registration',c['reg']),('completed',c['done']),('requests',c['requests'])]:cmd+=['--'+name,pin['path'],'--'+name+'-sha256',pin['sha256']]
            result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            frozen=json.loads(result.stdout)
            result=subprocess.run(base+['export','--freeze',frozen['path'],'--freeze-sha256',frozen['sha256'],'--out',str(self.folder(c,'out'))],cwd=ROOT,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr);receipt=json.loads(_read(json.loads(result.stdout)))
            self.assertEqual(receipt['optimizer_updates'],0);self.assertFalse(receipt['independent_scientific_acceptance'])
            report=json.loads(_read(receipt['reports'][0]));v,run,_=_load(c['reg'],c['done'])
            request=json.loads(_read(c['requests']))['blocks'][0];args={k:request[k] for k in ('tile_id','target_ids','class_id','interventions')}
            plan=BlockInputPlan.freeze(run,**args)
            self.assertEqual(report['input_diagnostics'],review_block(run,plan,expected_plan_hash=plan.plan_hash))
            self.assertEqual(report['branches']['count'],report['heads']['fused']['components']['groups']['all']['count'])
            self.assertEqual(set(report['heads']['fused']['boundary']),{'1','2','3'})
            for p,raw in before.items():self.assertEqual(p.read_bytes(),raw)

    def test_freeze_only_rechecks_existing_selection_and_partial_rejected(self):
        c=self.cases[0]
        from model.VCASAHGCN import MultiDatasetVCASAHGCN
        original=MultiDatasetVCASAHGCN.forward;calls=[]
        def forward(model,*args,**kwargs):
            self.assertFalse(model.training);calls.append(1);return original(model,*args,**kwargs)
        with patch.object(MultiDatasetVCASAHGCN,'forward',forward),patch('torch.optim.Adam.step',side_effect=AssertionError('optimized')),patch('evaluation.registered_spatial_diagnostics.review_block',side_effect=AssertionError('freeze diagnostic')):
            pin=self.frozen(c)
        _,base,_=_load_registration(c['reg']);self.assertEqual(len(calls),sum(t['role']=='devval' for t in base.record['roles']))
        value=json.loads(_read(pin));self.assertTrue(value['selection_validation_rechecked']);self.assertFalse(value['diagnostic_scores_computed'])
        _,partial,_=_load_registration(c['reg']);folder=self.folder(c,'partial');folder.mkdir();p=folder/'step0.pt';pin=dict(path=str(p),sha256=partial.save(p))
        with self.assertRaises(ValueError):freeze_diagnostics(c['reg'],pin,c['requests'],out=self.folder(c,'bad'),expected_commit=c['sha'])
        self.assertFalse(self.folder(c,'bad').exists())

    def test_query_duplicate_extra_fields_and_changed_hash_rejected(self):
        c=self.cases[0];_,run,_=_load(c['reg'],c['done']);request=json.loads(_read(c['requests']))
        testid=next(t['id'] for t in run.record['roles'] if t['role']=='test')
        bad=[];v=copy.deepcopy(request);v['blocks'][0]['tile_id']=testid;bad.append(v)
        v=copy.deepcopy(request);v['blocks']*=2;bad.append(v)
        v=copy.deepcopy(request);v['blocks'][0]['query_labels']=[1];bad.append(v)
        for i,value in enumerate(bad):
            folder=self.folder(c,'req'+str(i));folder.mkdir();pin=_json_new(folder/'request.json',value)
            with self.assertRaises(ValueError):freeze_diagnostics(c['reg'],c['done'],pin,out=self.folder(c,'bad'+str(i)),expected_commit=c['sha'])
        with self.assertRaises(ValueError):freeze_diagnostics(c['reg'],dict(c['done'],sha256='0'*64),c['requests'],out=self.folder(c,'wrong'),expected_commit=c['sha'])
        with self.assertRaises(ValueError):freeze_diagnostics(c['reg'],c['done'],c['requests'],out=self.folder(c,'wrongcode'),expected_commit='0'*40)

    def test_output_isolation_and_frozen_plan_tamper_before_result(self):
        c=self.cases[0];frozen=self.frozen(c)
        for out in (c['root']/'inputs/new',c['root'],Path(frozen['path']).parent):
            with self.assertRaises((ValueError,FileExistsError)):export_diagnostics(frozen,out=out,expected_commit=c['sha'])
        value=json.loads(_read(frozen));value['blocks'][0]['input_plan']['class_id']=2;value['freeze_hash']=digest({k:v for k,v in value.items() if k!='freeze_hash'})
        folder=self.folder(c,'tamper');folder.mkdir();pin=_json_new(folder/'freeze.json',value)
        with self.assertRaises(ValueError):export_diagnostics(pin,out=self.folder(c,'bad'),expected_commit=c['sha'])
        self.assertFalse(self.folder(c,'bad').exists())
        with self.assertRaises(ValueError):export_diagnostics(dict(frozen,sha256='0'*64),out=self.folder(c,'wrong'),expected_commit=c['sha'])

    def test_failed_export_keeps_status_and_cannot_overwrite(self):
        c=self.cases[0];frozen=self.frozen(c);out=self.folder(c,'failed')
        with patch('evaluation.registered_spatial_diagnostics.review_block',side_effect=RuntimeError('injected diagnostic failure')):
            with self.assertRaisesRegex(RuntimeError,'injected'):export_diagnostics(frozen,out=out,expected_commit=c['sha'])
        self.assertTrue((out/'started.json').exists());self.assertFalse((out/'receipt.json').exists())
        self.assertEqual(json.loads((out/'failure.json').read_text())['status'],'failed')
        with self.assertRaises(FileExistsError):export_diagnostics(frozen,out=out,expected_commit=c['sha'])

    def test_zero_optimization_and_repeated_export_is_exact(self):
        c=self.cases[1];frozen=self.frozen(c)
        with patch('torch.optim.Adam.step',side_effect=AssertionError('optimized')):
            a=json.loads(_read(export_diagnostics(frozen,out=self.folder(c,'one'),expected_commit=c['sha'])))
            b=json.loads(_read(export_diagnostics(frozen,out=self.folder(c,'two'),expected_commit=c['sha'])))
        self.assertEqual(_read(a['reports'][0]),_read(b['reports'][0]));self.assertEqual(a['selected_step'],b['selected_step'])


if __name__=='__main__':unittest.main()
