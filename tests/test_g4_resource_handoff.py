"""Ownership/resource negatives and an actual frozen-function full1500 worker."""
import copy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import g4_resource_handoff as m
from research_runtime import ROOT, code_snapshot, write_json
from loadData.dataset_registry import file_hash


class ResourceTests(unittest.TestCase):
    def test_exact_partition_dynamic_three_card_stages_and_total_budget(self):
        names = ["case-"+str(i) for i in range(54)]
        old = [{"name": n, "updates": 1500} for n in names[:47]]
        jobs = [(n,"continue" if i==47 else "first2") for i,n in enumerate(names) if i>=47]
        ledger=m.Ledger(jobs,old,names)
        def queue(gpu):
            while True:
                job=ledger.claim(gpu)
                if job is None:return
                name,stage=job;time.sleep(.001)
                ledger.finish(gpu,{"case":name,"stage":stage,"gpu":gpu},
                    {"name":name,"updates":1500} if stage=="audit" else None)
        threads=[threading.Thread(target=queue,args=(g,)) for g in (1,3,6)]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual(len(ledger.rows),54)
        self.assertEqual(sum(m.STEPS[e['stage']][1] for e in ledger.events if e['stage']!='audit'),10498)
        self.assertEqual(len({(e['case'],e['stage']) for e in ledger.events}),len(ledger.events))
        self.assertFalse(ledger.active or ledger.pending)
        self.assertEqual(sum(r['updates'] for r in ledger.rows.values()),81000)

    def test_duplicate_missing_gpu_owner_and_failure_stop_are_rejected(self):
        for jobs,rows,order in (([("a","first2"),("a","continue")],[],["a"]),
            ([("a","first2")],[{"name":"a"}],["a"]),([],[],["a"]),([("a","unknown")],[],["a"])):
            with self.assertRaises(ValueError):m.Ledger(jobs,rows,order)
        ledger=m.Ledger([("a","first2"),("b","first2")],[],["a","b"])
        self.assertEqual(ledger.claim(3),("a","first2"))
        with self.assertRaises(ValueError):ledger.claim(3)
        ledger.error="owned failure"
        self.assertIsNone(ledger.claim(6))
        ledger.finish(3,{"case":"a","stage":"first2","gpu":3})
        self.assertEqual(list(ledger.pending),[("b","first2")])

    def test_zero_update_resource_deferral_can_move_to_another_card(self):
        ledger=m.Ledger([('a','continue')],[],['a'])
        self.assertEqual(ledger.claim(1),('a','continue'))
        ledger.defer(1,{'case':'a','stage':'continue','gpu':1,
                       'result':{'operation':'resource_wait','updates_this_call':0}})
        self.assertEqual(ledger.claim(3),('a','continue'))
        with self.assertRaises(ValueError):
            ledger.defer(3,{'case':'a','stage':'continue','gpu':3,
                           'result':{'operation':'interrupted','updates_this_call':2}})

    def test_phase_only_accepts_unstarted_exact2_or_completed1500_and_owned_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/"case";case={"out":str(root)}
            self.assertEqual(m.phase(case),'first2')
            root.mkdir();state=root/'state-000002.pt';state.write_bytes(b'checkpoint')
            write_json(root/'progress.json',{'status':'interrupted','cursor':2,'state_file':state.name,'state_sha256':file_hash(state)})
            self.assertEqual(m.phase(case),'continue')
            (root/'.lock').write_text('owned')
            with self.assertRaises(ValueError):m.phase(case)
            (root/'.lock').unlink();state.write_bytes(b'changed')
            with self.assertRaises(ValueError):m.phase(case)
            write_json(root/'progress.json',{'status':'running','cursor':3})
            with self.assertRaises(ValueError):m.phase(case)
            write_json(root/'progress.json',{'status':'completed','cursor':1500});write_json(root/'result.json',{})
            self.assertEqual(m.phase(case),'readonly')

    def test_gate_enforces_physical_device_memory_and_utilization(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,{},clear=True), \
             patch.object(m.subprocess,'check_output',side_effect=['12000,10','13000,61','13000,60']) as query, \
             patch.object(m.time,'sleep'):
            self.assertTrue(m.gate(3,Path(temp)/'slot.json'))
            self.assertEqual(query.call_count,3)
            self.assertEqual(m.read(Path(temp)/'slot.json')['status'],'ready')
            os.environ['CUDA_VISIBLE_DEVICES']='6'
            with self.assertRaises(ValueError):m.gate(3,Path(temp)/'wrong.json')
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,{},clear=True),patch.object(m.subprocess,'check_output') as query:
            self.assertFalse(m.gate(3,Path(temp)/'slot.json',lambda:True));query.assert_not_called()

    def test_process_identity_rejects_pid_reuse_and_changed_files(self):
        pin=m.process(os.getpid());self.assertTrue(m.same_process(pin))
        pin['start_ticks']='wrong';self.assertFalse(m.same_process(pin))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'f';path.write_text('first');pins={str(path):m.sha(path)}
            m.check(pins);path.write_text('second')
            with self.assertRaises(ValueError):m.check(pins)

    def test_registration_pauses_only_owned_dummy_checkpoint_and_pins_inputs(self):
        from test_g4_matched_target_execution import registration_fixture
        from test_g4_matched_target_study import launch_fixture
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);code,reg,rp=registration_fixture(root,out=root/'run')
            out=Path(reg['out_dir']);out.mkdir();launch=launch_fixture(out)
            launch.update(code=code,phase='full_campaign',execution_enabled=True,optimizer_updates=81000,
                          registration_path=str(rp),registration_sha256=file_hash(rp))
            lp=out/'launch.json';write_json(lp,launch);write_json(out/'protected-files.json',{str(lp):file_hash(lp)})
            write_json(out/'progress.json',[])
            er=root/'resource-engineering.json';write_json(er,{'status':'engineering_passed','tested_commit':code['commit'],
                                                             'cpu_passed':True,'gpu_passed':True,'full_passed':True})
            dummy=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)','matched_target_campaign.py'],cwd=ROOT)
            try:
                args=SimpleNamespace(frozen_checkout=str(ROOT),expected_controller_commit=code['commit'],
                    engineering_receipt=str(er),engineering_sha256=file_hash(er),launch=str(lp),launch_sha256=file_hash(lp),
                    registration=str(rp),registration_sha256=file_hash(rp),gpu=[1,3,6],out=str(root/'resource'),
                    legacy_pid=dummy.pid,estimated_seconds=1000)
                with patch.object(m,'HERE',ROOT),io.StringIO() as buffer,patch('sys.stdout',buffer):m.register(args)
                self.assertEqual(m.process(dummy.pid)['state'],'T')
                a=m.read(root/'resource/amendment.json')
                self.assertEqual((a['optimizer_updates_total'],a['remaining_updates']),(81000,81000))
                self.assertEqual(len(a['jobs']),54)
                self.assertEqual(file_hash(lp),args.launch_sha256)
                with self.assertRaises(ValueError):m.load_amendment(root/'resource/amendment.json','0'*64)
                dummy.kill();dummy.wait()
            finally:
                if dummy.poll() is None:dummy.kill();dummy.wait()

    @unittest.skipUnless(os.environ.get('RESOURCE_TEST_GPU'),'Explicit physical GPU engineering fixture only')
    def test_actual_frozen_worker_2_plus1498_readonly0_and_original_strict_audit(self):
        from test_g4_matched_target_execution import registration_fixture
        from test_g4_matched_target_study import launch_fixture
        from test_g4_matched_target_audit import actual_case
        from g4_matched_target_audit import audit_case
        from test_joint_training import hashes
        import torch
        gpu=int(os.environ['RESOURCE_TEST_GPU'])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);items,actual,_=actual_case(root,device='cuda')
            code,reg,rp=registration_fixture(root,out=root/'campaign')
            reg['gpu']=gpu;write_json(rp,reg)
            out=Path(reg['out_dir']);out.mkdir();launch=launch_fixture(out)
            launch.update(code=code,phase='full_campaign',execution_enabled=True,optimizer_updates=81000,
                          registration_path=str(rp),registration_sha256=file_hash(rp))
            for case in launch['cases']:
                if case['target']=='xuzhou':
                    for k in ('inputs','target_context','graph_binding'):case[k]=copy.deepcopy(actual[k])
                if case['name']==actual['name']:
                    actual['out']=case['out'];case.update(actual);selected=case
            lp=out/'launch.json';write_json(lp,launch)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
            original=[sys.executable,str(ROOT/'scripts/train_g4_matched_target_full.py'),
                '--expected-commit',code['commit'],'--registration',str(rp),'--registration-sha256',file_hash(rp),
                '--launch-plan',str(lp),'--launch-sha256',file_hash(lp),'--target',selected['target'],
                '--seed',str(selected['seed']),'--method',selected['method'],'--mode',selected['mode'],'--stop-after','2']
            result=subprocess.run(original,cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode,0,result.stdout[-7000:])
            first=json.loads(result.stdout.splitlines()[-1])
            self.assertEqual((first['operation'],first['updates_this_call']),('interrupted',2))
            ar=root/'resource';ar.mkdir();(ar/'events').mkdir()
            others=[{'name':c['name'],'updates':1500} for c in launch['cases'] if c['name']!=selected['name']]
            a={'policy':m.POLICY,'gpus':[gpu,gpu+1],'optimizer_updates_total':81000,'query_labels_accessed':False,
               'frozen_checkout':str(ROOT),'controller_checkout':str(ROOT),'producer':code,'controller_code':code,
               'pins':{str(lp):file_hash(lp),str(rp):file_hash(rp)},'run':str(out),
               'launch':{'path':str(lp),'sha256':file_hash(lp)},'registration':{'path':str(rp),'sha256':file_hash(rp)},
               'jobs':{selected['name']:'continue'},'completed_rows':others,'remaining_updates':1498}
            ap=ar/'amendment.json';write_json(ap,a)
            command=[sys.executable,str(ROOT/'g4_resource_handoff.py'),'worker','--amendment',str(ap),
                '--amendment-sha256',file_hash(ap),'--case',selected['name'],'--gpu',str(gpu),'--stage']
            for stage,expected in m.STEPS.items():
                if stage=='first2':continue
                before=hashes(Path(selected['out'])) if stage=='readonly' else None
                result=subprocess.run(command+[stage],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode,0,result.stdout[-7000:])
                row=json.loads(result.stdout.splitlines()[-1])
                self.assertEqual((row['operation'],row['updates_this_call']),expected)
                if before is not None:self.assertEqual(before,hashes(Path(selected['out'])))
            checked=audit_case(selected,code,file_hash(lp),file_hash(rp),device='cuda')
            self.assertEqual((checked['updates'],checked['head_metrics'],checked['class_recalls'],checked['exact_reload_repeats']),
                             (1500,6,54,3))
            del items;torch.cuda.empty_cache()


if __name__=='__main__':unittest.main()
