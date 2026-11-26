"""Behavioral recovery, accounting and fault tests; synthetic observations only."""
import copy
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from transactions import execute,inspect,locked,paired_plan,sha


class Toy:
    def __init__(self,*,ties=False,mutate=False):
        self.model=nn.Sequential(nn.Linear(4,8),nn.BatchNorm1d(8),nn.ReLU(),nn.Dropout(.3),nn.Linear(8,3))
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4)
        self.x=torch.arange(24,dtype=torch.float32).reshape(6,4)/24
        self.y=torch.tensor([0,1,2,0,1,2]);self.ties=ties;self.mutate=mutate

    def loss(self,block):
        # Exercise all three saved global RNGs, dropout and mutable BN buffers.
        x=self.x+float(np.random.uniform())*.01+random.random()*.01+block*.001
        return F.cross_entropy(self.model(x),self.y)

    def evaluate(self):
        if self.mutate:self.model[1].running_mean.add_(1)
        return {'nll':1.0 if self.ties else float(F.cross_entropy(self.model(self.x),self.y))}


def contract(pin):
    roster=[2,5,8];plan=paired_plan(roster,3,64)
    return {'roster':roster,'sweeps':3,'seed':64,'plan':plan,'updates':len(plan),
            'selection_steps':[0,3,6,9],'device':'cpu','query_reads':0,'confirmation':False,
            'optimizer':{'name':'Adam','lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,'amsgrad':False},
            'protected':{str(pin):sha(pin)},'scene':'synthetic','method':'synthetic-BN-dropout'}


def state(root,step):return torch.load(root/'stages'/f'{step:06d}'/'state.pt',weights_only=True)


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if type(a)!=type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.pin=self.root/'pin.txt';self.pin.write_text('immutable synthetic identity')
        self.c=contract(self.pin)

    def tearDown(self):self.tmp.cleanup()

    def test_exact_graceful_resume_all_state_and_selection(self):
        full=self.root/'full';split=self.root/'split'
        execute(full,self.c,Toy)
        r=execute(split,self.c,Toy,stop_after=4);self.assertEqual(r['new_updates'],4)
        r=execute(split,self.c,Toy,resume=True);self.assertEqual(r['new_updates'],5)
        for step in range(10):self.assertTrue(equal(state(full,step),state(split,step)),f'state drift at {step}')
        before={str(p):sha(p) for p in split.rglob('*') if p.is_file()}
        r=execute(split,self.c,lambda:self.fail('completed reentry constructed task'),resume=True)
        self.assertEqual(r['new_neural_calls'],0);self.assertEqual(r['new_updates'],0)
        self.assertEqual(before,{str(p):sha(p) for p in split.rglob('*') if p.is_file()})

    def test_after_commit_crash_resumes_exact_boundary(self):
        a=self.root/'a';b=self.root/'b';execute(a,self.c,Toy)
        def crash(phase,step):
            if phase=='after_commit' and step==3:raise RuntimeError('simulated process loss after durable commit')
        with self.assertRaisesRegex(RuntimeError,'durable commit'):execute(b,self.c,Toy,fault=crash)
        r=execute(b,self.c,Toy,resume=True);self.assertEqual(r['new_updates'],6)
        self.assertTrue(equal(state(a,9),state(b,9)))

    def test_interrupted_update_never_replayed(self):
        for phase in ('after_intent','after_backward','after_optimizer'):
            with self.subTest(phase=phase):
                root=self.root/phase
                def crash(actual,step):
                    if actual==phase and step==2:raise RuntimeError('uncertain update')
                with self.assertRaisesRegex(RuntimeError,'uncertain'):execute(root,self.c,Toy,fault=crash)
                saved={str(p):sha(p) for p in root.rglob('*') if p.is_file()}
                with self.assertRaisesRegex(ValueError,'Pending'):
                    execute(root,self.c,lambda:self.fail('fault recovery constructed task'),resume=True)
                self.assertEqual(saved,{str(p):sha(p) for p in root.rglob('*') if p.is_file()})
                failure=json.loads((root/'stages'/'.pending-000002'/'failure.json').read_text())
                self.assertEqual(failure['committed_updates_before'],1)
                self.assertFalse(failure['automatic_retry'])

    def test_completed_corruption_and_source_drift_precede_factory(self):
        root=self.root/'a';execute(root,self.c,Toy)
        checkpoint=root/'stages'/'000004'/'state.pt'
        checkpoint.write_bytes(checkpoint.read_bytes()+b'corrupt')
        with self.assertRaisesRegex(ValueError,'changed'):
            execute(root,self.c,lambda:self.fail('corruption constructed task'),resume=True)
        other=self.root/'other';execute(other,self.c,Toy)
        self.pin.write_text('changed original')
        with self.assertRaisesRegex(ValueError,'protected'):
            execute(other,self.c,lambda:self.fail('drift constructed task'),resume=True)

    def test_first_minimum_ties_keep_initial(self):
        root=self.root/'ties';r=execute(root,self.c,lambda:Toy(ties=True))
        self.assertEqual(r['selected_step'],0)
        self.assertEqual(state(root,9)['best']['step'],0)

    def test_equal_exposure_sweeps_and_no_implicit_overwrite(self):
        plan=self.c['plan']
        for i in range(0,len(plan),3):self.assertEqual(sorted(plan[i:i+3]),self.c['roster'])
        root=self.root/'a';execute(root,self.c,Toy,stop_after=2)
        with self.assertRaisesRegex(ValueError,'Explicit resume'):execute(root,self.c,Toy)
        changed=copy.deepcopy(self.c);changed['plan'][0]=999
        with self.assertRaisesRegex(ValueError,'plan'):execute(root,changed,Toy,resume=True)
        self.assertEqual(len(inspect(root,self.c)[0]),3)

    def test_selection_mutation_rejected_and_writer_locked(self):
        with self.assertRaisesRegex(ValueError,'Selection mutated'):
            execute(self.root/'bad',self.c,lambda:Toy(mutate=True))
        root=self.root/'locked';execute(root,self.c,Toy,stop_after=1)
        with locked(root):
            with self.assertRaisesRegex(ValueError,'writer'):execute(root,self.c,Toy,resume=True)

    def test_zero_update_post_final_commit_recovery(self):
        root=self.root/'final'
        def crash(phase,step):
            if phase=='after_commit' and step==9:raise RuntimeError('before completion receipt')
        with self.assertRaises(RuntimeError):execute(root,self.c,Toy,fault=crash)
        r=execute(root,self.c,Toy,resume=True)
        self.assertEqual(r['new_updates'],0);self.assertTrue(r['complete'])


if __name__=='__main__':unittest.main()
