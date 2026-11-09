"""Actual-network paired optimization, exposure, selection and exact recovery."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from loadData.spatial_tiles import tile_slice
from model.task_losses import probability_loss_components
from transfer.curriculum_exposure import ExposurePlan
from transfer.curriculum_training import CurriculumRun
from transfer.development_training import sparse_labels, evaluate_validation
from transfer.g3_training import _forward, state_hash
from transfer.sample_curriculum import SampleCurriculum, score_training_support
from transfer.tile_sampling import SupportTileSampler
from tests.test_int8_model import fixture


def setup(seed=64, order='easy_to_hard', updates=24):
    source, scene, record = fixture()
    plan = ExposurePlan.freeze(scene, record, updates=updates, seed=seed,
            minimum_class_blocks={str(c):1 for c in record['classes']}, minimum_support_visits=1)
    scores = score_training_support(source, scene, record, expected_teacher_state=state_hash(source))
    policy = None if order is None else SampleCurriculum.freeze(scores, updates=updates, order=order, seed=512,
            start_fraction=.2, full_at=updates, minimum_weight=.1)
    return source, scene, record, plan, policy


def make(args):
    source, scene, record, plan, policy = args
    return CurriculumRun(source, scene, record, plan=plan, policy=policy, lr=.0001, torch_seed=64,
            expected_source_state=state_hash(source), expected_plan_hash=plan.plan_hash,
            expected_policy_hash=None if policy is None else policy.policy_hash)


def load(args, path, sha, contract):
    source, scene, record, plan, policy = args
    return CurriculumRun.load(source, scene, record, path, plan=plan, policy=policy,
            expected_sha256=sha, expected_contract_hash=contract)


class CurriculumTrainingTests(unittest.TestCase):
    def test_six_arms_24_vs_7_plus_17_exact_and_completed_zero(self):
        reference_draws = None
        for order in (None, 'balanced', 'fixed', 'random', 'easy_to_hard', 'hard_to_easy'):
            args = setup(order=order); run = make(args); initial = state_hash(args[0])
            tr = torch.get_rng_state().clone(); nr = np.random.get_state()
            with tempfile.TemporaryDirectory() as d:
                zero = Path(d)/'zero.pt'; zs = run.save(zero)
                self.assertEqual(load(args,zero,zs,run.contract_hash).completed,0)
                run.advance(7); path=Path(d)/'partial.pt'; sha=run.save(path)
                run.advance(24); resumed=load(args,path,sha,run.contract_hash); resumed.advance(24)
                self.assertEqual(run.history,resumed.history)
                self.assertEqual(state_hash(run.model),state_hash(resumed.model))
                self.assertEqual(run.finish(),resumed.finish())
                self.assertTrue(torch.equal(run.torch_rng,resumed.torch_rng))
                for i,slot in run.optimizer.state_dict()['state'].items():
                    for k,t in slot.items(): self.assertTrue(torch.equal(t,resumed.optimizer.state_dict()['state'][i][k]))
                full=Path(d)/'full.pt'; fs=run.save(full)
                with patch('torch.optim.Adam.step',side_effect=AssertionError('extra optimizer update')):
                    finished=load(args,full,fs,run.contract_hash); finished.advance(24); finished.finish()
                    with self.assertRaises(ValueError):finished.step()
                self.assertEqual(finished.completed,24)
            draws=[(r['tile_id'],r['anchor'],r['support_ids']) for r in run.history]
            if reference_draws is None:reference_draws=draws
            self.assertEqual(reference_draws,draws)
            self.assertEqual(initial,state_hash(args[0])); self.assertNotEqual(initial,state_hash(run.model))
            self.assertTrue(torch.equal(tr,torch.get_rng_state())); np.testing.assert_array_equal(nr[1],np.random.get_state()[1])
            self.assertEqual(run.selected_step,min(range(24),key=lambda i:run.history[i]['validation']['nll'])+1)
            for group in ('adapter','cnn','graph','heads'):
                self.assertGreater(sum(r['gradient_l1'][group] for r in run.history),0)
                self.assertGreater(sum(r['changed_parameters'][group] for r in run.history),0)
            self.assertTrue(run.finish()['exposure']['exposure']['meets_minimum'])

    def test_off_and_balanced_exact_original_manual_three_head_control(self):
        args=setup(order=None); run=make(args)
        source,scene,record,plan,_=args
        scores=score_training_support(source,scene,record,expected_teacher_state=state_hash(source))
        balanced=SampleCurriculum.freeze(scores,updates=24,order='balanced',seed=512,start_fraction=.2,full_at=24,minimum_weight=.1)
        other=make((*args[:4],balanced))
        model=copy.deepcopy(source); opt=torch.optim.Adam(model.parameters(),lr=.0001)
        sampler=SupportTileSampler(scene.plan,record['roles'],sparse_labels(record,scene.plan,'train_support'),record['classes'],'class_first')
        rng=np.random.default_rng(64)
        for step in range(1,25):
            tile,anchor=sampler.draw(rng); model.train(); opt.zero_grad(set_to_none=True)
            target=torch.tensor(sparse_labels(record,scene.plan,'train_support')[tile_slice(tile)].ravel())
            mask=(target>0)[:,None].expand(-1,scene.classes).float()
            onehot=torch.nn.functional.one_hot((target-1).clamp_min(0),scene.classes).float()*mask
            loss=probability_loss_components(_forward(model,scene,tile),onehot,mask,class_balance=True)['total']
            loss.backward(); opt.step(); val=evaluate_validation(model,scene,record)
            a=run.step(); b=other.step()
            self.assertEqual(a,b); self.assertEqual(a['loss'],float(loss.detach()));self.assertEqual(a['validation'],val)
            self.assertEqual(state_hash(model),state_hash(run.model))

    def test_underexposure_bad_identity_and_incompatible_contract_before_updates(self):
        with patch('torch.optim.Adam.step') as update:
            with self.assertRaisesRegex(ValueError,'exposure'):make(setup(updates=1))
            args=setup();s,scene,r,p,c=args
            for delta in ({'expected_source_state':'0'*64},{'expected_plan_hash':'0'*64},{'expected_policy_hash':'0'*64},{'torch_seed':True},{'lr':float('nan')}):
                kw=dict(plan=p,policy=c,lr=.0001,torch_seed=64,expected_source_state=state_hash(s),expected_plan_hash=p.plan_hash,expected_policy_hash=c.policy_hash);kw.update(delta)
                with self.assertRaises(ValueError):CurriculumRun(s,scene,r,**kw)
            update.assert_not_called()

    def test_data_mutation_detected_test_context_not_read_and_roles_isolated(self):
        args=setup();s,scene,r,_,_=args
        for tile in r['roles']:
            if tile['role'] not in ('devtrain','devval'):scene.cube[tile_slice(tile)]=np.nan
        run=make(args)
        with patch('transfer.curriculum_training._forward',wraps=_forward) as train:
            run.step()
            self.assertEqual(r[ 'roles'][train.call_args.args[2]['id']]['role'],'devtrain')
        tile=next(t for t in r['roles'] if t['role']=='devval')
        scene.cube[tile_slice(tile)]+=1
        with patch('torch.optim.Adam.step') as update:
            with self.assertRaisesRegex(ValueError,'Frozen'):run.step()
            update.assert_not_called()

    def test_fault_after_optimizer_latches_failed_and_does_not_claim_exposure(self):
        run=make(setup())
        with patch('transfer.curriculum_training.evaluate_validation',side_effect=ValueError('injected validation failure')):
            with self.assertRaisesRegex(ValueError,'injected'):run.step()
        self.assertTrue(run.failed);self.assertEqual(run.completed,0)
        self.assertTrue(any(int(x['step'])==1 for x in run.optimizer.state.values()))
        with self.assertRaisesRegex(ValueError,'Failed'):run.advance(24)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'Failed'):run.save(Path(d)/'invalid.pt')
        with self.assertRaisesRegex(ValueError,'Failed'):run.finish()

    def test_authenticated_resume_rejects_semantic_tamper_and_overwrite(self):
        args=setup();run=make(args);run.advance(3)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.pt';sha=run.save(path)
            with self.assertRaises(FileExistsError):run.save(path)
            with patch('torch.load') as reader:
                with self.assertRaisesRegex(ValueError,'hash mismatch'):load(args,path,'0'*64,run.contract_hash)
                reader.assert_not_called()
            packet=torch.load(path,weights_only=True)
            def alter(kind):
                v=copy.deepcopy(packet)
                if kind=='budget':v['progress']['completed_updates']=4
                if kind=='selection':v['selected_step']=0
                if kind=='optimizer':next(iter(v['optimizer']['state'].values()))['step']+=1
                if kind=='history':v['history'][0]['support_ids'].append(999)
                if kind=='weights':v['history'][0]['weights'][0]=0
                if kind=='validation':v['history'][0]['validation']['recall'][0]=2
                if kind=='last':next(iter(v['state'].values())).add_(.1)
                if kind=='selected':next(iter(v['selected_state'].values())).add_(.1)
                return v
            for kind in ('budget','selection','optimizer','history','weights','validation','last','selected'):
                bad=Path(d)/(kind+'.pt');torch.save(alter(kind),bad); h=hashlib.sha256(bad.read_bytes()).hexdigest()
                with self.assertRaises(ValueError,msg=kind):load(args,bad,h,run.contract_hash)

    def test_live_optimizer_and_sampler_changes_fail_without_extra_updates(self):
        run=make(setup());run.optimizer.param_groups[0]['lr']*=2
        with self.assertRaisesRegex(ValueError,'optimizer'):run.step()
        run=make(setup());wrong=copy.deepcopy(run.plan.event(1));wrong['anchor']=-1
        with patch.object(run.sampler,'draw',return_value=(run.scene.plan['tiles'][wrong['tile_id']],-1)):
            with self.assertRaisesRegex(ValueError,'sampler'):run.step()
        self.assertTrue(run.failed);self.assertEqual(run.completed,0)


if __name__=='__main__':unittest.main()
