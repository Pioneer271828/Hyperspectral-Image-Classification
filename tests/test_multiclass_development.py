"""Fixed 10/5 per-class development contracts across 3/9/22-class networks."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch
from loadData.spatial_tiles import make_tile_plan,tile_slice
from loadData.spatial_development import make_development
from loadData.observation_sampling import BandMetadata
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint
from model.research_config import digest,canonical_json
from transfer.development_training import prepare_scene,validate,development_counts,evaluate_validation,sparse_labels
from transfer.observation_policy import scene_identity,ObservationPolicy,evaluate_conditions
from transfer.graph_route_context import RouteContextBank,route_state_hash
from transfer.graph_route_training import GraphRouteRun
from transfer.curriculum_exposure import ExposurePlan
from transfer.registered_development import export_development,load_development,register_route
from transfer.g3_training import _forward
from tests.test_registered_graph_route import command

ROOT=Path(__file__).resolve().parents[1]
BASE='b50d532e96862f43d4c5782e24704f705e6546e2'
torch.set_num_threads(1)


def fixture(classes,seed=64,fixed=False):
    spatial=make_tile_plan((60,60),20,1,2026)
    record=make_development(np.resize(np.arange(1,classes+1),(60,60)),spatial,list(range(1,classes+1)))
    cube=np.random.default_rng(3).normal(size=(60,60,8)).astype(np.float32)
    bands=BandMetadata('multiclass-fixture',tuple(range(8)),tuple(np.linspace(400,1000,8)),
                       'nm','approximate',(True,)*8)
    scene=prepare_scene(cube,bands,spatial,record);scene.dataset='synthetic'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(512)
        model=MultiDatasetVCASAHGCN({scene.dataset:classes},latent_channels=8,adapter_embed_dim=8,
            locality_window=None,norm_type='group_norm',graph_mode='learned',
            graph_route=dict(budget=2,mode='random',seed=64)).eval()
    model.backbone.adapter.cache_attention=False
    plan=ExposurePlan.freeze(scene,record,updates=24,seed=seed,
        minimum_class_blocks={str(c):1 for c in record['classes']},minimum_support_visits=1)
    ids=[t['id'] for t in record['roles'] if t['role'] in ('devtrain','devval')]
    bank=RouteContextBank.freeze(model,scene,record,tile_ids=ids,step=7,
        expected_source_state=route_state_hash(model)) if fixed else None
    return model,scene,record,plan,bank


def make(a):
    source,scene,record,plan,bank=a
    return GraphRouteRun(source,scene,record,plan=plan,bank=bank,expected_bank_hash=None if bank is None else bank.bank_hash,
        lr=.0001,torch_seed=512,expected_source_state=route_state_hash(source),expected_plan_hash=plan.plan_hash)


def reload(a,path,sha,contract):
    source,scene,record,plan,bank=a
    return GraphRouteRun.load(source,scene,record,path,plan=plan,bank=bank,expected_bank_hash=None if bank is None else bank.bank_hash,
        expected_sha256=sha,expected_contract_hash=contract)


class MulticlassDevelopmentTests(unittest.TestCase):
    def test_three_class_counts_dynamic_fixed_exact_recovery_and_zero(self):
        for classes in (3,9,22):
            for fixed in (False,True):
                with self.subTest(classes=classes,fixed=fixed),tempfile.TemporaryDirectory() as tmp:
                    a=fixture(classes,fixed=fixed);run=make(a);source=route_state_hash(a[0]);rng=torch.get_rng_state().clone()
                    run.advance(7);path=Path(tmp)/'partial.pt';sha=run.save(path);run.advance(24)
                    restored=reload(a,path,sha,run.contract_hash);restored.advance(24)
                    self.assertEqual(run.history,restored.history);self.assertEqual(run.finish(),restored.finish())
                    self.assertEqual(route_state_hash(run.model),route_state_hash(restored.model))
                    self.assertTrue(torch.equal(run.torch_rng,restored.torch_rng))
                    for key,slot in run.optimizer.state_dict()['state'].items():
                        for name,t in slot.items():self.assertTrue(torch.equal(t,restored.optimizer.state_dict()['state'][key][name]))
                    done=Path(tmp)/'done.pt';ds=run.save(done)
                    with patch('torch.optim.Adam.step',side_effect=AssertionError('completed update')):
                        rr=reload(a,done,ds,run.contract_hash);rr.advance(24);rr.finish()
                    self.assertEqual(route_state_hash(a[0]),source);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                    self.assertEqual(run.selected_step,min(range(24),key=lambda i:run.history[i]['validation']['nll'])+1)
                    for row in run.history:
                        self.assertEqual(row['validation']['count'],5*classes)
                        self.assertEqual(row['validation']['support'],[5]*classes)
                    if classes==22:
                        v=torch.load(done,weights_only=True);v['history'][0]['validation']['count']=45
                        bad=Path(tmp)/'bad.pt';torch.save(v,bad)
                        with self.assertRaisesRegex(ValueError,'validation history'):
                            reload(a,bad,hashlib.sha256(bad.read_bytes()).hexdigest(),run.contract_hash)

    def test_original_nine_class_arithmetic_exact(self):
        source,scene,record,_,_=fixture(9)
        text=subprocess.check_output(['git','show',BASE+':transfer/development_training.py'],cwd=ROOT,text=True)
        old=types.ModuleType('old_development');exec(compile(text,'old_development','exec'),old.__dict__)
        self.assertEqual(evaluate_validation(source,scene,record),old.evaluate_validation(source,scene,record))
        self.assertEqual(development_counts(record),(90,45))

    def test_three_heads_and_conditions_independent_confusion(self):
        for classes in (3,9,22):
            source,scene,record,_,_=fixture(classes)
            # Observation policy uses the original uncoupled model without route extra state.
            kwargs=copy.deepcopy(source.backbone_kwargs);kwargs.pop('graph_route')
            plain=MultiDatasetVCASAHGCN({scene.dataset:classes},**kwargs).eval()
            plain.load_state_dict({k:v for k,v in source.state_dict().items() if not k.startswith('backbone.graph_router.')})
            plain.backbone.adapter.cache_attention=False
            policy=ObservationPolicy.freeze(scene,record,updates=24,training_configs=[{'mode':'none'}],train_seed=64,
                validation_seed=512,conditions={'full':{'mode':'none'},'half':{'mode':'uniform_stride','stride':2,'offset':0}})
            result=evaluate_conditions(plain,scene,record,policy)
            target=sparse_labels(record,scene.plan,'validation_support')
            for condition in ('full','half'):
                matrices=[np.zeros((classes,classes),dtype=int) for _ in range(3)];loss=[0.,0.,0.]
                for tile in scene.plan['tiles']:
                    if record['roles'][tile['id']]['role']!='devval':continue
                    y=target[tile_slice(tile)].ravel();mask=y>0;y=y[mask]-1
                    view=policy.view(scene,record,tile,phase='validation',condition=condition)
                    with torch.no_grad():outputs=_forward(plain,scene,tile,observation_view=view)
                    for i,z in enumerate(outputs[:3]):
                        a=z.numpy()[mask];loss[i]-=np.log(np.maximum(a[np.arange(len(y)),y],1e-8)).sum()
                        for truth,pred in zip(y,a.argmax(1)):matrices[i][truth,pred]+=1
                for i,head in enumerate(('fused','graph','cnn')):
                    row=result['conditions'][condition][head]
                    self.assertEqual(row['confusion_matrix'],matrices[i].tolist())
                    self.assertAlmostEqual(row['nll'],loss[i]/(5*classes),places=12)
                    self.assertEqual([r['support'] for r in row['per_class']],[5]*classes)

    def test_22_class_real_cli_segments_and_complete_zero(self):
        a=fixture(22);source,scene,record,plan,_=a
        sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inputs=root/'inputs';inputs.mkdir()
            data=export_development(scene,record,inputs/'data');path=inputs/'source.pt';save_model_checkpoint(source,path)
            reg=register_route(path,data,plan,inputs/'registration.json',lr=.0001,torch_seed=512,expected_commit=sha)
            def cli(name,n,resume=None):
                p=subprocess.run(command(reg,sha,root/name,n,resume),cwd=ROOT,text=True,capture_output=True)
                self.assertEqual(p.returncode,0,p.stdout+p.stderr)
                return json.loads((root/name/'status.json').read_text())
            part=cli('partial',7);done=cli('done',24,part['checkpoint']);zero=cli('zero',24,done['checkpoint'])
            self.assertEqual([part['new_updates'],done['new_updates'],zero['new_updates']],[7,17,0])
            loaded=torch.load(done['checkpoint']['path'],weights_only=True)
            self.assertEqual(loaded['history'][-1]['validation']['support'],[5]*22)
            for pin in done['inputs']:self.assertEqual(hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest(),pin['sha256'])

    def test_unreleased_context_is_not_packaged_or_used(self):
        for classes in (3,22):
            source,scene,record,_,_=fixture(classes);other=copy.deepcopy(scene)
            for tile in record['roles']:
                if tile['role'] not in ('devtrain','devval'):other.cube[tile_slice(tile)]=np.nan
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);left=export_development(scene,record,root/'a');right=export_development(other,record,root/'b')
                self.assertEqual(left['sha256'],right['sha256']);loaded,rr,_=load_development(right)
                self.assertEqual(scene_identity(loaded,rr),scene_identity(scene,record))
                self.assertEqual(evaluate_validation(source,loaded,rr),evaluate_validation(source,scene,record))
                for tile in record['roles']:
                    if tile['role'] not in ('devtrain','devval'):self.assertTrue((loaded.cube[tile_slice(tile)]==0).all())

    def test_invalid_counts_labels_and_scene_mismatch(self):
        _,scene,record,_,_=fixture(22)
        for change in (lambda v:v.update(training_labels=90),lambda v:v.update(validation_labels=45),
                       lambda v:v.update(total_training_selection_labels=135),lambda v:v.update(classes=[]),
                       lambda v:v['train_support']['labels'].__setitem__(0,0),
                       lambda v:v['validation_support']['labels'].__setitem__(0,23),
                       lambda v:v['train_support']['labels'].__setitem__(0,True)):
            v=copy.deepcopy(record);change(v);v['validation_hash']=digest({k:x for k,x in v.items() if k!='validation_hash'})
            with self.assertRaises(ValueError):validate(v,scene.plan)
        other=copy.deepcopy(scene);other.classes=9
        with self.assertRaisesRegex(ValueError,'scene/record'):scene_identity(other,record)
        with self.assertRaises(ValueError):evaluate_validation(fixture(22)[0],other,record)


if __name__=='__main__':unittest.main()

