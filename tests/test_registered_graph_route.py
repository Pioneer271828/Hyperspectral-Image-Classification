"""Real CLI segments, authenticated inputs, output isolation and failure retention."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from model.checkpoint_io import save_model_checkpoint
from model.research_config import canonical_json
from research_runtime import ROOT, code_snapshot
from transfer.curriculum_exposure import ExposurePlan
from transfer.graph_route_training import GraphRouteRun
from transfer.registered_development import export_development, load_development, register_route, execute_route
from tests.test_graph_route_context import setup


def prepared(root, fixed=False):
    source,scene,record,ids,bank=setup('random');inputs=root/'inputs';inputs.mkdir()
    data=export_development(scene,record,inputs/'data')
    path=inputs/'source.pt';save_model_checkpoint(source,path)
    plan=ExposurePlan.freeze(scene,record,updates=24,seed=64,
        minimum_class_blocks={str(c):1 for c in record['classes']},minimum_support_visits=1)
    bankpath=None
    if fixed:
        bankpath=inputs/'bank.json';bankpath.write_text(canonical_json(bank.to_dict()))
    sha=code_snapshot()['commit']
    registration=register_route(path,data,plan,inputs/'registration.json',lr=.0001,torch_seed=512,
        expected_commit=sha,bank_path=bankpath,expected_bank_hash=bank.bank_hash if fixed else None)
    return source,scene,record,data,registration,sha


def command(reg,sha,out,target,resume=None):
    args=[sys.executable,str(ROOT/'scripts/run_registered_graph_route.py'),'--expected-commit',sha,
        '--registration',reg['path'],'--registration-sha256',reg['sha256'],'--to-step',str(target),'--out',str(out)]
    if resume is not None:args+=['--resume',resume['path'],'--resume-sha256',resume['sha256']]
    return args


class RegisteredRouteTests(unittest.TestCase):
    def test_cli_two_plus_twenty_two_complete_zero_matches_full(self):
        for fixed in (False,True):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);_,_,_,_,reg,sha=prepared(root,fixed)
                def cli(name,n,resume=None):
                    p=subprocess.run(command(reg,sha,root/name,n,resume),cwd=ROOT,text=True,capture_output=True,
                                     env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
                    self.assertEqual(p.returncode,0,p.stdout+p.stderr)
                    return json.loads((root/name/'status.json').read_text())
                partial=cli('partial',2);finished=cli('finished',24,partial['checkpoint'])
                zero=cli('zero',24,finished['checkpoint']);full=cli('full',24)
                self.assertEqual((partial['new_updates'],finished['new_updates'],zero['new_updates']),(2,22,0))
                self.assertEqual(zero['status'],'completed');self.assertFalse(zero['independent_acceptance'])
                a=torch.load(finished['checkpoint']['path'],weights_only=True)
                b=torch.load(full['checkpoint']['path'],weights_only=True)
                self.assertEqual(a['history'],b['history']);self.assertEqual(a['selected_step'],b['selected_step'])
                for k,v in a['state'].items():
                    if torch.is_tensor(v):self.assertTrue(torch.equal(v,b['state'][k]))
                    else:self.assertEqual(v,b['state'][k])
                for i,slot in a['optimizer']['state'].items():
                    for k,v in slot.items():self.assertTrue(torch.equal(v,b['optimizer']['state'][i][k]))
                for pin in finished['inputs']:self.assertEqual(hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest(),pin['sha256'])

    def test_only_development_arrays_are_exported_and_roundtrip_exact(self):
        source,scene,record,ids,bank=setup();other=copy.deepcopy(scene)
        from loadData.spatial_tiles import tile_slice
        for tile in record['roles']:
            if tile['role'] not in ('devtrain','devval'):other.cube[tile_slice(tile)]=np.nan
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);left=export_development(scene,record,root/'a');right=export_development(other,record,root/'b')
            self.assertEqual(left['sha256'],right['sha256'])
            restored,rr,files=load_development(right);self.assertEqual(rr,record)
            self.assertEqual(len(list((root/'b').glob('*.npy'))),len(ids))
            for tile in record['roles']:
                a=restored.cube[tile_slice(tile)]
                if tile['role'] in ('devtrain','devval'):np.testing.assert_array_equal(a,scene.cube[tile_slice(tile)])
                else:self.assertTrue(np.all(a==0))

    def test_wrong_hash_and_sidecar_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,_,_,_,reg,sha=prepared(root)
            wrong={**reg,'sha256':'0'*64}
            with patch('torch.load',side_effect=AssertionError('deserialized before auth')):
                with self.assertRaises(ValueError):execute_route(wrong,out=root/'bad',to_step=1,expected_commit=sha)
            self.assertFalse((root/'bad').exists())
            meta=root/'inputs/source.pt.json';meta.write_bytes(meta.read_bytes()+b' ')
            with patch('torch.load',side_effect=AssertionError('deserialized changed sidecar')):
                with self.assertRaises(ValueError):execute_route(reg,out=root/'bad',to_step=1,expected_commit=sha)
            self.assertFalse((root/'bad').exists())

    def test_budget_code_and_overlap_rejected_without_updates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,_,_,_,reg,sha=prepared(root)
            with patch('torch.optim.Adam.step',side_effect=AssertionError('unexpected update')):
                for target in (-1,25,True):
                    with self.assertRaises(ValueError):execute_route(reg,out=root/'bad',to_step=target,expected_commit=sha)
                with self.assertRaises(ValueError):execute_route(reg,out=root/'bad',to_step=1,expected_commit='0'*40)
                with self.assertRaises(ValueError):execute_route(reg,out=root/'inputs/nested',to_step=1,expected_commit=sha)
                with patch('transfer.registered_development.code_snapshot',return_value={**code_snapshot(),'dirty':True}):
                    with self.assertRaises(ValueError):execute_route(reg,out=root/'bad',to_step=1,expected_commit=sha)
            self.assertFalse((root/'bad').exists())
            good=execute_route(reg,out=root/'zero',to_step=0,expected_commit=sha)
            with self.assertRaises(ValueError):execute_route(reg,out=root/'zero/nested',to_step=1,expected_commit=sha,resume=good['checkpoint'])
            with self.assertRaises(FileExistsError):execute_route(reg,out=root/'zero',to_step=0,expected_commit=sha)

    def test_failure_keeps_last_success_and_nonresumable_partial_state(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,_,_,_,reg,sha=prepared(root)
            with patch.object(GraphRouteRun,'_evaluate',side_effect=RuntimeError('injected after update')):
                with self.assertRaises(RuntimeError):execute_route(reg,out=root/'failed',to_step=2,expected_commit=sha)
            s=json.loads((root/'failed/status.json').read_text());self.assertEqual(s['status'],'failed')
            self.assertEqual(s['completed_updates'],0);self.assertEqual(s['new_updates'],0)
            p=torch.load(s['failure']['path'],weights_only=True);self.assertFalse(p['resumable'])
            self.assertTrue(p['optimizer']['state'])
            with self.assertRaises(ValueError):execute_route(reg,out=root/'bad-resume',to_step=2,expected_commit=sha,resume=s['failure'])


if __name__=='__main__':unittest.main()
