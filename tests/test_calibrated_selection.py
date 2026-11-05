"""Calibrated validation selection must never corrupt raw optimizer continuation."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from joint_calibrated_selection import (VALIDATION_POLICY, calibrated_settings, enabled,
                                        first_maximum, load_selected, validation_candidate)
from joint_training import _tree_fingerprint, run_training
from loadData.dataset_registry import file_hash
from model.bn_calibration import state_digest
from model.checkpoint_io import load_model_checkpoint
from model.research_config import ResearchConfig
from test_joint_baseline import tiny_items
from test_joint_training import SOURCE, PROVENANCE, checkpoints, hashes, settings

torch.set_num_threads(1)
DEVICE = os.environ.get('BNALIGN2_DEVICE', 'cpu')


def items():
    result = tiny_items()
    for i in result.values(): i['zeros'] = torch.zeros(i['height']*i['width'])
    return result


class CalibratedSelectionTests(unittest.TestCase):
    def test_policy_is_explicit_versioned_and_invalid_policy_fails_before_writing(self):
        legacy = settings(); before = copy.deepcopy(legacy)
        opts = calibrated_settings(legacy)
        self.assertEqual(legacy, before); self.assertFalse(enabled(legacy)); self.assertTrue(enabled(opts))
        self.assertEqual(opts['validation_policy'], VALIDATION_POLICY)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'run'
            for bad in (dict(opts, validation_policy={}), dict(legacy, validation_policy=VALIDATION_POLICY),
                        {k:v for k,v in opts.items() if k!='validation_policy'}):
                with self.assertRaises(ValueError): run_training(bad, items(), path, PROVENANCE, SOURCE)
                self.assertFalse(path.exists())

    def test_selection_strict_greater_first_tie_and_chronological_finite_guards(self):
        events = [{'step':3,'macro_oa':.8},{'step':33,'macro_oa':.8},{'step':63,'macro_oa':.7}]
        self.assertIs(first_maximum(events), events[0])
        events.append({'step':93,'macro_oa':.9}); self.assertIs(first_maximum(events),events[-1])
        for bad in ([], events[::-1], [{'step':3,'macro_oa':float('nan')}]):
            with self.assertRaises(ValueError): first_maximum(bad)

    def test_validation_clone_source_digest_and_rng_are_unchanged(self):
        data = items(); config = ResearchConfig.from_dict(settings()['config'])
        model = config.build_model({n:i['class_num'] for n,i in data.items()}).to(DEVICE).eval()
        before = state_digest(model); rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'raw.pt'
            candidate, receipt = validation_candidate(model, data, path)
            self.assertIsNot(candidate, model); self.assertEqual(state_digest(model), before)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(receipt['source_checkpoint_sha256'], file_hash(path))
            self.assertEqual(_tree_fingerprint(torch.load(path,weights_only=True)),
                             _tree_fingerprint(model.state_dict()))
            self.assertNotEqual(state_digest(candidate), before)
            self.assertEqual(receipt['optimizer_updates'],0)

    def test_calibrated_scheduler_primary_binding_and_raw_last(self):
        opts = calibrated_settings(settings()); data = items(); seen = []
        original = torch.optim.lr_scheduler.ReduceLROnPlateau.step
        def record(scheduler, metric, *args, **kwargs):
            seen.append(float(metric)); return original(scheduler, metric, *args, **kwargs)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'run'
            with patch.object(torch.optim.lr_scheduler.ReduceLROnPlateau, 'step', record):
                outcome = run_training(opts,data,path,PROVENANCE,SOURCE,device=DEVICE)
            state = checkpoints(path)
            self.assertEqual(seen,[e['val_loss'] for e in state['history']])
            self.assertEqual(outcome['result']['selected_step'],first_maximum(state['history'])['step'])
            self.assertEqual(outcome['result']['optimizer_steps'],12)
            raw = torch.load(path/f'raw-validation-{state["cursor"]:06d}.pt',weights_only=True)
            self.assertEqual(_tree_fingerprint(raw),_tree_fingerprint(state['model']))
            last = load_model_checkpoint(path/'last.pt',device=DEVICE)
            self.assertEqual(_tree_fingerprint(last.state_dict()),_tree_fingerprint(raw))
            self.assertNotIn('bn_calibration',last.training_config)
            selected = state['best_step']; source = path/f'raw-validation-{selected:06d}.pt'
            primary = load_selected(path/'primary.pt',data,source,device=DEVICE)
            self.assertEqual(primary.training_config['checkpoint_optimizer_step'],selected)
            self.assertEqual(primary.training_config['joint_runtime']['selected_step'],selected)
            self.assertEqual(primary.training_config['bn_calibration'],state['best_calibration'])
            self.assertEqual(_tree_fingerprint(primary.state_dict()),_tree_fingerprint(state['best_model']))
            with source.open('ab') as stream: stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError,'binding mismatch'):
                load_selected(path/'primary.pt',data,source,device=DEVICE)

    def test_exact_mid_epoch_and_validation_boundary_continuation(self):
        opts = calibrated_settings(settings()); data = items()
        with tempfile.TemporaryDirectory() as temp:
            reference = Path(temp)/'reference'
            run_training(opts,data,reference,PROVENANCE,SOURCE,device=DEVICE)
            expected = checkpoints(reference)
            for cursor in (2,3,5):
                path = Path(temp)/str(cursor)
                run_training(opts,data,path,PROVENANCE,SOURCE,device=DEVICE,stop_after=cursor)
                result = run_training(opts,data,path,PROVENANCE,SOURCE,device=DEVICE,resume=True)
                actual = checkpoints(path)
                self.assertEqual(result['updates_this_call'],12-cursor)
                for key in ('model','optimizer','scheduler','rng','trace','history','best_model','best_probabilities'):
                    self.assertEqual(_tree_fingerprint(actual[key]),_tree_fingerprint(expected[key]),key)
                self.assertEqual(actual['best_calibration'],expected['best_calibration'])

    def test_completed_resume_is_readonly_and_cross_policy_resume_rejected(self):
        opts = calibrated_settings(settings()); data = items()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'run'; run_training(opts,data,path,PROVENANCE,SOURCE,device=DEVICE)
            before = hashes(path)
            with patch('model.research_config.ResearchConfig.build_model',side_effect=AssertionError('must skip')):
                result = run_training(opts,data,path,PROVENANCE,SOURCE,device=DEVICE,resume=True)
            self.assertEqual(result['operation'],'verified_completed')
            self.assertEqual(result['result']['status'],'completed'); self.assertEqual(result['updates_this_call'],0)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                run_training(settings(),data,path,PROVENANCE,SOURCE,device=DEVICE,resume=True)
            self.assertEqual(before,hashes(path))

    def test_default_path_does_not_calibrate_or_create_calibration_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'run'
            with patch('joint_calibrated_selection.validation_candidate',side_effect=AssertionError('off path')):
                run_training(settings(),items(),path,PROVENANCE,SOURCE,device=DEVICE)
            self.assertFalse(list(path.glob('raw-validation-*')))
            self.assertNotIn('best_calibration',checkpoints(path))


if __name__ == '__main__': unittest.main()
