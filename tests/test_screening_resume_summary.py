"""Real short train + repeated skip must preserve completed artifacts and timing."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from research_runtime import code_snapshot, hash_file
from train_spatial import SpatialConfig

class ResumeSummaryTests(unittest.TestCase):
    def test_completed_then_two_resumes_and_missing_or_legacy_summary(self):
        cube=np.random.default_rng(9).normal(size=(60,60,6)).astype(np.float32)
        labels=np.tile(np.arange(1,10),400).reshape(60,60)
        for entry in ('scripts.run_coordinate_screening','scripts.run_m6_screening'):
            with self.subTest(entry=entry), tempfile.TemporaryDirectory() as directory:
                module=importlib.import_module(entry)
                config=SpatialConfig({'name':'resume_summary','dataset':'paviaU',
                    'model':{'graph_mode':'learned','latent_channels':8,'adapter_embed_dim':8,'norm_type':'group_norm'},
                    'training':{'epochs':0,'steps_per_epoch':2,'primary_checkpoint':'val_loss'},'tile_size':12,'gap':1})
                state=module.initial_state(config); digest=module.state_digest(state)
                root=Path(directory); code=code_snapshot(); loader=lambda:(cube,labels,{},None)
                first=module.run_one('none',config,state,digest,root,code,loader)
                folder=root/'training'/first['run_id']/'attempts'/first['attempt']
                original={p.name:hash_file(p) for p in folder.iterdir() if p.is_file()}
                # Distinct sentinel timings catch accidental overwrite by skip timing.
                first.update(elapsed_seconds=123.5,median_train_step_seconds=.25)
                (root/'none-result.json').write_text(json.dumps(first))
                for operation_attempt in ('0002','0003'):
                    with patch('torch.optim.Adam',side_effect=AssertionError('resume must not optimize')):
                        summary=module.run_one('none',config,state,digest,root,code,loader,True)
                    self.assertEqual(summary['status'],'completed')
                    self.assertEqual(summary['attempt'],'0001')
                    self.assertEqual(summary['verified_attempt'],'0001')
                    self.assertEqual(summary['operation_status'],'skipped')
                    self.assertEqual(summary['operation_attempt'],operation_attempt)
                    self.assertEqual(summary['elapsed_seconds'],123.5)
                    self.assertEqual(summary['median_train_step_seconds'],.25)
                    self.assertTrue((folder/'model.pt').is_file());self.assertTrue((folder/'metrics.json').is_file())
                # No summary, or one produced by the former buggy code: recover artifact identity.
                for legacy in (False,True):
                    if legacy:(root/'none-result.json').write_text(json.dumps({'status':'skipped','attempt':'0002'}))
                    else:(root/'none-result.json').unlink()
                    with patch('torch.optim.Adam',side_effect=AssertionError('must not optimize')):
                        summary=module.run_one('none',config,state,digest,root,code,loader,True)
                    self.assertEqual(summary['attempt'],'0001')
                    self.assertEqual(summary['elapsed_seconds_source'],'completed_manifest_wall_time')
                    self.assertIsNone(summary['median_train_step_seconds'])
                self.assertEqual(original,{p.name:hash_file(p) for p in folder.iterdir() if p.is_file()})
                before=(root/'none-result.json').read_bytes()
                # Corrupt only the disposable fixture. Verification must fail without replacing summary.
                with (folder/'metrics.json').open('a') as stream:stream.write('corrupted')
                with self.assertRaisesRegex(ValueError,'artifact missing or changed'):
                    module.run_one('none',config,state,digest,root,code,loader,True)
                self.assertEqual(before,(root/'none-result.json').read_bytes())

if __name__=='__main__':unittest.main()
