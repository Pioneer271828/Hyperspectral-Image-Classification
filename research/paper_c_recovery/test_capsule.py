"""Compact recovery retains complete selected/current states and explicit gaps."""
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import capsule
import verify_capsule
from test_transactions import Toy,contract,equal,state
from transactions import execute,new_json,sha,inspect


class PriorityRecovery(unittest.TestCase):
    def test_compact_state_and_manifest_restore_without_claiming_all_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);suite=root/'suite';suite.mkdir();run=suite/'execution';run.mkdir()
            pin=root/'pin';pin.write_text('synthetic immutable input');c=contract(pin)
            execute(run/'toy',c,Toy);new_json(run/'registration.json',{'producer':'synthetic','cases':[{'id':'toy'}]})
            new_json(run/'summary.json',{'complete':True})
            archive=root/'capsule.tar.gz'
            with patch.object(sys,'argv',['capsule.py','--suite',str(suite),'--run',str(run),'--out',str(archive)]):
                with patch('sys.stdout',new_callable=io.StringIO):capsule.main()
            unpacked=root/'unpacked';unpacked.mkdir()
            with tarfile.open(archive) as t:
                for item in t:
                    p=unpacked/item.name;p.parent.mkdir(parents=True,exist_ok=True)
                    with t.extractfile(item) as src,p.open('xb') as dst:shutil.copyfileobj(src,dst)
            index=json.loads((unpacked/'object-index01.json').read_text());restored=unpacked/'restored';restored.mkdir()
            for row in index['files']:
                p=restored/row['name'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(unpacked/'objects'/row['sha256'],p)
                self.assertEqual(sha(p),row['sha256'])
            receipt=root/'verified.json'
            with patch.object(sys,'argv',['verify_capsule.py','--unpacked',str(unpacked),'--out',str(receipt)]):
                with patch('sys.stdout',new_callable=io.StringIO):verify_capsule.main()
            verified=json.loads(receipt.read_text());self.assertEqual(verified['current_states'],1)
            self.assertGreater(verified['omitted_intermediate_checkpoint_files'],0)
            self.assertTrue(equal(state(run/'toy',9),state(restored/'execution'/'toy',9)))
            self.assertFalse(verified['exact_cross_host_optimizer_continuation_exercised'])
            # Original exhaustive run inspection must not silently accept a
            # capsule as if every historical state.pt had been restored.
            with self.assertRaises(ValueError):inspect(restored/'execution'/'toy',c)


if __name__=='__main__':unittest.main()
