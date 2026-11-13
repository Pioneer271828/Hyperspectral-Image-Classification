"""Execute the fixed upstream scoring path and the original authenticated CLI chain."""

from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluation.open_set import OpenSetScorer, detection_metrics
from evaluation.wp09_openood import evaluate_approval
from evaluation.wp09_upstream import OpenOODReference
from research_runtime import write_json
from test_wp09_known_admission import Pipeline, _args, _read
from transfer.registered_development import _pin


SOURCE = os.environ.get("WP09_OPENOOD_SOURCE")


@unittest.skipUnless(SOURCE, "Pinned OpenOOD source required")
class OpenOODTests(unittest.TestCase):
    def test_actual_msp_ebo_roc_orientation_and_boundaries(self):
        provider=OpenOODReference(SOURCE)
        known=[dict(id=i,name=str(i)) for i in (19,2,7)]
        unknown=[dict(id=31,name="open")]
        logits=np.array([[1,2,3],[3,3,1],[0,0,0],[7,2,1],[4,8,1]],dtype=np.float64)[:,::-1]
        for method in ("msp","energy"):
            for temperature in (1e-4,1.,1e4):
                scorer=OpenSetScorer.freeze(method,known,unknown,model_sha256="1"*64,
                                             representation="fixture",temperature=temperature)
                actual, guess=provider.compare_logits(scorer,logits,input_kind="raw_logits",column_ids=[19,2,7])
                self.assertTrue(np.isfinite(actual).all())
                with self.assertRaises(ValueError):
                    provider.compare_logits(scorer,logits,input_kind="probabilities",column_ids=[19,2,7])
                with self.assertRaises(ValueError):
                    provider.compare_logits(scorer,logits,input_kind="raw_logits",column_ids=[2,19,7])
                for bad in (logits[:0],logits*np.nan,logits+np.inf):
                    with self.assertRaises(ValueError):
                        provider.compare_logits(scorer,bad,input_kind="raw_logits",column_ids=[19,2,7])
            shift = provider.compare_logits(scorer,logits+3.,input_kind="raw_logits",column_ids=[19,2,7])[0]
            np.testing.assert_allclose(shift, actual-(3. if method=="energy" else 0.),atol=1e-11)
        y=np.array([2,2,7,-1,-1])
        for s in (np.ones(5),np.array([.1,.8,.5,.5,.9]),np.array([.1,.2,.3,.7,.9])):
            local=detection_metrics(s,y==-1)
            got=provider.metrics_module.auc_and_fpr_recall(-s,y,.95)
            self.assertAlmostEqual(got[0],local["auroc"])
            self.assertAlmostEqual(got[3],local["fpr95"])
        self.assertIsNone(detection_metrics(np.ones(3),np.array([True]*3))["fpr95"])
        trap = np.array([.1,.2,.85,.7,.9])
        y = np.array([2,2,7,-1,-1])
        self.assertNotEqual(provider.metrics_module.fpr_recall(-trap,y,.95)[0],
                            provider.metrics_module.auc_and_fpr_recall(-trap,y,.95)[3])

    def test_v2_frozen_artifacts_and_separate_release(self):
        evidence=os.environ.get("WP09_OPENOOD_FIXTURE_DIR")
        temp=None if evidence else tempfile.TemporaryDirectory()
        if temp:
            self.addCleanup(temp.cleanup)
        root=Path(evidence or temp.name)
        root.mkdir(exist_ok=True)
        pipeline=Pipeline(root)
        pipeline.train_and_admit()
        args=[*_args("authority",pipeline.authority),*_args("source-receipt",pipeline.source),
            "--expected-commit",pipeline.sha,"--openood",SOURCE]
        command=["scripts/review_wp09_openood.py"]
        prepared=pipeline.cli("ood-prepare",[*command,"prepare"],*args,"--out",str(root/"openood"))
        self.assertFalse(pipeline.label_path.exists())
        forged=root/"forged"
        forged.mkdir()
        original=_read(prepared)
        for mutation in ("columns","source_receipt","provider"):
            bad=dict(original)
            value=bad[mutation]
            bad[mutation]=list(reversed(value)) if mutation=="columns" else dict(value,wrong=True)
            f=forged/(mutation+".json")
            write_json(f,bad)
            org_read=Path.read_bytes
            def deny(path):
                if path==pipeline.label_path:
                    self.fail("Unapproved read of held-back labels")
                return org_read(path)
            with patch.object(Path,"read_bytes",deny),self.assertRaises(ValueError):
                evaluate_approval(_pin(f),pipeline.authority,pipeline.source,SOURCE,
                                  out=forged/"unexpected",expected_commit=pipeline.sha)
        release=pipeline.label_bytes
        self.assertEqual(sha256(release).hexdigest(),original["released_labels"]["sha256"])
        pipeline.label_path.write_bytes(release)
        # The E4c export reauthenticates the released file; the second reader
        # independently checks the metrics using the exact same pinned bytes.
        audit=evaluate_approval(prepared,pipeline.authority,pipeline.source,SOURCE,
                                out=root/"scored",expected_commit=pipeline.sha)
        summary=_read(audit)
        self.assertEqual(len(summary["methods"]),9)
        self.assertEqual(len([x for x in summary["methods"] if "openood" in x]),6)
        self.assertEqual({x["metrics"]["unlabeled_excluded"] for x in summary["methods"]},{2})
        evidence = dict(native_training_updates=2, scoring_updates=0, approved=prepared, audit=audit,
                        label_hash=sha256(release).hexdigest(), commands=pipeline.commands)
        write_json(root/"fixture-summary.json",evidence)
