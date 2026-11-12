"""Exercise forged synthetic evidence joins against frozen campaign artifacts."""

from copy import deepcopy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from evaluation import wp12_controlled_graph_evidence as evidence
from research_runtime import hash_file


@unittest.skipUnless(os.environ.get("WP06_CONTROLLED_RUN"), "requires frozen controlled evidence")
class ControlledGraphEvidenceTests(unittest.TestCase):
    def test_valid_controlled_evidence_and_forged_joins(self):
        run=Path(os.environ["WP06_CONTROLLED_RUN"])
        ctl=Path(os.environ["WP06_CONTROLLED_CTL"])
        def pin(path):
            return dict(path=str(path),sha256=hash_file(path))
        spec={"registration":pin(run/"registration.json"),"audit":pin(run/"audit.json"),
              "oracle":pin(ctl/"oracle.json"),"development_guard":pin(ctl/"development-guard.json")}
        value=evidence.adapt(spec)
        self.assertEqual(value["actual_optimizer_updates"],{"signal":14400,"evolution":4320})
        self.assertEqual(value["real_model_rows_added"],0)
        self.assertEqual(value["support_engineering"]["cases"],18)
        read=evidence.read
        def rejects(key,mutate,expected):
            original=read(spec[key])
            forged=deepcopy(original)
            mutate(forged)
            with patch.object(evidence,"read",side_effect=lambda p:forged if p==spec[key] else read(p)):
                with self.assertRaisesRegex(ValueError,expected):
                    evidence.adapt(spec)
        rejects("audit",lambda x:x.update(commit="0"*40),"producer")
        rejects("audit",lambda x:x["cases"].pop(next(iter(x["cases"]))),"roster")
        rejects("registration",lambda x:x["maximum"].update(signal=14401),"budget")
        rejects("development_guard",lambda x:x.update(model_updates=1),"permissions")
        rejects("audit",lambda x:x["cases"][next(iter(x["cases"]))].update(confirmed_once=False),"confirmation")
        with self.assertRaisesRegex(ValueError,"identity changed"):
            evidence.adapt(dict(spec,audit=dict(spec["audit"],sha256="0"*64)))
