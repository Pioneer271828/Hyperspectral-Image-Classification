from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import sys
import unittest

import torch

from scripts.run_native_released_points import arithmetic
from transfer import wp07_graph_training as train
from transfer.native_scale_campaign import packet
from transfer import registered_development as dev
from transfer.xuzhou_multiscale_audit import _exact


class FrozenRangeTrainingTests(unittest.TestCase):
    def test_update_restore_rejects_changed_frozen_graph_and_wrong_budget(self):
        torch.set_num_threads(1)
        provider = train.FixtureProvider()
        entries = train.fixture_cache(provider, "fixture")
        with TemporaryDirectory() as name, arithmetic(Path(name), None) as execution:
            root = Path(name)
            spec = train.contract(provider, 64, "content", "fixture", execution.numeric,
                                  {"fixture": "geometry"}, {"synthetic": True})
            path = root / "prefix.pt"
            train.execute(spec, "prefix", path, provider, entries, "fixture", execution)
            state = packet(dev._pin(path), root)
            run = train.RangeRun(spec, provider, entries, "fixture", execution)
            run.restore(state, 3)
            self.assertTrue(_exact(run.pack(), state))
            replaced = deepcopy(entries)
            first = next(iter(replaced))
            q = replaced[first]["q"].coalesce()
            replaced[first]["q"] = torch.sparse_coo_tensor(q.indices(), q.values()*0.99, q.shape).coalesce()
            with self.assertRaisesRegex(ValueError, "frozen graph"):
                train.RangeRun(spec, provider, replaced, "fixture", execution).restore(state, 3)
            wrong = {**spec, "steps": spec["steps"]+1}
            with self.assertRaisesRegex(ValueError, "budget differs"):
                train.RangeRun(wrong, provider, entries, "fixture", execution)

    def test_independent_audit_counts_only_released_points(self):
        torch.set_num_threads(1)
        provider = train.FixtureProvider()
        entries = train.fixture_cache(provider, "fixture")
        with TemporaryDirectory() as name, arithmetic(Path(name), None) as execution:
            root = Path(name)
            spec = train.contract(provider, 64, "mean", "fixture", execution.numeric,
                                  {"fixture": "geometry"}, {"synthetic": True})
            train.execute(spec, "full", root / "full.pt", provider, entries, "fixture", execution)
            state = packet(dev._pin(root / "full.pt"), root)
            result = train.audit(spec, state, provider, entries, "fixture", execution)
            self.assertEqual(result["audit_optimizer_updates"], 0)
            self.assertTrue(result["state_unchanged"])
            for endpoint in result["endpoints"].values():
                points = sum(row["released"]["points"] for row in endpoint["validation"]["rows"])
                area = sum(row["area"] for row in endpoint["validation"]["rows"])
                self.assertLess(points, area)
                self.assertEqual(len(endpoint["original_point_predictions"]), 3*points)
                self.assertEqual(sum(v["count"] for v in endpoint["metrics"]["fused"].values()), points)

    def test_optimized_training_cli_refused_before_execution(self):
        p = subprocess.run([sys.executable, "-O", "scripts/run_wp07_graph_pair.py", "--help"],
                           cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("refuses optimized Python", p.stderr)
