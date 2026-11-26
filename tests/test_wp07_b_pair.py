"""Exercise B's admission override and the actual unchanged production worker."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch

from research_runtime import code_snapshot
from scripts import run_wp07_b_pair as b
from tests.native_released_bank_fixture import fixture
from tests.test_native_released_production import release_for
from transfer import native_released_real as real, registered_development as dev
from transfer.native_scale_campaign import packet, read
from transfer.xuzhou_multiscale_audit import _exact
from transfer.spatial_multiscale import initialize
from transfer.native_scale_training import baseline


class BPairTests(TestCase):
    def test_fallback_accepts_only_missing_cnn_and_graph_metadata_layers(self):
        source = baseline(9, 64)
        for arm in ("mean", "content"):
            trained = initialize(source, arm, 7385)
            restored = b.missing_fallback(trained.state_dict(), source, arm, 7385)
            for name, value in trained.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]))
            damaged = dict(trained.state_dict())
            damaged.pop(next(name for name in damaged if "heads" in name))
            with self.assertRaisesRegex(ValueError, "non-metadata"):
                b.missing_fallback(damaged, source, arm, 7385)

    def test_both_gate_layers_observe_and_restore_on_error(self):
        original = b.resource.gate
        with patch.object(b.resource, "snapshot", return_value={"free_mib": 0, "utilization": 100}) as sample:
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                with b.observation_only_resources():
                    self.assertIs(b.worker.p4.gate, sample)
                    self.assertEqual(b.resource.gate("fixture")["free_mib"], 0)
                    with b.observation_only_resources():
                        self.assertEqual(b.resource.gate("fixture")["utilization"], 100)
                    raise RuntimeError("fixture failure")
        self.assertIs(b.resource.gate, original)

    def test_production_continuation_and_zero_update_reentry(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as name:
            root = Path(name).resolve()
            code = code_snapshot()
            pins, intake = fixture(root, code, design=real.DUTY)
            preflight = pins["xuzhou"]
            for arm in ("content", "cnn_content", "graph_content"):
                case_id = f"xuzhou/64/2/{arm}"
                case = real.registered_case(read(preflight, root), case_id, engineering=True)
                states = {}
                for route, phases in (("full", ("full", "completed")),
                                      ("split", ("prefix", "resume", "completed"))):
                    dest = root / f"{arm}-{route}"
                    approval = release_for(root, preflight, case, dest, code, route)
                    counts = []
                    for phase in phases:
                        with b.observation_only_resources():
                            value = b.worker.real_stage(root, dest, preflight, approval, case_id,
                                                        phase, code, None, intake=intake)
                        counts.append(read(value, root)["updates"])
                    self.assertEqual(sum(counts), case["planned_updates"])
                    self.assertEqual(counts[-1], 0)
                    states[route] = packet(read(dev._pin(dest / "completed.json"), root)["checkpoint"], root)
                    with self.assertRaises(ValueError):
                        b.worker.real_stage(root, dest, preflight, approval, case_id,
                                            "completed", code, None, intake=intake)
                self.assertTrue(all(_exact(states["full"][key], states["split"][key])
                                    for key in states["full"] if key != "registration"))
                state = states["split"]
                self.assertEqual(state["selected"]["step"], min(state["validation"], key=lambda x: x["fused_nll"])["step"])
                selector_events = [e for row in state["history"] for key, e in row["fresh"].items()
                                   if "selector" in key or "range_content" in key]
                self.assertTrue(selector_events)
                # This small fixture collapses graph radii to identical supports;
                # a selector need not update when its candidates are identical.
                if arm != "graph_content":
                    self.assertTrue(any(e["changed"] for e in selector_events))

    def test_preparation_rejects_synthetic_preflight_before_output(self):
        with TemporaryDirectory() as name:
            root = Path(name).resolve()
            code = code_snapshot()
            pins, _ = fixture(root, code, design=real.DUTY)
            dest = root / "b-real"
            with self.assertRaises(ValueError):
                b.prepare(root, dest, pins["xuzhou"], ["GPU-" + "0"*36]*2, code)
            self.assertFalse(dest.exists())
