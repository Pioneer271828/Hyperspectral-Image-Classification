"""Zero-update checks for selector replay, native integration and metadata boundaries."""

from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from evaluation.native_support_reference import support_trace
from loadData.observation_sampling import BandMetadata
from model.research_config import digest
from research_runtime import tensor_hash
from transfer.conditional_multiscale import ObservationConditionEncoder
from transfer.curriculum_exposure import ExposurePlan
from transfer.g3_training import _forward
from transfer.multiscale_training import MultiscaleTrial
from transfer.native_support import NativeSupportModel, make_bank, make_selector, native_forward
from transfer.native_support_campaign import _fixture
from transfer.regional_support_evolution import Schedule, development_context
from transfer.regional_support_v2 import RegionalSupport


class RegionalSupportContractTests(unittest.TestCase):
    def inputs(self, role="devtrain"):
        # Stable IDs are deliberately noncontiguous; rows include ties, empty
        # remote eligibility, mandatory pairs, and one-row candidate pools.
        ids = [79, 3, 501, 14, 101, 33, 703, 99]
        partitions = [1, 1, 1, 1, 1, 2, 2, 3]
        adjacency = torch.eye(8)
        adjacency[0, 1:4] = .5
        adjacency[1, 2] = .5
        adjacency[2, 1] = .5
        adjacency[5, 6] = .5
        x = torch.tensor([[0., 0.], [1., 2.], [1., 2.], [-1., 2.], [2., 0.],
                          [1., 0.], [1., 0.], [0., 0.]])
        context = development_context(role=role, block="8", node_ids=ids, partitions=partitions)
        identity = dict(scene="fixture", grid=[2, 4], **{key: "a" * 64 for key in
                        ("source", "bands", "assignment", "block_input", "admission")})
        schedule = Schedule(3, 2, 2, 2, 3, 12, pool=1)
        return dict(adjacency=adjacency, spectra=x, graph_features=x.flip(1), context=context,
                    identity=identity, mandatory=[[0, 1]], schedule=schedule, seed=512)

    def test_independent_candidate_modes_and_jump_replay(self):
        for mode in ("local", "random", "spectral", "graph", "joint"):
            with self.subTest(mode=mode):
                data = self.inputs()
                live = RegionalSupport(**data, mode=mode)
                sequential = RegionalSupport(**data, mode=mode)
                alternate = RegionalSupport(**data, mode=mode)
                ref = support_trace(data["adjacency"].numpy(), data["spectra"].numpy(), data["graph_features"].numpy(),
                    node_ids=data["context"]["node_ids"], partitions=data["context"]["partitions"],
                    mandatory=data["mandatory"], schedule=asdict(data["schedule"]), mode=mode, seed=data["seed"], steps=12)
                for step in range(13):
                    sequential.advance(step)
                    self.assertEqual({k: sequential.snapshot()[k] for k in ref[step]}, ref[step])
                early = live.snapshot()
                self.assertLessEqual(max(np.bincount(np.asarray(early["selected"])[:, 0])), 3)
                live.advance(12)
                alternate.advance(6)
                resumed = RegionalSupport(**data, mode=mode)
                resumed.restore(alternate.snapshot())
                resumed.advance(12)
                self.assertEqual(live.snapshot(), sequential.snapshot())
                self.assertEqual(live.snapshot(), resumed.snapshot())
                with self.assertRaisesRegex(ValueError, "active selected"):
                    live.render_live(data["adjacency"], data["spectra"], support=early["selected"])
                previous = live.render_snapshot(data["adjacency"], data["spectra"], early)
                self.assertEqual(previous._nnz(), len(early["selected"]))
                self.assertEqual(live.snapshot(), sequential.snapshot())
                self.assertTrue(torch.equal(previous.to_dense(), live.render_snapshot(
                    data["adjacency"].to_sparse(), data["spectra"].T.contiguous().T, early).to_dense()))

    def test_reject_duplicate_cross_partition_budget_and_modified_state(self):
        data = self.inputs()
        graph = data["adjacency"].to_sparse().coalesce()
        idx, weights = graph.indices(), graph.values()
        duplicate = torch.sparse_coo_tensor(torch.cat((idx, idx[:, :1]), 1),
                                           torch.cat((weights, weights[:1])), graph.shape)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            RegionalSupport(**{**data, "adjacency": duplicate}, mode="joint")
        crossed = data["adjacency"].clone()
        crossed[7, 2] = .25
        with self.assertRaisesRegex(ValueError, "Cross-partition"):
            RegionalSupport(**{**data, "adjacency": crossed}, mode="joint")
        with self.assertRaisesRegex(ValueError, "Mandatory"):
            RegionalSupport(**{**data, "mandatory": [[0, 1], [0, 2]]}, mode="joint")
        live = RegionalSupport(**data, mode="joint")
        live.advance(8)
        changed = {**data, "spectra": data["spectra"] + .125}
        with self.assertRaisesRegex(ValueError, "context differs"):
            RegionalSupport(**changed, mode="joint").restore(live.snapshot())
        wrong = live.snapshot()
        wrong["budget"] += 1
        wrong["state_sha256"] = digest({k: v for k, v in wrong.items() if k != "state_sha256"})
        with self.assertRaisesRegex(ValueError, "deterministic replay"):
            RegionalSupport(**data, mode="joint").restore(wrong)
        validation = RegionalSupport(**self.inputs("devval"), mode="joint")
        with self.assertRaisesRegex(ValueError, "cannot advance"):
            validation.advance(1)
        with self.assertRaisesRegex(ValueError, "Global support step"):
            live.advance(13)

    def test_current_a0_is_differentiable_and_rows_normalize(self):
        data = self.inputs()
        live = RegionalSupport(**data, mode="joint")
        live.advance(3)
        current = data["adjacency"].clone().requires_grad_(True)
        descriptor = data["spectra"].clone().requires_grad_(True)
        graph = live.render_live(current, descriptor)
        self.assertTrue(torch.allclose(graph.to_dense().sum(1), torch.ones(8)))
        (graph.values() * torch.arange(graph._nnz())).sum().backward()
        self.assertGreater(float(current.grad.abs().sum()), 0.)
        self.assertGreater(float(descriptor.grad.abs().sum()), 0.)


class NativeIntegrationTests(unittest.TestCase):
    def test_native_source_capture_matches_original_three_head_forward(self):
        with TemporaryDirectory() as name, torch.no_grad():
            pins, baseline, scene, record = _fixture(Path(name) / "fixture", 9, 8, 64, trusted=False)
            plan = ExposurePlan.freeze(scene, record, updates=48, seed=64,
                minimum_class_blocks={str(c): 2 for c in record["classes"]}, minimum_support_visits=1)
            rng = torch.get_rng_state().clone()
            model = NativeSupportModel.initialize(baseline, 7321)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            bank = make_bank(baseline, scene, record, plan, pins["development"]["sha256"], 7321)
            self.assertEqual(sum(entry["context"]["role"] == "devtrain" for entry in bank.values()), 2)
            self.assertEqual(sum(entry["context"]["role"] == "devval" for entry in bank.values()), 1)
            for key, entry in bank.items():
                tile = scene.plan["tiles"][int(key)]
                before = _forward(baseline, scene, tile)
                captured, frame = native_forward(model, scene, tile, capture=True)
                self.assertEqual([tensor_hash(x) for x in before[:3]], [tensor_hash(x) for x in captured[:3]])
                self.assertEqual(tensor_hash(frame["A0"]), tensor_hash(entry["adjacency"]))
                self.assertEqual(tensor_hash(frame["Q"]), entry["identity"]["assignment"])
                selector = make_selector(entry, {"mode": "joint",
                    "schedule": asdict(Schedule(10, 6, 3, 7, 6, 48, pool=5))}, 19100)
                output, trace = native_forward(model, scene, tile, selector=selector, detail=True)
                self.assertEqual(trace["trace"]["snapshot"]["step"], 0)
                self.assertEqual(output[0].shape[-1], 9)
            variant = MultiscaleTrial(graph_multiscale="fixed", graph_radius=2).initialize(baseline, 64).eval()
            self.assertEqual(variant.backbone.learnable_graph.graph_offsets.shape, (25, 2))
            self.assertEqual(variant.export_config()["backbone_kwargs"]["learned_graph_radius"], 2)
            for key, value in baseline.state_dict().items():
                self.assertTrue(torch.equal(value, variant.state_dict()[key]))
            with self.assertRaisesRegex(ValueError, "unquantized"):
                NativeSupportModel.initialize(variant, 7321)

    def test_metadata_permutation_missing_and_identity(self):
        bands = BandMetadata("synthetic", (1, 2, 3), (500., 650., 850.), "nm", "verified",
                             (True, True, True), (7., 11., 15.), "nm")
        encoder = ObservationConditionEncoder()
        reference = torch.zeros(1)
        original = encoder(bands, .8, 2.5, reference)
        self.assertTrue(torch.equal(original, encoder(bands.subset([2, 0, 1]), .8, 2.5, reference)))
        self.assertTrue(torch.equal(original, encoder(replace(bands, source_id="other"), .8, 2.5, reference)))
        self.assertFalse(torch.equal(original, encoder(replace(bands, wavelengths=(515., 670., 840.)), .8, 2.5, reference)))
        self.assertFalse(torch.equal(original, encoder(replace(bands, fwhm=None, fwhm_unit=None), .8, 2.5, reference)))
        self.assertIsNone(encoder(replace(bands, wavelength_quality="approximate"), None, None, reference))
        with self.assertRaisesRegex(ValueError, "Unique"):
            replace(bands, band_ids=(1, 1, 3))
        with self.assertRaisesRegex(ValueError, "at least one"):
            replace(bands, valid=(False, False, False))


if __name__ == "__main__":
    unittest.main()
