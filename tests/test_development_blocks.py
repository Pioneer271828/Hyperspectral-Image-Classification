"""Publication, input boundary and existing-consumer regression for WP02 blocks."""

import io
from pathlib import Path
from dataclasses import asdict, replace
import tempfile
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

import numpy as np
from scipy.io import savemat
import torch

from loadData.observation_sampling import BandMetadata
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.research_config import canonical_json, digest
from research_runtime import hash_file
from transfer import development_blocks as blocks, development_block_probe as probe
from transfer.development_mat import open_guarded
from transfer.g3_training import state_hash
from transfer.native_support import NativeSupportModel, capture_context, make_selector, native_forward
from transfer.regional_support_evolution import Schedule


def fixture(scene, classes, bands, root):
    shape = (24, 24, bands)
    box_a, box_b = [2, 10, 2, 10], [2, 10, 12, 20]
    roles = (dict(id=3, role="devtrain", partition="train", box=box_a, parent_tile=3,
                  samples=[[2 * 24 + 2, 1]], origin={"fixture": "first"}),
             dict(id=89, role="devval", partition="val", box=box_b, parent_tile=81,
                  samples=[[2 * 24 + 12, classes]], origin={"fixture": "second"}))
    physical = BandMetadata(scene, tuple(range(bands)), tuple(np.linspace(480, 830, bands)),
                            "nm", "approximate", (True,) * bands)
    stats = dict(schema_version=1, fit_partition="train", plan_hash="fixture-plan",
                 minimum=[-4.] * bands, maximum=[4.] * bands)
    stats["statistics_hash"] = digest(stats)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(643)
        model = MultiDatasetVCASAHGCN({scene: classes}, graph_mode="learned", norm_type="group_norm",
                                     latent_channels=128, adapter_embed_dim=64).eval()
    model.backbone.adapter.cache_attention = False
    state = state_hash(model)
    release = blocks.Release(scene, "engineering_fixture", shape, classes, "fixture-plan", physical, stats,
                             roles, {"minimum_buffer": 2}, {"gsd_status": "missing"},
                             {"state": state, "shared_state": state})
    rng = np.random.default_rng(972 + classes)
    arrays = {row["id"]: rng.normal(size=(8, 8, bands)).astype(np.float32) for row in roles}
    roles = tuple(dict(row, origin_content_sha256=blocks.array_hash(arrays[row["id"]])) for row in roles)
    return replace(release, roles=roles), model, arrays


class DevelopmentBlocksTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name).resolve()
        self.release, self.model, self.arrays = fixture("paviaU", 9, 7, self.root)

    def publish(self, release=None, arrays=None, name="release.zip"):
        release = self.release if release is None else release
        arrays = self.arrays if arrays is None else arrays
        return blocks.publish(self.root / name, release, lambda row: arrays[row["id"]])

    def tamper(self, pin, transform):
        path = self.root / "modified.zip"
        with ZipFile(pin["path"]) as original, ZipFile(path, "w") as output:
            for name in original.namelist():
                output.writestr(name, transform(name, original.read(name)))
        return {"path": str(path), "manifest_sha256": pin["manifest_sha256"]}

    def test_atomic_no_overwrite_and_repeat_reads(self):
        pin = self.publish()
        before = hash_file(pin["path"])
        with self.assertRaises(FileExistsError):
            blocks.publish(pin["path"], self.release, lambda _: self.fail("Decoded despite existing destination"))
        opened = blocks.Blocks(pin, self.root, self.release)
        for _ in range(2):
            a = opened.read(89, "devval", [2, 10, 12, 20])
            np.testing.assert_array_equal(a, self.arrays[89])
            self.assertFalse(a.flags.writeable)
        recovered = blocks.recover(pin["path"], self.root, self.release, lambda row: self.arrays[row["id"]])
        self.assertEqual(recovered, pin)
        self.assertEqual(hash_file(pin["path"]), before)

    def test_recovery_checks_existing_manifest_before_any_array_reader(self):
        pin = self.publish()
        first, second = self.release.roles
        invalid = (
            replace(self.release, roles=(dict(first, role="test"), second)),
            replace(self.release, roles=(dict(first, box=[2, 25, 2, 10]), second)),
            replace(self.release, bands=self.release.bands.subset(tuple(reversed(range(7))))),
            replace(self.release, source={"state": "wrong-source", "shared_state": "wrong-source"}),
            replace(self.release, roles=(first, dict(second, box=[3, 11, 12, 20], samples=[[3 * 24 + 12, 9]]))),
        )
        source = Mock(side_effect=AssertionError("read an array before validating custody"))
        with patch.object(blocks.np, "load", side_effect=AssertionError("decoded unaccepted archive member")):
            for release in invalid:
                with self.subTest(release=release), self.assertRaises(ValueError):
                    blocks.recover(pin["path"], self.root, release, source)
            safe = self.root / "authorized"
            safe.mkdir()
            with self.assertRaisesRegex(ValueError, "escapes"):
                blocks.recover(pin["path"], safe, self.release, source)
            altered = self.tamper(pin, lambda name, raw: raw.replace(b'"devtrain"', b'"bad_role"')
                                  if name == "development.json" else raw)
            with self.assertRaisesRegex(ValueError, "manifest"):
                blocks.recover(altered["path"], self.root, self.release, source)
        source.assert_not_called()

    def test_wrong_request_does_not_read_or_decode_arrays(self):
        opened = blocks.Blocks(self.publish(), self.root, self.release)
        with patch.object(blocks, "ZipFile", side_effect=AssertionError("opened unauthorized container")), (
                patch.object(blocks.np, "load", side_effect=AssertionError("decoded unauthorized array"))):
            for wrong in ((3, "devval", [2, 10, 2, 10]), (81, "devval", [2, 10, 12, 20]),
                          (89, "devval", [2, 10, 11, 20]), (True, "devtrain", [2, 10, 2, 10]),
                          (89, "test", [2, 10, 12, 20])):
                with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                    opened.read(*wrong)

    def test_manifest_role_bands_and_label_identity_rejected_before_payload(self):
        pin = self.publish()
        first, second = self.release.roles
        mismatches = [
            replace(self.release, bands=self.release.bands.subset(tuple(reversed(range(7))))),
            replace(self.release, roles=(dict(first, samples=[[2 * 24 + 3, 1]]), second)),
            replace(self.release, roles=(dict(first, partition="test"), second)),
            replace(self.release, roles=(dict(first, box=[-1, 10, 2, 10]), second)),
            replace(self.release, roles=(first, dict(second, box=[2, 10, 9, 17]))),
            replace(self.release, source={"state": "altered"}),
        ]
        with patch.object(blocks.np, "load", side_effect=AssertionError("decoded before checking manifest")):
            for release in mismatches:
                with self.subTest(release=release), self.assertRaises(ValueError):
                    blocks.Blocks(pin, self.root, release)
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                blocks.Blocks(dict(pin, manifest_sha256="0" * 64), self.root, self.release)

    def test_altered_member_rejected_before_numpy(self):
        pin = self.publish()
        def transform(name, data):
            return data[:-1] + bytes((data[-1] ^ 1,)) if name == "tile-3.npy" else data
        modified = self.tamper(pin, transform)
        opened = blocks.Blocks(modified, self.root, self.release)
        with patch.object(blocks.np, "load", side_effect=AssertionError("decoded hash mismatch")):
            with self.assertRaisesRegex(ValueError, "file hash"):
                opened.read(3, "devtrain", [2, 10, 2, 10])
        # Only the requested file is opened. The valid sibling remains readable.
        np.testing.assert_array_equal(opened.read(89, "devval", [2, 10, 12, 20]), self.arrays[89])

    def test_declared_content_digest_must_match_custody_before_decode(self):
        pin = self.publish()
        import json
        new_hash = None

        def transform(name, data):
            nonlocal new_hash
            if name == "development.json":
                raw = json.loads(data)
                raw["members"][0]["array_sha256"] = "0" * 64
                data = (canonical_json(raw) + "\n").encode()
                new_hash = blocks.transport._sha(data)
            return data

        modified = self.tamper(pin, transform)
        with patch.object(blocks.np, "load", side_effect=AssertionError("decoded forged origin")):
            with self.assertRaisesRegex(ValueError, "FP32 development member"):
                blocks.Blocks(dict(modified, manifest_sha256=new_hash), self.root, self.release)

    def test_invalid_data_and_duplicate_role_content_leave_no_publication(self):
        nan = self.arrays[3].copy()
        nan[0, 0, 0] = np.nan
        for altered in (nan, self.arrays[3].astype(np.float64), self.arrays[3][:, :, :-1], self.arrays[89]):
            with self.subTest(dtype=altered.dtype), self.assertRaises(ValueError):
                self.publish(arrays={**self.arrays, 3: altered})
            self.assertEqual(list(self.root.iterdir()), [])

    def test_sample_and_alias_conflicts(self):
        self.assertEqual(blocks.scene_roster(["PaviaU", "WHU_HongHu", "XUZHOU"]),
                         ["paviaU", "whu-honghu", "xuzhou"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            blocks.scene_roster(["HongHu", "WHUH"])
        first, second = self.release.roles
        for points in ([[2 * 24 + 12, 1]], [[2 * 24 + 2, 1], [2 * 24 + 2, 2]], [[2 * 24 + 2, 10]]):
            with self.assertRaises(ValueError):
                replace(self.release, roles=(dict(first, samples=points), second)).identity()

    def test_symlink_and_outside_root_before_archive_open(self):
        pin = self.publish()
        safe = self.root / "safe"
        safe.mkdir()
        link = safe / "escape.zip"
        link.symlink_to(pin["path"])
        with patch.object(blocks, "ZipFile", side_effect=AssertionError("opened unauthorized file")):
            for path in (link, Path(pin["path"])):
                with self.assertRaises(ValueError):
                    blocks.Blocks(dict(pin, path=str(path)), safe, self.release)

    def test_actual_band_deletion_and_per_view_native_graph(self):
        for release, model, arrays in (
                (self.release, self.model, self.arrays), fixture("whu-honghu", 22, 10, self.root)):
            with self.subTest(scene=release.scene):
                pin = self.publish(release, arrays, release.scene + ".zip")
                opened = blocks.Blocks(pin, self.root, release)
                state = state_hash(model)
                result = probe.forward_blocks(opened, model, pin)
                self.assertEqual(state_hash(model), state)
                self.assertEqual(result["actual_forwards"], 32)
                self.assertEqual(result["fallback"], {"metadata": "mean", "joint": "content"})
                for row in result["rows"]:
                    current = next(item for item in release.roles if item["id"] == row["context"]["tile"])
                    _, tile, observed, context = blocks.view(opened, current, probe.VIEWS[row["condition"]])
                    expected = arrays[tile["id"]][:, :, context["selection"]["selected_positions"]]
                    np.testing.assert_array_equal(observed.observation.spectral, (expected + 4.) / 8.)
                    self.assertEqual(row["context"], context)
                    self.assertEqual(row["graph"]["native_step_zero"]["snapshot"]["step"], 0)
                    self.assertEqual(row["probabilities"]["off"]["fused"]["shape"], [64, release.classes])
                with self.assertRaisesRegex(ValueError, "Caller source"):
                    with torch.no_grad():
                        next(model.parameters()).add_(1)
                    probe.forward_blocks(opened, model, pin)

    def test_selector_rejects_a_different_materialized_view_before_forward(self):
        pin = self.publish()
        opened = blocks.Blocks(pin, self.root, self.release)
        role = self.release.roles[1]
        scene, tile, view, _ = blocks.view(opened, role, probe.VIEWS["full"])
        native = NativeSupportModel.initialize(self.model, probe.SEED)
        entry = capture_context(native, scene, role, "1" * 64, state_hash(self.model), observation_view=view)
        selector = make_selector(entry, dict(mode="joint", schedule=asdict(Schedule(10, 6, 3, 7, 6, 48, pool=5))),
                                 probe.SUPPORT_SEED)
        raw = scene.cube[blocks.tile_slice(tile)].copy()
        changed_cube = blocks.BlockCube(scene.cube.shape, tile, raw + 0.01)
        changes = (
            ("omitted view", scene, None),
            ("different scene", replace(scene, dataset="xuzhou"), view),
            ("different raw", replace(scene, cube=changed_cube), view),
            ("different statistics", replace(scene, stats=dict(scene.stats, minimum=[-3.] * 7)), view),
            ("different spectral", scene,
             replace(view, observation=replace(view.observation, spectral=view.observation.spectral + 0.01))),
            ("different coordinates", scene,
             replace(view, observation=replace(view.observation, coordinates=view.observation.coordinates + 0.01))),
            ("different ids", scene,
             replace(view, observation=replace(view.observation, global_ids=view.observation.global_ids + 1))),
            ("different plan", scene,
             replace(view, observation=replace(view.observation, plan_hash="different-plan"))),
        )
        with patch.object(native, "forward", side_effect=AssertionError("forward used a different observation")):
            for name, candidate_scene, candidate_view in changes:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Selector"):
                    native_forward(native, candidate_scene, tile, selector=selector, observation_view=candidate_view)
        output, frame = native_forward(native, scene, tile, selector=selector, observation_view=view)
        self.assertEqual(tuple(output[0].shape), (64, 9))
        self.assertEqual(frame["trace"]["q"], entry["identity"]["assignment"])

    def test_bounded_mat_read_and_role_reject(self):
        array = np.arange(np.prod(self.release.shape), dtype=np.float32).reshape(self.release.shape)
        path = self.root / "guarded.mat"
        savemat(path, {"guarded_variable": array}, do_compression=False)
        pin = {"path": str(path), "sha256": hash_file(path)}
        reader = open_guarded(pin, self.root, "guarded_variable", array.shape, self.release.roles)
        np.testing.assert_array_equal(reader(self.release.roles[0]), array[2:10, 2:10])
        np.testing.assert_array_equal(reader(self.release.roles[1]), array[2:10, 12:20])
        with patch.object(Path, "open", side_effect=AssertionError("opened rejected MAT")):
            with self.assertRaisesRegex(ValueError, "escapes"):
                reader(dict(self.release.roles[1], role="devtrain"))
        with self.assertRaises(ValueError):
            open_guarded(pin, self.root, "different_variable", array.shape, self.release.roles)
        second = self.root / "compressed.mat"
        savemat(second, {"guarded_variable": array}, do_compression=True)
        with self.assertRaisesRegex(ValueError, "uncompressed"):
            open_guarded({"path": str(second), "sha256": hash_file(second)}, self.root,
                         "guarded_variable", array.shape, self.release.roles)


if __name__ == "__main__":
    unittest.main()
