"""Exercise the two-factor INT8 and frozen deployment paths on accepted fixtures."""

import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch

from compression.int8 import CalibratedMatrix, IntegerMatrix
from compression.int8_model import calibrate_model, calibration_contract, convert_model
from compression.low_rank_factor_checkpoint import load, save
from compression.low_rank_factor_precision import calibrate, contract, deploy, fallback, validate, with_precision
from compression.low_rank_factor_qat import FactorPrecisionRun, new_run
from compression.low_rank_recovery import TARGET, ladder, transform
from compression.precision_cost import _state_identity
from inference import wp10_low_rank_execution as frozen
from model.research_config import digest
from research_runtime import ROOT, code_snapshot
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.curriculum_exposure import ExposurePlan
from transfer.g3_training import _forward, state_hash
from transfer.xuzhou_multiscale_audit import SCIENCE, _exact


@unittest.skipUnless(os.environ.get("WP10_LR_BASE_SYNTHETIC"), "requires completed rank ladder fixtures")
class LowRankDeploymentTests(unittest.TestCase):
    def test_composition_reload_fallback_and_qat_resume(self):
        torch.set_num_threads(1)
        with TemporaryDirectory() as directory:
            self.matrix(Path(directory))

    def matrix(self, root):
        code = code_snapshot()
        fixtures = Path(os.environ["WP10_LR_BASE_SYNTHETIC"])
        for classes in (9, 22):
            with self.subTest(classes=classes):
                source = fixtures / str(classes)
                path = root / str(classes)
                path.mkdir()
                self.combination(path, source)
                self.execution(path, source, code)

    def combination(self, path, source):
        spec = admission.read(dev._pin(source / "request.json"))
        dense = dev._load_source(spec["weights"], spec["sidecar"])
        scene, record, _ = dev.load_development(spec["development"])
        setting = ladder(dense)["arms"]["svd-r16"]
        factors = transform(dense, setting, 64)
        identity = state_hash(factors)
        original = state_hash(dense)
        train = [t["id"] for t in record["roles"] if t["role"] == "devtrain"][:2]
        val = next(t["id"] for t in record["roles"] if t["role"] == "devval")
        frozen_factor_state = factors.get_submodule(TARGET)
        self.assertEqual(state_hash(with_precision(factors, scene, record)), identity)
        with self.assertRaisesRegex(ValueError, "devtrain"):
            contract(factors, scene, record, "both", train + [val], qat=True)
        old_contract = calibration_contract(dense, scene, record, [TARGET], train)
        legacy = calibrate_model(dense, scene, record, paths=[TARGET], contract=old_contract, expected_hash=digest(old_contract))
        routed = with_precision(dense, scene, record, dense_paths=[TARGET], tile_ids=train)
        self.assertEqual(_state_identity(convert_model(legacy)), _state_identity(convert_model(routed)))

        candidates = {}
        for layout in ("both", "right", "left"):
            registration = contract(factors, scene, record, layout, train, qat=True)
            ready = calibrate(factors, scene, record, registration=registration, expected_hash=digest(registration))
            plan = validate(ready)
            self.assertEqual(set(plan["observers"]), set(registration["active"]))
            pair = ready.get_submodule(TARGET)
            self.assertEqual(pair.right.layer.bias if layout != "left" else pair.right.bias, None)
            for part in registration["active"]:
                self.assertEqual(int(getattr(pair, part).batches_seen), len(train))
            deployed = deploy(ready)
            path_to_save = path / f"{layout}.pt"
            pin = save(deployed, path_to_save)
            with patch.object(CalibratedMatrix, "observe", side_effect=AssertionError("unexpected calibration")):
                loaded = load(factors, path_to_save, pin, expected_contract_hash=digest(registration))
            self.assertEqual(_state_identity(loaded), _state_identity(deployed))
            traces = {}
            hooks = []
            for part in registration["active"]:
                kernel = getattr(loaded.get_submodule(TARGET), part)
                self.assertIs(type(kernel), IntegerMatrix)
                def seen(_, inputs, output, part=part):
                    self.assertTrue(inputs[0].is_quantized and output.is_quantized)
                    traces[part] = traces.get(part, 0) + 1
                hooks.append(kernel.kernel.register_forward_hook(seen))
            try:
                with torch.no_grad():
                    actual = _forward(loaded, scene, scene.plan["tiles"][val])
                    expected = _forward(deployed, scene, scene.plan["tiles"][val])
            finally:
                for hook in hooks:
                    hook.remove()
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual[:3], expected[:3], strict=True)))
            self.assertEqual(traces, dict.fromkeys(registration["active"], 1))
            candidates[layout] = loaded
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load(factors, path_to_save, "0"*64, expected_contract_hash=digest(registration))
            with self.assertRaisesRegex(ValueError, "original factorized"):
                wrong = transform(dense, ladder(dense)["arms"]["svd-r32"], 64)
                load(wrong, path_to_save, pin, expected_contract_hash=digest(registration))

        kwargs = dict(expected_dense_state=original, expected_factor_state=identity)
        for declared, allowed, expected in (
                (True, {"both": True, "right": True, "left": True}, "both"),
                (True, {"both": False, "right": False, "left": True}, "left"),
                (True, {"both": True, "right": True, "left": False}, "right"),
                (True, {"both": True}, "float_factors"), (False, {"left": True}, "dense")):
            selected, model = fallback(dense, factors, candidates, allowed, low_rank_passed=declared, **kwargs)
            self.assertEqual(selected, expected)
            if expected in ("dense", "float_factors"):
                self.assertEqual(state_hash(model), original if expected == "dense" else identity)
        with self.assertRaisesRegex(ValueError, "different upstream"):
            fallback(dense, factors, {"left": candidates["both"]}, {"left": True}, low_rank_passed=True, **kwargs)

        reg = contract(factors, scene, record, "both", train, qat=True)
        prepared = calibrate(factors, scene, record, registration=reg, expected_hash=digest(reg))
        initial_path = path / "qat-source.pt"
        checksum = save(prepared, initial_path)
        exposure = ExposurePlan.freeze(scene, record, updates=4, seed=64,
                                       minimum_class_blocks={str(c): 1 for c in record["classes"]}, minimum_support_visits=1)
        direct = new_run(prepared, scene, record, exposure)
        with patch.object(CalibratedMatrix, "observe", side_effect=AssertionError("resumed calibration")):
            direct.advance(4)
            prefix = new_run(prepared, scene, record, exposure)
            prefix.advance(2)
            pre = path / "prefix.pt"
            pre_hash = prefix.save(pre)
            reloaded_source = load(factors, initial_path, checksum, expected_contract_hash=digest(reg))
            resumed = FactorPrecisionRun.load(reloaded_source, scene, record, pre, plan=exposure, policy=None,
                                               expected_sha256=pre_hash, expected_contract_hash=direct.contract_hash)
            resumed.advance(4)
            full, split = path / "full.pt", path / "split.pt"
            direct.save(full)
            split_hash = resumed.save(split)
            with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("extra update")):
                finished = FactorPrecisionRun.load(reloaded_source, scene, record, split, plan=exposure, policy=None,
                                                     expected_sha256=split_hash, expected_contract_hash=direct.contract_hash)
                finished.advance(4)
            for key in SCIENCE:
                self.assertTrue(_exact(torch.load(full, weights_only=True)[key], torch.load(split, weights_only=True)[key]), key)
        self.assertEqual(state_hash(dense), original)
        self.assertEqual(state_hash(factors), identity)
        self.assertTrue(any((direct.model.get_submodule(TARGET).right.layer.weight
                             != prepared.get_submodule(TARGET).right.layer.weight).flatten().tolist()))

    def execution(self, path, source, code):
        root = source / "run"
        request = dev._json_new(path / "execution-request.json", dict(kind=frozen.KIND,
                        registration=dev._pin(root/"registration.json"), audit=dev._pin(root/"audit.json")))
        pin = frozen.freeze(request, code, path/"manifest")
        reg, groups = frozen.resident(pin, code)
        self.assertEqual(len(groups), 3)
        self.assertEqual(len(reg["class_ids"]), int(source.name))
        self.assertEqual(sum(g["kind"] == "expansion_control" for g in groups.values()), 1)
        name = "s64-svd-r16"
        argv = [sys.executable, str(ROOT/"scripts/run_frozen_cpu_deployment.py"), "--expected-commit", code["commit"],
                "run", "--input", pin["path"], pin["sha256"], "--mode", name, "--out", str(path/"inference"),
                "--save-predictions"]
        process = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        output = admission.read(dev._pin(path/"inference/execution.json"))
        self.assertEqual(output["factor_calls"], dict(right=len(reg["tiles"]), left=len(reg["tiles"])))
        self.assertEqual(output["output_hashes"], frozen.startup(pin, code, name)["output_hashes"])
        selected = reg["cases"][name]
        changed = dict(reg, cases=dict(reg["cases"], **{
            name: dict(selected, package=reg["cases"]["s64-dense"]["package"])
        }))
        forged = path / "manifest" / "forged.json"
        with self.assertRaisesRegex(ValueError, "audited recovery case or source"):
            frozen.execute(dev._json_new(forged, changed), code, name, path/"wrong-source")
        with self.assertRaisesRegex(ValueError, "registered low-rank"):
            frozen.execute(pin, code, "adaptive", path/"denied")
