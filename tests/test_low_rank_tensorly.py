"""Attest the optional initializer and exercise the unchanged native matrix module."""

import copy
from pathlib import Path
import os
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from compression.low_rank import LowRankMatrix, factorize_model
from compression.low_rank_checkpoint import load_low_rank, save_low_rank
from compression.low_rank_recovery import HEADS, TARGET, LowRankRecoveryRun, initial_evidence, new_run
from compression.low_rank_tensorly import RANKS, TensorLySVD
from compression.registered_low_rank_recovery import _design
from research_runtime import write_json
from tests.test_precision_recovery import fixture
from transfer import matched_route_admission as admission, registered_development as dev
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale import SEED
from transfer.xuzhou_multiscale_audit import SCIENCE, _exact


SOURCE = os.environ.get("WP10_TENSORLY_SOURCE")


class DefaultPathTests(unittest.TestCase):
    def test_native_dependency_remains_optional(self):
        layer = nn.Conv2d(64, 128, 1)
        with patch("compression.low_rank_tensorly.util.find_spec", return_value=None):
            output = LowRankMatrix(layer, 16)(torch.ones(1, 64, 2, 3))
        self.assertEqual(tuple(output.shape), (1, 128, 2, 3))


@unittest.skipUnless(SOURCE, "Explicit, pinned TensorLy experiment")
class TensorLyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = TensorLySVD(SOURCE)
        evidence = os.environ.get("WP10_TENSORLY_FIXTURE_EVIDENCE")
        cls.evidence = Path(evidence) if evidence else None
        if cls.evidence is not None:
            cls.evidence.mkdir()

    def test_missing_and_wrong_provider_fail_closed(self):
        with patch("compression.low_rank_tensorly.util.find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "installation required"):
                TensorLySVD(SOURCE)
        with patch("compression.low_rank_tensorly.subprocess.check_output",
                   side_effect=[str(Path(SOURCE).resolve()).encode(), b"0"*40]):
            with self.assertRaisesRegex(ValueError, "source revision"):
                TensorLySVD(SOURCE)
        package = Path(self.oracle.tensorly.__file__).parent
        original = Path.read_bytes

        def corrupt(path):
            return original(path) + b"\n" if path == package / "tenalg/svd.py" else original(path)

        with patch.object(Path, "read_bytes", corrupt):
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                TensorLySVD(SOURCE)
        with patch.object(self.oracle.tensorly, "__version__", "0.0.0"):
            with self.assertRaisesRegex(ValueError, "origin or version"):
                TensorLySVD(SOURCE)

    def test_ranks_and_degenerate_spectra(self):
        rng = np.random.default_rng(64)
        dense = rng.normal(size=(128, 64))
        degenerate = np.zeros_like(dense)
        np.fill_diagonal(degenerate, np.r_[np.full(12, 8.), np.full(16, 4.), np.geomspace(1., 1e-12, 36)])
        layer = nn.Conv2d(64, 128, 1, bias=False).double()
        results = {}
        for name, weight in (("dense", dense), ("ties", degenerate), ("zero", np.zeros_like(dense))):
            with torch.no_grad():
                layer.weight.copy_(torch.from_numpy(weight)[:, :, None, None])
            ref_u, ref_s, ref_vh = np.linalg.svd(weight, full_matrices=False)
            gram_s = np.linalg.eigvalsh(weight.T @ weight)[::-1].clip(min=0)**0.5
            np.testing.assert_allclose(ref_s, gram_s, atol=1e-10, rtol=1e-11)
            for rank in RANKS:
                with self.subTest(name=name, rank=rank):
                    u, s, vh = (value.numpy() for value in self.oracle.decompose(layer, rank))
                    actual = (u * s) @ vh
                    error = np.linalg.norm(weight - actual)
                    tail = np.linalg.norm(ref_s[rank:])
                    np.testing.assert_allclose(s, ref_s[:rank], atol=1e-12, rtol=1e-12)
                    np.testing.assert_allclose(u.T @ u, np.eye(rank), atol=1e-12)
                    np.testing.assert_allclose(vh @ vh.T, np.eye(rank), atol=1e-12)
                    np.testing.assert_allclose(error, tail, atol=1e-11, rtol=1e-12)
                    if rank == 64 or not np.isclose(ref_s[rank-1], ref_s[rank], rtol=1e-12, atol=1e-13):
                        np.testing.assert_allclose(actual, (ref_u[:, :rank] * ref_s[:rank]) @ ref_vh[:rank],
                                                   atol=1e-11, rtol=1e-12)
                    native = LowRankMatrix(layer, rank, allow_expansion=rank == 64)
                    native_weight = (native.left @ native.right).detach().numpy()
                    np.testing.assert_allclose(np.linalg.norm(weight - native_weight), tail, atol=1e-11, rtol=1e-12)
                    results[f"{name}-{rank}"] = dict(residual=float(error), optimal_tail=float(tail))
        for rank in (0, 12, 65, True, 16.0):
            with self.assertRaisesRegex(ValueError, "registered WP10 rank"):
                self.oracle.decompose(layer, rank)
        with self.assertRaisesRegex(ValueError, "point convolution"):
            self.oracle.decompose(nn.Linear(64, 128), 16)
        if self.evidence is not None:
            write_json(self.evidence / "spectra.json", results)

    def test_fp64_forward_gradients_and_update(self):
        torch.manual_seed(64)
        device = os.environ.get("WP10_TENSORLY_MICRO_DEVICE", "cpu")
        results = {}
        if device == "cuda":
            was_strict = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(True)
            self.addCleanup(torch.use_deterministic_algorithms, was_strict)
        for biased in (False, True):
            layer = nn.Conv2d(64, 128, 1, bias=biased).double().to(device)
            for rank in RANKS:
                u, s, vh = self.oracle.decompose(layer, rank)
                native = LowRankMatrix(layer, rank, allow_expansion=rank == 64)
                np.testing.assert_allclose((u * s) @ vh, (native.left @ native.right).detach().cpu(), atol=1e-12)
                factor = copy.deepcopy(native)
                with torch.no_grad():
                    factor.left.copy_(u * s.sqrt())
                    factor.right.copy_(s.sqrt()[:, None] * vh)
                x = torch.randn(2, 64, 6, 4, dtype=torch.float64, device=device).transpose(2, 3).requires_grad_()
                target = torch.randn(2, 128, 4, 6, dtype=x.dtype, device=device)
                params = dict(factor.named_parameters())
                refs = {name: value.detach().clone().requires_grad_() for name, value in params.items()}
                xref = x.detach().clone().requires_grad_()
                dense_weight = refs["left"] @ refs["right"]
                expected = torch.einsum("oi,bihw->bohw", dense_weight, xref)
                if biased:
                    expected = expected + refs["bias"][None, :, None, None]
                with patch.object(F, "conv2d", wraps=F.conv2d) as conv:
                    actual = factor(x)
                self.assertEqual(conv.call_count, 2)
                first, second = conv.call_args_list
                self.assertEqual(tuple(first.args[1].shape), (rank, 64, 1, 1))
                self.assertEqual(tuple(second.args[1].shape), (128, rank, 1, 1))
                self.assertEqual(len(first.args), 2)
                self.assertIs(second.args[2], factor.bias)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
                (actual - target).square().mean().backward()
                (expected - target).square().mean().backward()
                torch.testing.assert_close(x.grad, xref.grad, atol=1e-12, rtol=1e-11)
                differences = {"input": float((x.grad - xref.grad).abs().max())}
                for name, value in params.items():
                    torch.testing.assert_close(value.grad, refs[name].grad, atol=1e-12, rtol=1e-11)
                    differences[name] = float((value.grad - refs[name].grad).abs().max())
                before = {name: value.detach().clone() for name, value in params.items()}
                torch.optim.Adam(params.values(), lr=1e-4).step()
                torch.optim.Adam(refs.values(), lr=1e-4).step()
                for name, value in params.items():
                    torch.testing.assert_close(value, refs[name], atol=1e-12, rtol=1e-11)
                    self.assertGreater(int((value != before[name]).count_nonzero()), 0)
                results[f"{biased}-{rank}"] = differences
        if self.evidence is not None:
            write_json(self.evidence / "gradients.json", dict(device=device, differences=results))

    def test_native_checkpoint_and_full_recovery_resume(self):
        results = {}
        with TemporaryDirectory() as directory:
            for classes in (9, 22):
                for method in ("native", "tensorly"):
                    with self.subTest(classes=classes, method=method):
                        root = Path(directory) / f"{classes}-{method}"
                        root.mkdir()
                        spec = admission.read(fixture(root / "inputs", classes))
                        del spec["prefix"]
                        request = dev._json_new(root / "request.json", spec)
                        reg, dense, scene, record, plans = _design(request)
                        assignment = reg["ladder"]["arms"]["svd-r16"]
                        seed = 64
                        source = (self.oracle.initialize(dense, 16, seed=seed) if method == "tensorly" else
                                  factorize_model(dense, {TARGET: 16}, seed=seed))
                        evidence = initial_evidence(dense, source, assignment)
                        checkpoint = root / "initial.pt"
                        digest = save_low_rank(source, checkpoint)
                        with patch("compression.low_rank_tensorly.import_module", side_effect=AssertionError):
                            loaded = load_low_rank(checkpoint, digest)
                        self.assertEqual(state_hash(loaded), state_hash(source))
                        for name, value in source.get_submodule(TARGET).named_parameters():
                            self.assertTrue(torch.equal(value, loaded.get_submodule(TARGET).get_parameter(name)))
                        plan, trial = plans["64"], f"{method}-r16"
                        # Observe the same uncached native execution policy after package reload.
                        loaded.backbone.adapter.cache_attention = False
                        full = new_run(loaded, scene, record, plan, trial)
                        full.advance(4)
                        full_state = state_hash(full.model)
                        full_packet = root / "full.pt"
                        full.save(full_packet)
                        prefix = new_run(loaded, scene, record, plan, trial)
                        prefix.advance(2)
                        path = root / "step-2.pt"
                        prefix.save(path)
                        args = dict(plan=plan, trial=trial, selector_seed=SEED,
                                    expected_contract_hash=prefix.contract_hash)
                        with self.assertRaisesRegex(ValueError, "hash mismatch"):
                            LowRankRecoveryRun.load(loaded, scene, record, path,
                                                    expected_sha256="0"*64, **args)
                        resumed = LowRankRecoveryRun.load(loaded, scene, record, path,
                                                          expected_sha256=dev._pin(path)["sha256"], **args)
                        resumed.advance(4)
                        self.assertEqual(state_hash(resumed.model), full_state)
                        resumed_packet = root / "resumed.pt"
                        resumed.save(resumed_packet)
                        baseline, actual = [torch.load(item, weights_only=True) for item in (full_packet, resumed_packet)]
                        for name in SCIENCE:
                            if name == "history":
                                for left, right in zip(actual[name], baseline[name], strict=True):
                                    changed = [key for key in left if not _exact(left[key], right[key])]
                                    self.assertEqual(changed, [], f"step {left['step']}")
                            else:
                                self.assertTrue(_exact(actual[name], baseline[name]), name)
                        self.assertEqual(resumed.history, full.history)
                        self.assertEqual(set(resumed.history[-1]["deployment"]["heads"]), set(HEADS))
                        activity = resumed.history[-1]["target_activity"]
                        self.assertTrue(all(value["changed"] and value["gradient_nonzero"] for value in activity.values()))
                        results[f"{classes}-{method}"] = dict(state=full_state, initial=evidence["state"],
                            heads=list(HEADS), target_activity=activity, actual_updates=8, resume_mode="in-process")
        if self.evidence is not None:
            write_json(self.evidence / "resume.json", results)
