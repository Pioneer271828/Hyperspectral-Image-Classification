"""Runtime correctness on synthetic inputs, not real HSI accuracy evidence."""
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from lpft_probe import LPFTPlan, start_phase
from lpft_probe.runtime import LPFTRun
from test_lpft_probe import TinyThreeHead


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(250)
        self.model = TinyThreeHead().double()
        self.plan = LPFTPlan(8, 3, .03, .003)
        self.identity = {"source": "a" * 64, "support": "b" * 64}
        self.x = torch.randn(18, 4, dtype=torch.float64)
        self.y = torch.arange(18) % 3

    def loss(self, model):
        # Random augmentation exercises restoration of the global Torch RNG.
        x = self.x + .02 * torch.randn_like(self.x)
        return sum(nn.functional.cross_entropy(o, self.y) for o in model(x)) / 3

    def assert_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_equal(x, y)
        else:
            self.assertEqual(a, b)

    def new_run(self):
        return LPFTRun(self.model, self.plan, self.identity)

    def test_every_update_checkpoint_resumes_exact_trajectory(self):
        run = self.new_run()
        snapshots = [run.snapshot()]
        losses = []
        for _ in range(8):
            losses.append(run.step(self.loss))
            snapshots.append(run.snapshot())
        final = run.snapshot()
        for cursor, snapshot in enumerate(snapshots):
            with self.subTest(cursor=cursor):
                stream = io.BytesIO()
                torch.save(snapshot, stream)
                stream.seek(0)
                loaded = torch.load(stream, weights_only=True)
                resumed = LPFTRun.restore(self.model, self.plan, self.identity, loaded)
                actual = [resumed.step(self.loss) for _ in range(cursor, 8)]
                self.assertEqual(losses[cursor:], actual)
                self.assert_equal(final, resumed.snapshot())

    def test_runtime_matches_original_boundary_policy(self):
        runtime = self.new_run()
        reference = copy.deepcopy(self.model)
        opt, _ = start_phase(reference, self.plan, 0)
        for cursor in range(8):
            if cursor == 3:
                opt, _ = start_phase(reference, self.plan, 3)
            rng = torch.get_rng_state().clone()
            actual_loss = runtime.step(self.loss)
            torch.set_rng_state(rng)
            opt.zero_grad(set_to_none=True)
            expected_loss = self.loss(reference)
            expected_loss.backward()
            opt.step()
            self.assertEqual(actual_loss, float(expected_loss.detach()))
            self.assert_equal(reference.state_dict(), runtime.model.state_dict())

    def test_exact_boundary_resets_optimizer_once(self):
        run = self.new_run()
        for _ in range(3):
            run.step(self.loss)
        self.assertEqual(run.completed_steps, 3)
        self.assertFalse(run.optimizer.state)
        resumed = LPFTRun.restore(self.model, self.plan, self.identity, run.snapshot())
        resumed.step(self.loss)
        self.assertTrue(all(v["step"].item() == 1 for v in resumed.optimizer.state.values()))

    def test_completion_rejects_updates_without_callback_or_mutation(self):
        run = self.new_run()
        for _ in range(8):
            run.step(self.loss)
        run = LPFTRun.restore(self.model, self.plan, self.identity, run.snapshot())
        before = run.snapshot()
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            run.step(lambda _: self.fail("Completed callback must not execute"))
        self.assert_equal(before, run.snapshot())

    def test_malformed_restore_never_changes_template_or_global_rng(self):
        run = self.new_run()
        for _ in range(5):
            run.step(self.loss)
        good = run.snapshot()
        corruptions = [
            lambda s: s.update(completed_steps=True),
            lambda s: s.update(completed_steps=9),
            lambda s: s["input_sha256"].update(source="c" * 64),
            lambda s: s["contract"].update(backbone_lr=.01),
            lambda s: s["binding"].update(sha256="0" * 64),
            lambda s: s["model"].pop("heads.bias"),
            lambda s: s["model"]["heads.weight"].fill_(float("nan")),
            lambda s: s["model"]["backbone.1.running_mean"].add_(1),
            lambda s: s["optimizer"]["param_groups"][0].update(lr=.9),
            lambda s: s["optimizer"]["state"][0]["step"].fill_(1),
            lambda s: s["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1),
            lambda s: s["optimizer"]["state"].pop(0),
            lambda s: s.update(torch_rng=torch.zeros(2, dtype=torch.uint8)),
        ]
        before = copy.deepcopy(self.model.state_dict())
        flags = [p.requires_grad for p in self.model.parameters()]
        for index, corrupt in enumerate(corruptions):
            with self.subTest(index=index):
                bad = copy.deepcopy(good)
                corrupt(bad)
                rng = torch.get_rng_state().clone()
                with self.assertRaises((ValueError, RuntimeError, TypeError)):
                    LPFTRun.restore(self.model, self.plan, self.identity, bad)
                self.assert_equal(before, self.model.state_dict())
                self.assert_equal(rng, torch.get_rng_state())
                self.assertTrue(self.model.training)
                self.assertEqual(flags, [p.requires_grad for p in self.model.parameters()])

    def test_snapshot_is_detached_and_constructor_preserves_template(self):
        before = copy.deepcopy(self.model.state_dict())
        run = self.new_run()
        run.step(self.loss)
        saved = run.snapshot()
        saved["model"]["heads.weight"].fill_(999)
        saved["input_sha256"]["source"] = "c" * 64
        self.assertFalse(torch.equal(saved["model"]["heads.weight"], run.model.heads.weight))
        self.assertEqual(run.input_sha256, self.identity)
        self.assert_equal(before, self.model.state_dict())

    def test_failure_poisoning_prevents_continuation_and_checkpoint(self):
        run = self.new_run()
        with self.assertRaisesRegex(ValueError, "scalar loss"):
            run.step(lambda model: model(self.x)[0].sum() * float("nan"))
        self.assertEqual(run.completed_steps, 0)
        with self.assertRaisesRegex(RuntimeError, "Failed update"):
            run.step(self.loss)
        with self.assertRaisesRegex(RuntimeError, "Failed update"):
            run.snapshot()

    def test_missing_head_gradients_and_train_mode_are_rejected(self):
        run = self.new_run()
        with self.assertRaisesRegex(ValueError, "gradient"):
            run.step(lambda model: model(self.x)[0].sum())
        other = self.new_run()
        other.model.train()
        with self.assertRaisesRegex(ValueError, "eval mode"):
            other.step(self.loss)
        self.assertEqual(other.completed_steps, 0)

    def test_input_identity_requires_explicit_hashes(self):
        for identity in ({}, {"source": "path"}, {"": "a" * 64}, {"source": 1}):
            with self.assertRaises(ValueError):
                LPFTRun(self.model, self.plan, identity)

    def _offload_comparison(self, template):
        native = LPFTRun(template, self.plan, self.identity)
        offload = LPFTRun(template, self.plan, self.identity, saved_tensors="cpu")
        midpoint = None
        tail = []
        for cursor in range(8):
            rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state() if template.heads.weight.is_cuda else None
            expected = native.step(self.loss)
            torch.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
            self.assertEqual(expected, offload.step(self.loss))
            self.assert_equal(native.model.state_dict(), offload.model.state_dict())
            self.assert_equal(native.optimizer.state_dict(), offload.optimizer.state_dict())
            if cursor == 3:
                midpoint = offload.snapshot()
            elif cursor > 3:
                tail.append(expected)
        final = offload.snapshot()
        resumed = LPFTRun.restore(template, self.plan, self.identity, midpoint,
                                  saved_tensors="cpu")
        self.assertEqual(tail, [resumed.step(self.loss) for _ in range(4)])
        self.assert_equal(final, resumed.snapshot())

    def test_cpu_offload_exact_updates_moments_and_resume(self):
        self._offload_comparison(self.model)

    def test_memory_policy_is_explicit_and_restore_cannot_switch_it(self):
        with self.assertRaises(ValueError):
            LPFTRun(self.model, self.plan, self.identity, saved_tensors="auto")
        run = LPFTRun(self.model, self.plan, self.identity, saved_tensors="cpu")
        run.step(self.loss)
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            LPFTRun.restore(self.model, self.plan, self.identity, run.snapshot())
        self.assert_equal(rng, torch.get_rng_state())

    @unittest.skipUnless(os.environ.get("LPFT_RUNTIME_DEVICE") == "cuda",
                         "Explicit isolated GPU allocation required")
    def test_cuda_offload_tiny_model_exact_updates_and_resume(self):
        self.assertTrue(torch.cuda.is_available())
        self.x, self.y = self.x.cuda(), self.y.cuda()
        self._offload_comparison(self.model.cuda())

    def test_live_optimizer_change_rejected_before_callback(self):
        run = self.new_run()
        run.optimizer.param_groups[0]["lr"] = .9
        before = copy.deepcopy(run.model.state_dict())
        with self.assertRaisesRegex(ValueError, "options changed"):
            run.step(lambda _: self.fail("Changed policy must not execute"))
        self.assertEqual(run.completed_steps, 0)
        self.assert_equal(before, run.model.state_dict())

    def test_existing_g4_network_interior_resume_and_input_guard(self):
        from model.research_config import ResearchConfig
        from target_initialization import initialize_target, METHODS
        from target_training import loss_terms, selected_outputs, target_settings
        from test_target_training import fixture
        with tempfile.TemporaryDirectory() as directory:
            (items, _, guard), _ = fixture(Path(directory))
            template, _ = initialize_target('xuzhou', METHODS[0], model_seed=64, head_seed=1064)
            config = ResearchConfig.from_dict(target_settings('xuzhou', METHODS[0])['config'])
            plan = LPFTPlan(6, 2, .001, .0001)

            def loss(model):
                outputs = selected_outputs(model, 'xuzhou', items['xuzhou'], 'train')
                return loss_terms(outputs, items['xuzhou']['train_labels'], config)['total']

            run = LPFTRun(template, plan, self.identity)
            for _ in range(3):
                run.step(loss)
            snapshot = run.snapshot()
            losses = [run.step(loss) for _ in range(3)]
            final = run.snapshot()
            resumed = LPFTRun.restore(template, plan, self.identity, snapshot)
            self.assertEqual(losses, [resumed.step(loss) for _ in range(3)])
            self.assert_equal(final, resumed.snapshot())
            guard()

    @unittest.skipUnless(os.environ.get("LPFT_RUNTIME_DEVICE") == "cuda",
                         "Explicit isolated GPU allocation required")
    def test_cuda_same_device_resume(self):
        self.assertTrue(torch.cuda.is_available())
        self.x, self.y = self.x.cuda(), self.y.cuda()
        template = self.model.cuda()
        run = LPFTRun(template, self.plan, self.identity)
        for _ in range(4):
            run.step(self.loss)
        midpoint = run.snapshot()
        losses = [run.step(self.loss) for _ in range(4)]
        final = run.snapshot()
        resumed = LPFTRun.restore(template, self.plan, self.identity, midpoint)
        self.assertEqual(losses, [resumed.step(self.loss) for _ in range(4)])
        self.assert_equal(final, resumed.snapshot())


if __name__ == "__main__":
    unittest.main()
