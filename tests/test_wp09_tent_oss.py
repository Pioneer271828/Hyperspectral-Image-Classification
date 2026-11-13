"""Fixed Tent upstream behavior and the optional WP09 session contract."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from evaluate_tta import _Session
from evaluation.wp09_upstream import tent_reference
from scripts.review_wp09_tent import compare, run_child
from test_wp09_online_tta import fixture, registration
from transfer.online_tta_run import execute
from transfer.wp09_tent import execute_tent


SOURCE = os.environ.get("WP09_TENT_SOURCE")


@unittest.skipUnless(SOURCE, "Pinned Tent source required")
class TentTests(unittest.TestCase):
    def test_upstream_entropy_collect_configure_reset_and_steps(self):
        upstream, _ = tent_reference(SOURCE)
        x = torch.tensor([[[[2., 4.],[3., 5.]],[[0., 1.],[2., 1.]]]])
        class Small(nn.Module):
            def __init__(self):
                super().__init__()
                self.active = nn.BatchNorm2d(2)
                self.inactive = nn.BatchNorm2d(2)
                self.downstream = nn.BatchNorm1d(2)
                self.head = nn.Linear(2, 3)
            def forward(self, x):
                return self.head(self.active(x).permute(0,2,3,1).reshape(-1,2))
        torch.manual_seed(64)
        model = upstream.configure_model(Small())
        parameters, names = upstream.collect_params(model)
        self.assertEqual(names, ["active.weight","active.bias","inactive.weight","inactive.bias"])
        self.assertIsNone(model.active.running_mean)
        self.assertFalse(model.active.track_running_stats)
        self.assertFalse(model.head.weight.requires_grad)
        rng = torch.randn(11, 9, requires_grad=True)
        p = rng.softmax(1)
        prior = -(p*p.clamp_min(1e-8).log()).sum(1).mean()
        current = upstream.softmax_entropy(rng).mean()
        torch.testing.assert_close(prior, current, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(torch.autograd.grad(prior,rng)[0], torch.autograd.grad(current,rng)[0],
                                   rtol=1e-5, atol=1e-6)
        solver = torch.optim.SGD(parameters, lr=.05)
        tent = upstream.Tent(model, solver, steps=1, episodic=True)
        start = model(x).detach().clone()
        self.assertTrue(torch.equal(tent(x).detach(), start))
        self.assertFalse(torch.equal(model(x).detach(), start))
        tent.steps = 2
        self.assertFalse(torch.equal(tent(x).detach(), start))
        tent.reset()
        self.assertTrue(torch.equal(model(x).detach(), start))
        case = upstream.Tent(model, solver, steps=1, episodic=False)
        first, second = case(x), case(x)
        self.assertFalse(torch.equal(first.detach(), second.detach()))

    def test_prediction_is_durable_before_update_and_failure_is_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _, _, spec=fixture(root/"inputs",9)
            reg=registration(root/"inputs",spec,"affine")
            out=root/"failed"
            from transfer.wp09_tent import TentEntropySession
            def check(session, inputs):
                pending=out/".pending-000001"
                self.assertTrue((pending/"prediction.npz").exists())
                self.assertFalse((pending/"checkpoint.pt").exists())
                raise RuntimeError("injected before update")
            with patch.object(TentEntropySession,"adapt",check), self.assertRaisesRegex(RuntimeError,"before update"):
                execute_tent(reg,out,SOURCE)
            with self.assertRaisesRegex(RuntimeError,"attempt retained"):
                execute_tent(reg,out,SOURCE)
            _, _, stats=fixture(root/"group",22,"group_norm")
            with self.assertRaisesRegex(ValueError,"no supported mutable state"):
                execute_tent(registration(root/"group",stats,"statistics"),root/"stats",SOURCE)

    def test_nine_twenty_two_modes_restart_and_gradients(self):
        evidence = os.environ.get("WP09_TENT_FIXTURE_DIR")
        folder = tempfile.TemporaryDirectory() if not evidence else None
        if folder:
            self.addCleanup(folder.cleanup)
        root = Path(evidence or folder.name)
        root.mkdir(exist_ok=True)
        from model.research_config import digest
        upstream, identity = tent_reference(SOURCE)
        provider = digest(dict(identity, component="softmax_entropy", native_clamped_floor=1e-8))
        for classes in (9, 22):
            _, _, spec = fixture(root / str(classes), classes)
            local = root / str(classes)
            for mode in ("frozen","statistics","affine","adapter"):
                with self.subTest(classes=classes, mode=mode):
                    reg = registration(local, spec, mode)
                    native, optional = local/f"{mode}-native", local/f"{mode}-tent"
                    execute(reg, native)
                    split = local/f"{mode}-split"
                    execute_tent(reg, optional, SOURCE)
                    if mode in ("affine", "adapter"):
                        for stop in (1,2,6):
                            run_child(reg,split,stop,SOURCE,local/f"{mode}-{stop}.log")
                    else:
                        for stop in (1,2,6):
                            execute_tent(reg,split,SOURCE,to_frame=stop)
                    with patch.object(_Session,"forward",side_effect=AssertionError("completed forward")):
                        self.assertEqual(execute_tent(reg,split,SOURCE)["added"],0)
                    from transfer.online_tta_run import committed_history
                    from transfer.online_tta_stream import StreamRegistration
                    view = StreamRegistration(reg)
                    view.adaptation_provider = provider
                    self.assertEqual(committed_history(view, optional, complete=True),
                                     committed_history(view, split, complete=True))
                    compare(mode,native,optional,reg,reg,provider)
                    with self.assertRaisesRegex(ValueError, "different registration"):
                        execute(reg, split)
                    with self.assertRaisesRegex(ValueError, "different registration"):
                        execute_tent(reg,native,SOURCE)
