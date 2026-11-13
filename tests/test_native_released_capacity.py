"""Same-capacity condition isolation, gradients and deterministic recovery."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch

from research_runtime import code_snapshot
from transfer import native_released_capacity as control, native_released_training as sparse
from transfer.native_released_points import MODE
from transfer.native_scale_fixture import Fixture
from transfer.native_scale_observation import forward
from transfer.native_scale_supervision import loss
from transfer.native_scale_training import baseline
from transfer.xuzhou_multiscale_audit import _exact
from joint_training import _rng_state, _restore_rng


class CapacityTests(TestCase):
    def test_joint_and_constant_share_parameters_and_do_not_leak_conditions(self):
        torch.set_num_threads(1)
        for classes in (9, 22):
            base = baseline(classes, 64)
            a, b = (control.model(base, mode, 7397) for mode in ("joint", control.CONSTANT))
            self.assertTrue(_exact(a.state_dict(), b.state_dict()))
            self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
            self.assertEqual(sum(p.numel() for p in a.parameters())-sum(p.numel() for p in base.parameters()), 2336)
            fixture = Fixture(classes, "devtrain")
            first, second = [sparse.materialize(fixture, {"tile": tile, "factor": 4, "band": "full",
                         "profile": "a"}, "control") for tile in fixture.tiles[:2]]
            actual, _, target = first
            model = a.eval()
            values = {key: control.condition(actual, key) for key in (*control.CONDITIONS, control.CONSTANT)}
            results = {key: forward(model, actual, "control", "devtrain", condition=value) for key, value in values.items()}
            self.assertEqual(len({(result[1]["q"], result[1]["a0"]) for result in results.values()}), 1)
            self.assertEqual(values["permuted"].source_key, (4, "full", "b"))
            self.assertEqual(values["missing"].values(), (None, None, None))
            self.assertNotEqual(results["matched"][1]["weights"], results["permuted"][1]["weights"])
            for kind in ("matched", "permuted", control.CONSTANT):
                with self.assertRaisesRegex(ValueError, "condition|observation"):
                    forward(model, second[0], "control", "devtrain", condition=values[kind])
            restaged = forward(model, actual, "control", "devtrain", condition=values["matched"])
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(restaged[0][:3], results["matched"][0][:3])))

    def test_actual_condition_parameters_update_and_resume_exactly(self):
        torch.set_num_threads(1)
        fixture = Fixture(9, "devtrain")
        events = [{"tile": tile, "factor": 2, "band": "full", "profile": "a"} for tile in fixture.tiles[:3]]
        base = baseline(9, 512)
        mode = control.CONSTANT

        def model():
            m = control.model(base, mode, 7833).train()
            return m, torch.optim.Adam(m.parameters(), lr=1e-4)

        def update(m, o, event):
            actual, _, target = sparse.materialize(fixture, event, "control")
            o.zero_grad(set_to_none=True)
            output, _ = forward(m, actual, "control", "devtrain", condition=control.condition(actual, mode))
            terms = loss(output, *target.weights(MODE))
            terms["total"].backward()
            active = [p for n, p in m.named_parameters() if ".condition_encoder." in n or ".range_encoder." in n]
            self.assertTrue(active and all(p.grad is not None for p in active))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in active), 0)
            o.step()
            return float(terms["total"].detach())

        net, optim = model()
        initial = deepcopy(net.state_dict())
        full_loss = update(net, optim, events[0])
        packet = (deepcopy(net.state_dict()), deepcopy(optim.state_dict()), _rng_state(torch.device("cpu")))
        for event in events[1:]:
            update(net, optim, event)
        other, other_optim = model()
        other.load_state_dict(packet[0])
        other_optim.load_state_dict(packet[1])
        _restore_rng(packet[2], torch.device("cpu"))
        for event in events[1:]:
            update(other, other_optim, event)
        self.assertTrue(_exact(net.state_dict(), other.state_dict()))
        self.assertTrue(_exact(optim.state_dict(), other_optim.state_dict()))
        changed = [name for name, value in net.state_dict().items() if not torch.equal(value, initial[name])]
        self.assertTrue(any(".condition_encoder." in name for name in changed))
        self.assertGreater(full_loss, 0.)
