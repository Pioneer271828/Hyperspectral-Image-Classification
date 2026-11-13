"""Capacity collision, unreachable and shuffled-message oracles."""
from unittest import TestCase

import numpy as np
import torch

from evaluation import learned_graph_channel as task, learned_graph_channel_review as review


class ChannelTests(TestCase):
    def test_same_support_capacity_has_an_analytic_collision(self):
        torch.set_num_threads(1)
        batch = task.instances(64, "devval", "capacity", 0, 48)
        for arm in ("parallel", "bottleneck", "shuffle"):
            pairs = task.edges("capacity", arm)
            self.assertEqual(len(pairs), task.NODES+4)
            self.assertEqual([review.directed_distance(pairs, i) for i in (0, 1)], [2, 2])
        base = task.model_features(batch)
        swapped = base.clone()
        swapped[:, 0, 0], swapped[:, 1, 0] = base[:, 1, 0], base[:, 0, 0]
        model = task.ChannelModel(64).eval()
        for arm in ("parallel", "bottleneck"):
            matrix = task.arm_matrix(arm, "capacity")
            with torch.no_grad():
                left = model(base, matrix, batch, "capacity", arm)
                right = model(swapped, matrix, batch, "capacity", arm)
            self.assertEqual(torch.equal(left, right), arm == "bottleneck")
        self.assertTrue(np.all(batch["donors"] != np.arange(48)))

    def test_held_roles_separate_and_directed_negative(self):
        for key in task.STRATA:
            ids = {task.instances(64, role, key, 0, 48)["fingerprint"] for role in task.ROLE_SEED}
            self.assertEqual(len(ids), 3)
        self.assertEqual(review.directed_distance(task.edges("beyond", "parallel")), 4)
        self.assertIsNone(review.directed_distance(task.edges("disconnected", "parallel")))
        self.assertIsNone(review.directed_distance(task.edges("reachable", "local")))

    def test_wrong_checkpoint_role_seed_or_optimizer_cursor_rejected(self):
        torch.set_num_threads(1)
        origin = task.Run(64, "parallel")
        origin.advance(2)
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            task.Run(512, "parallel").restore(origin.pack("registration"), "registration", 2)
        packet = origin.pack("registration")
        first = next(iter(packet["optimizer"]["state"].values()))
        first["step"] = torch.tensor(3.)
        with self.assertRaisesRegex(ValueError, "Adam"):
            task.Run(64, "parallel").restore(packet, "registration", 2)
