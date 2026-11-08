import unittest

from monitoring.historical_screen import screen_trajectory


class HistoricalScreenTests(unittest.TestCase):
    def test_small_class_and_other_scene_can_each_veto(self):
        samples = []
        for step in (1, 2):
            samples.extend([
                {"scene": "a", "step": step, "rms": [1, 2, 3, 4, 5],
                 "protected_rms": {"3": [1, 5, 2, 3, 4]}},
                {"scene": "b", "step": step, "rms": [1, 2, 5, 3, 4],
                 "protected_rms": {"6": [1, 2, 3, 4, 5]}},
            ])
        kwargs = {"steps": [1, 2], "protected_classes": {"a": [3], "b": [6]},
                  "channels": 5, "bottom": 3, "cap": 1}
        receipt = screen_trajectory(samples, **kwargs)
        self.assertEqual(receipt["scene_eligible"], [0, 1])
        self.assertEqual(receipt["shortlist"], [0])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            screen_trajectory(samples[:-1], **kwargs)
        samples[0]["protected_rms"] = {}
        with self.assertRaisesRegex(ValueError, "small-class"):
            screen_trajectory(samples, **kwargs)

    def test_empty_intersection_does_not_backfill(self):
        receipt = screen_trajectory(
            [{"scene": "a", "step": 1, "rms": [1, 2], "protected_rms": {"3": [2, 1]}},
             {"scene": "a", "step": 2, "rms": [1, 2], "protected_rms": {"3": [2, 1]}}],
            steps=[1, 2], protected_classes={"a": [3]}, channels=2, bottom=1, cap=1)
        self.assertEqual(receipt["scene_eligible"], [0])
        self.assertEqual(receipt["shortlist"], [])
