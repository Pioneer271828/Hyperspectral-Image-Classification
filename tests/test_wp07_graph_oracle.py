from copy import deepcopy
import math
from pathlib import Path
import subprocess
import sys
import unittest

from scripts.check_wp07_graph_geometry import validate_view


class ScalarGeometryOracleTests(unittest.TestCase):
    def test_line_graph_has_two_additional_distance_two_edges(self):
        near = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]
        wide = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
        def data(links, weights, mean, hops):
            return {"positive_edges": [list(x) for x in zip(*links)], "positive_weights": weights,
                    "distance_min": 0., "distance_max": 1. if len(links) == 7 else 2.,
                    "distance_mean": mean, "chebyshev_hop_counts": hops}
        view = {"grid": [3, 1], "identity": {"grid_shape": [3, 1]}, "factor": 1,
                "q": {"shape": [3, 3]},
                "near": data(near, [.5, .5, 1/3, 1/3, 1/3, .5, .5], 4/7, {"0": 3, "1": 4}),
                "wide": data(wide, [1/3]*9, 8/9, {"0": 3, "1": 4, "2": 2})}
        self.assertEqual(validate_view(view)["added_distance2_edges"], 2)
        wrong = deepcopy(view)
        wrong["wide"] = wrong["near"]
        with self.assertRaisesRegex(ValueError, "Radius 2 positive support"):
            validate_view(wrong)

    def test_equivalent_support_with_different_weights_is_not_a_range_ablation(self):
        links = [(a, b) for a in range(4) for b in range(4)]
        data = {"positive_edges": [list(x) for x in zip(*links)], "positive_weights": [.25]*16,
                "distance_min": 0., "distance_max": math.sqrt(2),
                "distance_mean": (8+4*math.sqrt(2))/16, "chebyshev_hop_counts": {"0": 4, "1": 12}}
        view = {"grid": [2, 2], "identity": {"grid_shape": [2, 2]}, "factor": 1,
                "q": {"shape": [4, 4]}, "near": data, "wide": deepcopy(data)}
        view["wide"]["positive_weights"][:4] = [.1, .2, .3, .4]
        result = validate_view(view)
        self.assertFalse(result["identifiable"])
        self.assertTrue(result["support_equivalent"])
        self.assertFalse(result["weighted_equivalent"])

    def test_optimized_oracle_refused(self):
        p = subprocess.run([sys.executable, "-O", "scripts/check_wp07_graph_geometry.py", "--help"],
                           cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("refuses optimized Python", p.stderr)
