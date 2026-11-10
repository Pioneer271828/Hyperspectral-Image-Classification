import copy
import unittest

from evaluation.dynamic_curve import timing_summary as established_summary
from evaluation.wp12_cost_primitives import timing_summary
from model.research_config import digest
from scripts.measure_wp12_historical_cost import PRODUCER, preserved_runtime


class HistoricalCostTests(unittest.TestCase):
    def test_standalone_timing_matches_established_reducer(self):
        for values in ([1, 2, 3, 4], [.01, .07, .3, .6, 12, 2], [3] * 12):
            self.assertEqual(timing_summary(values, repeats=len(values), blocks=123),
                             established_summary(values, repeats=len(values), blocks=123))
        for values in ([1, 2, float("nan"), 3], [0, 2, 3, 4], [True, 2, 3, 4]):
            with self.assertRaises(ValueError):
                timing_summary(values, repeats=4, blocks=123)

    def test_preserved_sources_admit_additions_and_reject_changes(self):
        original = dict(commit=PRODUCER, dirty=False, files={"model.py": "a" * 64},
                        source_hash=digest({"model.py": "a" * 64}))
        freeze = dict(producer=original, status="all54_frozen_before_query",
                      policy="g4_matched_ordered_all54_frozen_query_v2")
        current = copy.deepcopy(original)
        current["files"]["collector.py"] = "b" * 64
        current["source_hash"] = digest(current["files"])
        self.assertEqual(preserved_runtime(freeze, current)["preserved_original_files"], 1)
        for defect in ("change", "missing", "dirty", "wrong_source", "wrong_digest"):
            f, c = copy.deepcopy((freeze, current))
            if defect == "change":
                c["files"]["model.py"] = "c" * 64
            elif defect == "missing":
                del c["files"]["model.py"]
            elif defect == "dirty":
                c["dirty"] = True
            elif defect == "wrong_source":
                f["producer"]["commit"] = "1" * 40
            else:
                f["producer"]["source_hash"] = "0" * 64
            c["source_hash"] = digest(c["files"])
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                preserved_runtime(f, c)
