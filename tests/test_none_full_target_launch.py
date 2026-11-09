"""Frozen cohort/label barriers before target updates, without query consumers."""
import copy
from pathlib import Path
import tempfile
import unittest

from joint_baseline import NAMES
from loadData.dataset_registry import file_hash
from none_full_target_launch import PROFILE, load_case_data, select_case
from none_full_target_runtime import settings
from research_runtime import code_snapshot
from scripts.train_joint_m4 import graph_binding
from target_initialization import METHODS
from test_none_full_transfer import fixture as source_fixture
from test_target_training import fixture as target_fixture


class NoneFullTargetLaunchTests(unittest.TestCase):
    def test_all_six_roles_budget_hash_and_initial_graph_barriers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, spec = source_fixture(root / "source")
            # Synthetic catalogue: this gate validates coverage and pins; the
            # real initializer independently validates target/seed/source roles.
            cases = [{"name": n + "-" + m, "target": n, "method": m,
                "out": str(root / "run" / "cases" / (n + "-" + m)),
                "settings": settings(n, m, device="cuda"), "initial_state_sha256": "a" * 64,
                "graph_binding": {n: {"Q": "b" * 64, "A": "c" * 64}},
                "source_spec": {} if m == METHODS[0] else spec} for n in NAMES for m in METHODS[:2]]
            code = code_snapshot()
            launch = {"protocol": PROFILE, "code": code, "out_dir": str(root / "run"), "cases": cases,
                      "optimizer_updates": 72, "query_labels_accessed": False, "test_scoring": False}
            self.assertEqual(select_case(launch, code, "xuzhou", METHODS[1])["source_spec"], spec)
            for mutate in (lambda x: x["cases"].pop(),
                           lambda x: x["cases"].__setitem__(0, x["cases"][1]),
                           lambda x: x["cases"][0].__setitem__("initial_state_sha256", "bad"),
                           lambda x: x["cases"][0]["settings"].__setitem__("optimizer_steps", 1500),
                           lambda x: x["cases"][0].__setitem__("graph_binding", {}),
                           lambda x: x.__setitem__("query_labels_accessed", True),
                           lambda x: x["cases"][1]["source_spec"].__setitem__("source_weight_sha256", "0" * 64)):
                bad = copy.deepcopy(launch)
                mutate(bad)
                with self.assertRaises(ValueError): select_case(bad, code, "xuzhou", METHODS[1])

    def test_actual_sanitized_target_context_and_graph_must_match_freeze(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (data, provenance, _), paths = target_fixture(root)
            # The accepted fixture returns observation, record and label family.
            observation, record, family = paths
            case = {"target": "xuzhou", "inputs": {"observation": str(observation), "record": str(record),
                "record_sha256": file_hash(record), "family": str(family), "family_sha256": file_hash(family)},
                "target_context": provenance, "graph_binding": graph_binding(data)}
            loaded, _, guard = load_case_data(case)
            self.assertEqual(graph_binding(loaded), case["graph_binding"])
            guard()
            bad = copy.deepcopy(case)
            bad["target_context"]["support_seed"] = 512
            with self.assertRaises(ValueError): load_case_data(bad)
            bad = copy.deepcopy(case)
            bad["graph_binding"]["xuzhou"]["Q"] = "0" * 64
            with self.assertRaises(ValueError): load_case_data(bad)


if __name__ == "__main__":
    unittest.main()
