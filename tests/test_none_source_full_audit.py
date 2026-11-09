"""Independent full verifier rejects partial budgets or changed producer files."""
import copy
import unittest
from model.research_config import digest
from research.reproduction.m4g4b.none_source_full_audit import acceptance_contract


class NoneSourceFullAuditTests(unittest.TestCase):
    def test_all_nine_full_acceptance_and_additive_only_verifier(self):
        producer = {"commit": "fixture", "dirty": False, "files": {"training.py": "a"},
                    "source_hash": digest({"training.py": "a"})}
        consumer = {"dirty": False, "files": {"training.py": "a", "audit.py": "b"},
                    "source_hash": digest({"training.py": "a", "audit.py": "b"})}
        launch = {"code": producer}
        rows = [{"name": str(i), "updates": 1002, "exact_reload_repeats": 3, "completed_resume_updates": 0,
                 "metrics": 12, "recalls": 186 if i < 6 else 108} for i in range(9)]
        acceptance = {"status": "passed", "code": producer, "optimizer_updates": 9018, "launch_sha256": "pin",
                      "query_labels_accessed": False, "test_scoring": False, "cases": rows}
        self.assertEqual(len(acceptance_contract(launch, acceptance, consumer, "fixture", "pin")), 9)
        for mutation in ("case", "budget", "query", "reload", "readonly", "metrics"):
            bad = copy.deepcopy(acceptance)
            if mutation == "case": bad["cases"].pop()
            if mutation == "budget": bad["optimizer_updates"] = 24
            if mutation == "query": bad["query_labels_accessed"] = True
            if mutation == "reload": bad["cases"][0]["exact_reload_repeats"] = 2
            if mutation == "readonly": bad["cases"][0]["completed_resume_updates"] = 1
            if mutation == "metrics": bad["cases"][0]["metrics"] = 11
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                acceptance_contract(launch, bad, consumer, "fixture", "pin")
        bad = copy.deepcopy(consumer)
        bad["files"]["training.py"] = "changed"
        bad["source_hash"] = digest(bad["files"])
        with self.assertRaises(ValueError): acceptance_contract(launch, acceptance, bad, "fixture", "pin")
        with self.assertRaises(ValueError): acceptance_contract(launch, acceptance, consumer, "other", "pin")


if __name__ == "__main__":
    unittest.main()
