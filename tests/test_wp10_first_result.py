"""Check first-result failure boundaries without loading project models."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

from scripts.measure_wp10_first_result import first_result, identity_matches, pin, read, timing


class FirstResultTests(unittest.TestCase):
    def launch(self, script, check=lambda row, pid: None, timeout=3):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            return first_result([sys.executable, "-c", script], directory, directory / "sample", check, timeout)

    def test_acknowledgement_precedes_exit(self):
        row, started, accepted, exited = self.launch(
            'import time; print("{\\"result\\": 1}", flush=True); time.sleep(0.1)')
        self.assertEqual(row, {"result": 1})
        self.assertLess(started, accepted)
        self.assertLess(accepted, exited)
        self.assertGreater(exited - accepted, 50_000_000)
        row["clocks_ns"] = dict.fromkeys(("python", "initialized", "inputs", "model", "forward", "verified"), started)
        result = timing(row, started, accepted, exited)
        self.assertAlmostEqual(result["first_result_ms"],
                               sum(result["stages_ms"].values()) - result["stages_ms"]["exit_after_result"])
        with self.assertRaisesRegex(ValueError, "clock ordering"):
            timing(row, started, started - 1, exited)

    def test_incomplete_failure_and_timeout(self):
        with self.assertRaisesRegex(RuntimeError, "complete first result"):
            self.launch("import sys; sys.exit(3)")
        with self.assertRaisesRegex(RuntimeError, "did not exit cleanly"):
            self.launch('print("{}", flush=True); raise SystemExit(3)')
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            self.launch('import time; print("{", end="", flush=True); time.sleep(30)', timeout=0.1)

    def test_missing_input_and_wrong_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.json"
            path.write_text(json.dumps({"input": "frozen"}))
            ref = pin(path)
            self.assertEqual(read(ref), {"input": "frozen"})
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "digest changed"):
                read(ref)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                read(ref)

    def test_false_result_and_wrong_package_rejected(self):
        known = dict(case="dense", manifest={}, tiles=[], output_hashes=["right"], state="original",
                     package_bytes=80, parameters=12, factor_calls={})
        row = dict(**known, pid=8, cpu=0, threads=1, query=0, gt=0, optimizer_updates=0, calibration=0)
        identity_matches(row, known, 8, 0)
        for changed in (dict(output_hashes=["wrong"]), dict(state="wrong"), dict(query=1)):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "First result differs"):
                identity_matches({**row, **changed}, known, 8, 0)


if __name__ == "__main__":
    unittest.main()
