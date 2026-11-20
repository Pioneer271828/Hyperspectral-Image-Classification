"""Check first-result audit rejection independent of the deployment runtime."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.audit_wp10_first_result import ordered_samples, verify_timing


class CostAuditTests(unittest.TestCase):
    def test_sample_sequence_rejects_swapped_rounds_and_repeated_index(self):
        with TemporaryDirectory() as tmp:
            work = Path(tmp)
            schedule = [{"case": f"package{i % 18}", "repeat": i // 18} for i in range(54)]
            for index, row in enumerate(schedule):
                (work / f"{index:02d}-{row['case']}.sample.json").write_text(json.dumps(dict(row, index=index)))
            self.assertEqual(len(list(ordered_samples(work, schedule))), 54)
            final = work / "53-package17.sample.json"
            row = json.loads(final.read_text())
            for change in ({"index": 17}, {"repeat": 1}, {"case": "package16"}):
                with self.subTest(change=change):
                    final.write_text(json.dumps({**row, **change}))
                    with self.assertRaisesRegex(ValueError, "frozen schedule"):
                        list(ordered_samples(work, schedule))
            final.rename(work / "52-package17.sample.json")
            with self.assertRaisesRegex(ValueError, "54 scheduled sample files"):
                list(ordered_samples(work, schedule))

    def test_stage_durations_derive_from_raw_clocks(self):
        clock = dict(python=1_000_000, initialized=3_000_000, inputs=4_000_000, model=7_000_000,
                     forward=8_000_000, verified=10_000_000)
        stages = dict(launch_to_python=1., initialize=2., authenticate_and_prepare_input=1.,
                      load_authenticated_package=3., first_full_graph_and_forward=1.,
                      validate_and_hash_outputs=2., serialize_deliver_and_verify=1., exit_after_result=2.)
        data = dict(start_ns=0, accepted_ns=11_000_000, exit_ns=13_000_000,
                    first_result_ms=11., subprocess_exit_ms=13., stages_ms=stages.copy())
        self.assertEqual(verify_timing(data, clock), 11.)
        data["stages_ms"].update(initialize=1.5, exit_after_result=2.5)
        with self.assertRaisesRegex(ValueError, "stage duration"):
            verify_timing(data, clock)
