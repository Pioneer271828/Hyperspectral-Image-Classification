"""Regression tests for lazy bank admission and the recoverable synthetic consumer."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from research_runtime import hash_file
from transfer import development_bank as bank, development_bank_training as training
from transfer.development_block_probe import VIEWS
from transfer.xuzhou_multiscale_audit import _exact


class DevelopmentBankTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.release, self.arrays = training.fixture(9)
        self.dest = self.root / "bank"
        self.reader = Mock(side_effect=lambda row: self.arrays[row["id"]])

    def create(self):
        return bank.publish(self.dest, self.root, self.release, self.reader)

    def open(self):
        return bank.Bank(self.create(), self.root, self.release)

    def test_role_and_batch_guards_precede_payload_decode(self):
        opened = self.open()
        self.assertEqual(opened.reads, 0)
        with patch.object(bank, "_decode", side_effect=AssertionError("unauthorized read")) as reader:
            for role, ids in (("devval", [3]), ("devtrain", [3, 3]), ("test", [3]), ("devtrain", [999])):
                with self.subTest(role=role, ids=ids), self.assertRaises(ValueError):
                    opened.batch(role, ids, VIEWS["full"])
            with self.assertRaises(ValueError):
                opened.materialize(dict(opened.rows[3], box=[0, 5, 0, 6]), VIEWS["full"])
        reader.assert_not_called()
        view = opened.batch("devtrain", [3, 10], VIEWS["even_retained_positions"])
        self.assertEqual(opened.reads, 0)
        self.assertEqual(next(view)[-1]["positions"], [0, 2, 4, 6])
        self.assertEqual(opened.reads, 1)

    def test_lazy_batch_snapshots_ids_role_and_condition(self):
        opened = self.open()
        for changed in ([81, 10], [3, 3], [999, 10]):
            ids = [3, 10]
            condition = {"mode": "uniform_stride", "stride": 2, "offset": 0}
            stream = opened.batch("devtrain", ids, condition)
            ids[:] = changed
            condition["offset"] = 1
            with patch.object(bank, "_decode", wraps=bank._decode) as reader:
                views = list(stream)
            self.assertEqual([call.args[3]["id"] for call in reader.call_args_list], [3, 10])
            self.assertTrue(all(call.args[3]["role"] == "devtrain" for call in reader.call_args_list))
            self.assertTrue(all(view[-1]["positions"] == [0, 2, 4, 6] for view in views))

    def test_public_bank_and_run_snapshots_cannot_change_admitted_input(self):
        opened = self.open()
        spec = training.contract(opened, 64, "off")
        run = training.Run(opened, spec)
        event = spec["exposure"][0].copy()
        row = opened.role_row("devtrain", event["tile"])
        expected_samples = [sample.copy() for sample in row["samples"]]
        before = opened.materialize(row, VIEWS["full"])[-1]
        copies = (opened.role_rows("devtrain")[0], opened.rows[row["id"]],
                  opened.role_row("devtrain", row["id"]))
        copies[0]["samples"][0][1] = 9
        copies[1]["role"] = "devval"
        copies[2]["box"][0] += 1
        release = opened.release
        release.statistics["minimum"][0] = 200.
        release.source["shared_state"] = "wrong"
        release.rows[0]["samples"][0][1] = 9
        opened.identity["statistics"]["minimum"][0] = 200.
        opened.pin["sha256"] = "wrong"
        opened.members[row["id"]]["content"]["bytes"] = 1
        before["box"][0] += 1
        before["statistics"]["minimum"][0] = 200.
        spec["exposure"][0]["tile"] = 81
        changed = run.contract
        changed["exposure"][0]["tile"] = 81
        changed["bank"]["sha256"] = "wrong"
        with (patch.object(bank, "_decode", side_effect=AssertionError("decoded altered input")) as reader,
              patch.object(training, "forward", side_effect=AssertionError("used altered input")) as forward,
              patch.object(run.optimizer, "step", side_effect=AssertionError("updated altered input")) as step):
            for copy in copies:
                with self.assertRaises(ValueError):
                    opened.materialize(copy, VIEWS["full"])
            with self.assertRaises(ValueError):
                run.update(changed["exposure"][0])
        reader.assert_not_called()
        forward.assert_not_called()
        step.assert_not_called()
        run.start()
        with patch.object(training, "_targets", wraps=training._targets) as targets:
            run.update(event)
        self.assertEqual(targets.call_args.args[0]["samples"], expected_samples)
        self.assertEqual(opened.materialize(row, VIEWS["full"])[-1]["source"],
                         self.release.source["shared_state"])

    def test_changed_authority_after_run_registration_rejects_start_and_update(self):
        opened = self.open()
        spec = training.contract(opened, 64, "off")
        started = training.Run(opened, spec)
        pending = training.Run(opened, spec)
        started.start()
        registration = self.dest / "registration.json"
        registration.chmod(0o644)
        registration.write_bytes(registration.read_bytes() + b" ")
        with (patch.object(bank, "_decode", side_effect=AssertionError("decoded altered authority")) as reader,
              patch.object(training, "forward", side_effect=AssertionError("used altered authority")) as forward,
              patch.object(started.optimizer, "step", side_effect=AssertionError("updated altered authority")) as step):
            with self.assertRaisesRegex(ValueError, "authority changed"):
                pending.start()
            with self.assertRaisesRegex(ValueError, "authority changed"):
                started.update(spec["exposure"][0])
        reader.assert_not_called()
        forward.assert_not_called()
        step.assert_not_called()

    def test_invalid_roster_statistics_and_alias_root_reject_before_reader(self):
        row = self.release.rows[0]
        changed = (
            replace(self.release, rows=(row, *self.release.rows)),
            replace(self.release, rows=(dict(row, box=[0, 99, 0, 9]), *self.release.rows[1:])),
            replace(self.release, rows=(dict(row, partition="val"), *self.release.rows[1:])),
            replace(self.release, rows=(dict(row, samples=[[9999, 1]]), *self.release.rows[1:])),
            replace(self.release, statistics=dict(self.release.statistics, maximum=[1.] * 8)),
        )
        for release in changed:
            with self.subTest(release=release), self.assertRaises(ValueError):
                bank.publish(self.dest, self.root, release, self.reader)
        with self.assertRaises(ValueError):
            bank.publish(self.root / ".." / "elsewhere", self.root, self.release, self.reader)
        self.reader.assert_not_called()

    def test_existing_manifest_checked_before_any_recovery_payload(self):
        bank.publish(self.dest, self.root, self.release, self.reader, stop=2)
        self.reader.reset_mock()
        item = self.dest / "tile-10.json"
        body = json.loads(item.read_text())
        body["row"]["role"] = "devval"
        item.chmod(0o644)
        item.write_text(json.dumps(body))
        with patch.object(bank, "_decode", side_effect=AssertionError("read before metadata checks")):
            with self.assertRaisesRegex(ValueError, "manifest"):
                bank.publish(self.dest, self.root, self.release, self.reader, resume=True)
        self.reader.assert_not_called()

    def test_pending_member_recovers_no_overwrite_and_original_exact(self):
        original = bank._atomic_new

        def fail_array(path, data):
            if path.name == "tile-10.npy":
                raise RuntimeError("simulated interruption after manifest")
            return original(path, data)

        with patch.object(bank, "_atomic_new", side_effect=fail_array), self.assertRaisesRegex(
                RuntimeError, "simulated interruption"):
            self.create()
        keep = [self.dest / "tile-3.npy", self.dest / "tile-3.json", self.dest / "tile-10.json"]
        prior = [hash_file(path) for path in keep]
        result = bank.publish(self.dest, self.root, self.release, self.reader, resume=True)
        self.assertEqual(result["recovered"], 1)
        self.assertEqual(result["committed"], 8)
        self.assertEqual([hash_file(path) for path in keep], prior)
        opened = bank.Bank(result, self.root, self.release)
        for key, item in opened.rows.items():
            got = opened.materialize(item, VIEWS["full"])
            np.testing.assert_array_equal(got[2].observation.spectral, (self.arrays[key] + 3.) / 6.)
        with self.assertRaises(FileExistsError):
            bank.publish(self.dest, self.root, self.release, Mock(side_effect=AssertionError), resume=True)

    def test_root_band_change_missing_member_and_content_hash(self):
        pin = self.create()
        with self.assertRaisesRegex(ValueError, "root hash"):
            bank.Bank(dict(pin, sha256="0" * 64), self.root, self.release)
        reverse = self.release.bands.subset(tuple(reversed(range(8))))
        with self.assertRaisesRegex(ValueError, "registration"):
            bank.Bank(pin, self.root, replace(self.release, bands=reverse))
        opened = bank.Bank(pin, self.root, self.release)
        member = self.dest / "tile-3.npy"
        original = member.read_bytes()
        member.chmod(0o644)
        member.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        with patch.object(np, "load", side_effect=AssertionError("decoded bad NPY")):
            with self.assertRaisesRegex(ValueError, "NPY digest"):
                opened.materialize(opened.rows[3], VIEWS["full"])
        member.unlink()
        with self.assertRaises(FileNotFoundError):
            bank.Bank(pin, self.root, self.release)

    def test_bounded_npy_header_rejects_shape_before_allocation(self):
        opened = self.open()
        record = opened.members[3]
        path = self.dest / record["member"]
        raw = path.read_bytes()
        forged = raw.replace(b"(8, 8, 8)", b"(9, 8, 8)", 1)
        self.assertNotEqual(forged, raw)
        path.chmod(0o644)
        path.write_bytes(forged)
        content = dict(record["content"], npy_sha256=hash_file(path))
        with patch.object(np, "load", side_effect=AssertionError("allocated using an unauthorized shape")):
            with self.assertRaisesRegex(ValueError, "NPY header"):
                bank._decode(self.dest, record["member"], content, opened.rows[3], 8)

    def test_source_identity_and_exact_synthetic_cursor(self):
        opened = self.open()
        spec = training.contract(opened, 64, "native")
        with self.assertRaises(ValueError):
            training.Run(opened, dict(spec, support_seed=0))
        run = training.Run(opened, spec)
        run.start()
        with patch.object(opened, "materialize", side_effect=AssertionError("read outside next exposure")):
            for event in (dict(step=1, tile=81, view="full"), dict(spec["exposure"][0], view="even_retained_positions")):
                with self.assertRaises(ValueError):
                    run.update(event)
        first = self.root / "prefix.pt"
        training.execute(opened, spec, "prefix", first)
        value = torch.load(first, map_location="cpu", weights_only=True)
        changed = dict(value, contract="invalid")
        with self.assertRaises(ValueError):
            training.execute(opened, spec, "resume", self.root / "wrong.pt", changed)
        checkpoint = self.root / "resume.pt"
        training.execute(opened, spec, "resume", checkpoint, value)
        restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("updated completed bank")):
            final = self.root / "complete.pt"
            self.assertEqual(training.execute(opened, spec, "completed", final, restored)["updates"], 0)
        self.assertTrue(_exact(torch.load(final, map_location="cpu", weights_only=True), restored))

    def test_duplicate_payloads_rejected(self):
        first, second = self.release.rows[0], self.release.rows[1]
        same = dict(second, origin=first["origin"], box=[2, 10, 12, 20], parent_box=[2, 10, 12, 20])
        release = replace(self.release, rows=(first, same, *self.release.rows[2:]))
        with self.assertRaisesRegex(ValueError, "Duplicate array"):
            bank.publish(self.dest, self.root, release, lambda row: self.arrays[3] if row["id"] == 10
                         else self.arrays[row["id"]])
        self.assertFalse((self.dest / "bank.json").exists())


if __name__ == "__main__":
    unittest.main()
