"""Frozen end-to-end approval and receipt integrity regressions."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation.wp12_frozen54_deployment import audit_stage, rebuild_stage, stagesummary
from scripts.wp12_frozen54_deployment_contract import (
    APPROVAL, COLD, COUNT, PHASES, SETUP, TIMING, WARM, WARMUPS, approval, digest_text, worker_binding,
)


def write(path, content):
    raw = json.dumps(content).encode()
    path.write_bytes(raw)
    return dict(path=str(path), sha256=digest_text(raw))


def phase(names, start):
    return {name: dict(start_ns=start+i*1000, finish_ns=start+(i+1)*1000, wall_ms=.001, cuda_peak_bytes=None)
            for i, name in enumerate(names)}


def fixture(root, scope, change=None):
    job = dict(checkpoint={"sha256": "a"*64}, workload=dict(
        file={"sha256": "b"*64}, tensor_sha256="c"*64, work_items=36), legacy_output="d"*64)
    lineage = dict(case="fixture", scope=scope, device="cpu", plan={"sha256":"f"*64})
    stage = root/scope
    attempt = stage/"attempt-fixture"
    attempt.mkdir(parents=True, exist_ok=True)
    sources = []
    for worker in range(COUNT[COLD] if scope == COLD else 1):
        spawn = worker*100_000+1000
        requests = []
        for i in range(1 if scope == COLD else WARMUPS[WARM]+COUNT[WARM]):
            start = spawn+(i+1)*10_000
            requests.append(dict(sequence=i, role="warmup" if i < WARMUPS[scope] else "measured",
                request_started_ns=start-100, first_prediction_ns=start+5000, phases=phase(PHASES,start),
                batch=1, observation_reads=1, graph_builds=0, skip_count=None, rss_kib=100, checkpoint="a"*64,
                observation="b"*64, workload="c"*64, pixels=36, prediction="e"*64, probability_state="d"*64))
        raw = dict(binding=worker_binding("f"*64, "fixture", scope, "cpu", attempt.name, worker),
                   timing=TIMING[scope], requests=requests, setup=phase(SETUP,spawn+100),
                   pid=worker+1, environment=dict(arithmetic="ordered", threads=1, device="cpu"),
                   optimizer_updates=0, query_accessed=False)
        if change is not None:
            change(raw, worker)
        content = write(attempt/f"worker-{worker}.json", raw)
        sources.append(write(attempt/f"worker-{worker}-invocation.json",
                             dict(stdout=content, process_started_ns=spawn, process_exited_ns=spawn+98_000)))
    item = rebuild_stage(root, stage, lineage, job, attempt.name, sources)
    return stage, item, lineage, job


class FrozenDeploymentTests(unittest.TestCase):
    def test_real_approval_binds_every_execution_field(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            plan_pin = dict(path=str(root/"plan"/"measurement-plan.json"), sha256="a"*64)
            plan = dict(engineering=False, execution=str(root/"run"), producer="b"*40)
            device = "GPU-0c958696-3721-4009-628f-befb0ef9b867"
            body = dict(schema=APPROVAL, decision="approved", plan_sha256="a"*64,
                        execution=plan["execution"], producer=plan["producer"], physical_uuid=device,
                        scopes=[COLD,WARM], data_access="sealed_observations_no_labels",
                        approval_reference="fixture approval")
            file = root/"permit.json"

            def permit(content):
                raw = json.dumps(content).encode()
                file.write_bytes(raw)
                return dict(path=str(file), sha256=digest_text(raw))

            approval(root, plan, plan_pin, permit(body), device)
            with self.assertRaisesRegex(ValueError, "Independent approval"):
                approval(root, plan, plan_pin, None, device)
            for field, value in (("decision", None), ("execution", str(root/"another")),
                                 ("physical_uuid", "cpu"), ("scopes", [WARM]), ("plan_sha256", "f"*64)):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "does not bind"):
                    approval(root, plan, plan_pin, permit(body | {field:value}), device)
            with self.assertRaisesRegex(ValueError, "do not consume"):
                approval(root, plan | {"engineering":True}, plan_pin, permit(body), device)

    def test_independent_audit_rejects_substituted_weights_observation_and_clocks(self):
        for fields, error in (({"checkpoint":"f"*64}, "checkpoint or observation"),
                              ({"observation":"f"*64}, "checkpoint or observation"),
                              ({"observation_reads":0}, "timing boundary"),
                              ({"first_prediction_ns":17000}, "host materialization"),
                              ({"role":"warmup"}, "sequence, role")):
            with self.subTest(fields=fields), TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, error):
                fixture(Path(tmp), COLD, lambda raw, _: raw["requests"][0].update(fields))

    def test_cold_and_warm_process_counts_and_repeat_budgets_are_frozen(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "process was reused"):
                fixture(Path(tmp), COLD, lambda raw, _: raw.update(pid=1))
            with self.assertRaisesRegex(ValueError, "repetition count"):
                fixture(Path(tmp), WARM, lambda raw, _: raw["requests"].pop())

    def test_six_copied_rows_fail_even_with_recomputed_percentiles(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage, record, lineage, job = fixture(root, WARM)
            self.assertEqual(audit_stage(root, stage, record, lineage, job), ("e"*64, "d"*64))
            self.assertEqual([row["sequence"] for row in record["samples"]], list(range(2,8)))
            altered = deepcopy(record)
            altered["samples"] = [altered["samples"][0]]*6
            altered["summary"] = stagesummary(altered["samples"], WARM)
            with self.assertRaisesRegex(ValueError, "pinned original worker requests"):
                audit_stage(root, stage, altered, lineage, job)

    def test_rehashed_worker_rejects_wrong_scope_device_or_timing(self):
        for target, fields, error in (
            ("binding", {"scope":COLD}, "attempt, scope or device"),
            ("binding", {"device":"GPU-2cbff259-d7db-32cf-b267-8f296e34b1d5"}, "attempt, scope or device"),
            ("environment", {"scope":"resident_forward"}, "timing boundary"),
            ("timing", {"start":"resident_forward"}, "timing boundary"),
        ):
            with self.subTest(fields=fields), TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, error):
                fixture(Path(tmp), WARM, lambda raw, _: raw[target].update(fields))
