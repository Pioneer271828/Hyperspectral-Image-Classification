"""Frozen end-to-end approval and receipt integrity regressions."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation.wp12_frozen54_deployment import audit_stage, stagesummary
from scripts.wp12_frozen54_deployment_contract import (
    APPROVAL, COLD, COUNT, PHASES, SETUP, WARM, WARMUPS, approval, digest_text,
)


def accepted(scope):
    job = dict(checkpoint={"sha256": "a"*64}, workload=dict(
        file={"sha256": "b"*64}, tensor_sha256="c"*64, work_items=36), legacy_output="d"*64)
    lineage = dict(case="fixture", scope=scope, device="cpu")
    samples = []
    for i in range(COUNT[scope]):
        wall = 20 if scope == COLD else 6
        samples.append(dict(scope=scope, device="cpu", batch=1, observation_reads=1, graph_builds=0,
            skip_count=None, pid=i+1 if scope == COLD else 1, wall_ms=wall, request_ms=6,
            unattributed_ms=wall-len(PHASES)-(len(SETUP) if scope == COLD else 0),
            process_exit_ms=25, cuda_peak_bytes=None, rss_kib=100, checkpoint="a"*64,
            observation="b"*64, workload="c"*64, pixels=36, prediction="e"*64, probability_state="d"*64,
            phases={p:dict(wall_ms=1, cuda_peak_bytes=None) for p in PHASES},
            setup={p:dict(wall_ms=1, cuda_peak_bytes=None) for p in SETUP}))
    record = dict(lineage=lineage, samples=samples, warmups=WARMUPS[scope], batch=1, optimizer_updates=0,
                  query_accessed=False, environment=dict(arithmetic="ordered", threads=1, device="cpu"),
                  summary=stagesummary(samples))
    return record, lineage, job


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
        for scope in (COLD, WARM):
            record, lineage, job = accepted(scope)
            self.assertEqual(audit_stage(record, lineage, job), ("e"*64, "d"*64))
            for label, values, error in (
                ("checkpoint", {"checkpoint":"f"*64}, "checkpoint or observation"),
                ("observation", {"observation":"f"*64}, "checkpoint or observation"),
                ("scope", {"scope":"resident_forward"}, "timing boundary"),
                ("read", {"observation_reads":0}, "timing boundary"),
                ("NaN", {"unattributed_ms":float("nan")}, "timing boundary"),
                ("impossible deadline", {"process_exit_ms":1}, "Stage wall clock"),
            ):
                changed = deepcopy(record)
                changed["samples"][0].update(values)
                with self.subTest(scope=scope,label=label), self.assertRaisesRegex(ValueError, error):
                    audit_stage(changed, lineage, job)

    def test_cold_and_warm_process_counts_and_repeat_budgets_are_frozen(self):
        record, lineage, job = accepted(COLD)
        record["samples"][-1]["pid"] = record["samples"][0]["pid"]
        with self.assertRaisesRegex(ValueError, "fresh processes"):
            audit_stage(record, lineage, job)
        record, lineage, job = accepted(WARM)
        record["samples"].pop()
        with self.assertRaisesRegex(ValueError, "repetition count"):
            audit_stage(record, lineage, job)
