"""Read stage-aware synthetic coverage receipts without running the trainer."""

from pathlib import Path

from evaluation.frozen_training_stages import (HEADS, POLICY, VIEWS, file_pin, json_pin,
                                               paired, require, scores)
from model.research_config import digest
from transfer.native_coverage_queue import load_queue


def publish(root, queue_pin, code):
    queue_path = Path(queue_pin["path"])
    queue = json_pin(root, queue_pin)
    require(queue_path.name == "queue.json" and load_queue(queue_path.parent, code) == queue
            and queue["frozen_stage_protocol"] == POLICY and queue["scope"] == "synthetic_only"
            and queue["query_accessed"] is queue["raw_gt_loaded"] is False, "Registered synthetic stage queue required")
    out = {}
    for name, profile in queue["cases"].items():
        path = queue_path.parent/"cases"/"cpu"/name/"receipt.json"
        pin = file_pin(path)
        case = json_pin(root, pin)
        spec = json_pin(root, case["registration_file"])
        require(spec["frozen_stage_protocol"] == POLICY and spec["case"] == profile
                and case["registration"] == digest(spec) and spec["steps"] == 24
                and case["optimizer_updates"] == 48 and case["real_updates"] == 0
                and case["query_accessed"] is False and case["resume_bitwise_equal"] is True
                and case["completed_bitwise_equal"] is True
                and [case["stages"][phase]["updates"] for phase in ("full", "prefix", "resume", "completed")]
                == [24, 12, 12, 0], "Synthetic native exposure or exact re-entry differs")
        expected = {"initial": 0, "intermediate": 12, "selected": case["selected_step"], "last": 24}
        require(set(case["frozen_stages"]) == expected.keys(), "Four synthetic stage roles required")
        base = case["frozen_stages"]["initial"]["views"]["full"]
        results = {}
        for name_stage, expected_step in expected.items():
            stage = case["frozen_stages"][name_stage]
            require(stage["step"] == expected_step and set(stage["views"]) == set(VIEWS),
                    "Synthetic stage boundary or views differ")
            full = stage["views"]["full"]
            views = {}
            for band, points in stage["views"].items():
                reduced = scores(points, spec["classes"])
                trainer = stage["trainer"][band]
                require(all(reduced[head]["confusion"] == trainer["heads"][head]["confusion"]
                            and reduced[head]["count"] == trainer["heads"][head]["n"] for head in HEADS),
                        "Independent synthetic point statistics differ")
                views[band] = dict(heads=reduced, transitions=paired(full, points, spec["classes"], {}))
            results[name_stage] = dict(step=expected_step, state=stage["state_sha256"], views=views,
                                       initial_to_full=paired(base, full, spec["classes"], {}))
        out[name] = dict(receipt=pin["sha256"], profile=profile, stages=results)
    return dict(policy=POLICY, scope="synthetic_engineering_only", cases=out, full_optimizer_updates=24*len(out),
                recovery_optimizer_updates=24*len(out), audit_optimizer_updates=0,
                completed_reentry_updates=0, raw_gt_loaded=False, query_accessed=False)
