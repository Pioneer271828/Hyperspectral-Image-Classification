"""Authenticate the reviewed PaviaU v1 campaign as read-only historical evidence."""
from pathlib import Path

from evaluation import frozen_probe_campaign as campaign, frozen_probe_registry as reg
from g4_support_context import check_files
from model.research_config import digest


POLICY = "wp06_pavia_v1_accepted_campaign_reuse_v1"
COMMIT = "0c512c3019088fe8b2c616cb2ee8db55c7cb2331"
SOURCE = "fabc2ba8abf4b9ec203c4f79ef0e918644e9d498ac171d86e599ffb7d9a28f45"
SHA256 = {
    "scope": "a1f2f36d3a391800691596432eca6396103044fd1fbed12cfb24b01ae281b49c",
    "execution": "8f8afa1ab99b62ebda39a5d1da88627ebb1699813df8a1111adc104ed734963a",
    "review": "0486f2f4e55fff9ab5349ce4b48a830bcc4e4d8c27c17a911605e024d1235d15",
    "acceptance": "20ce7f38f1e42e9023b26e9b453b094ebeb5ad36e113009f80670d58c8ad6ea2",
}


def open_history(scope_path, execution_path, review_path, *, synthetic=False):
    paths = {"scope": Path(scope_path).resolve(strict=True),
             "execution": Path(execution_path).resolve(strict=True),
             "review": Path(review_path).resolve(strict=True)}
    paths["acceptance"] = paths["review"].with_name("acceptance.json")
    reg.require([p.name for p in paths.values()] == ["scope.json", "execution.json", "receipt.json", "acceptance.json"]
                and len({paths[k].parent for k in ("scope", "execution", "review")}) == 3,
                "Distinct historical scope, execution and review roots required")
    payload = {key: reg.pinned_json(path, SHA256[key]) for key, path in paths.items()}
    scope, execution, review, acceptance = payload.values()
    expected_updates = 18*len(reg.LAYERS)*scope["probe_plan"]["steps"]*2
    reg.require((scope["commit"], scope["source_hash"]) == (COMMIT, SOURCE)
        and scope["policy"] == execution["policy"] == review["policy"] == acceptance["policy"] == reg.POLICY
        and scope["synthetic"] is review["synthetic"] is acceptance["synthetic"] is synthetic
        and execution["commit"] == review["commit"] == COMMIT
        and execution["status"] == "executed" and review["status"] == acceptance["status"] == "passed"
        and execution["scope_sha256"] == review["scope_sha256"] == acceptance["scope_sha256"] == SHA256["scope"]
        and review["execution_sha256"] == SHA256["execution"] and acceptance["independent_receipt_sha256"] == SHA256["review"]
        and scope["planned_probe_updates"] == execution["probe_updates"] == review["probe_updates"]
            == acceptance["probe_updates"] == expected_updates
        and review["accepted_cases"] == acceptance["cases"] == 18
        and review["accepted_probes"] == review["reviewed_probes"] == acceptance["probes"] == 72
        and review["failed_probes"] == review["audit_probe_updates"] == 0
        and review["strict_decisions"] is True
        and all(d["protected_files_unchanged"] is True and d["backbone_updates"] == d["bn_updates"] == 0
                and d["query_labels_accessed"] is False for d in (execution, review))
        and review["raw_gt_deserialized"] is False, "Historical PaviaU acceptance chain differs")
    return paths, payload


def verify_history(paths, payload, case_inputs, input_pins):
    scope, execution, review, _ = payload.values()
    scope_path = str(paths["scope"])
    execute_path = str(paths["execution"])
    protected = review["protected_files"]
    reg.require(scope["cases"] == [case for case, _, _ in case_inputs]
        and scope["layers"] == list(reg.LAYERS)
        and scope["protected_files_sha256"] == digest(input_pins)
        and scope["protected_file_count"] == len(input_pins)
        and scope["validation"] == "every_step_first_minimum_val_nll"
        and scope["optimizer"] == "cpu_fp64_full_batch_sgd_zero_initialization"
        and scope["columns"] == {stage: reg.probe_columns(64, 64, width)
                                 for stage, width in (("stage1", 128), ("stage2", 64))}
        and scope["raw_gt_deserialized"] is scope["query_labels_accessed"] is False
        and scope["backbone_updates"] == 0, "Historical registered roles, sources or probe protocol differ")
    plan, steps = scope["probe_plan"], scope["probe_plan"]["steps"]
    stops = [2, 4, steps] if scope["synthetic"] else reg.PATHS["split"]
    reg.require(plan == {**reg.PLAN, "steps": steps}
        and (scope["synthetic"] or steps == reg.PLAN["steps"])
        and scope["paths"] == {"full": [steps], "split": stops, "completed": [steps]},
        "Historical PaviaU plan or path differs")
    check_files({**protected, **review["outputs"]})
    expected_pins = {**input_pins, scope_path: SHA256["scope"], execute_path: SHA256["execution"]}
    results = {}
    count = len(reg.LAYERS)*steps*2
    for row, verdict, (case, _, local) in zip(execution["cases"], review["cases"], case_inputs, strict=True):
        name = case["name"]
        reg.require(row["case"] == verdict["case"] == name and row["status"] == "executed"
                    and verdict["status"] == "accepted" and row["probe_updates"] == verdict["probe_updates"] == count,
                    "Historical execution or review role differs")
        folder = campaign.folder_at(paths["execution"].parent, name)
        _, _, _, _, path_updates, files = campaign.checked_execution(folder, row["execution_sha256"],
            scope, SHA256["scope"], case, local)
        expected_pins.update({str(folder/"execution.json"): row["execution_sha256"], **files})
        result_path = campaign.folder_at(paths["review"].parent, name+".json")
        proof = reg.pinned_json(result_path, review["outputs"][str(result_path)])
        reg.require(proof["case"] == name and proof["role"] == case and proof["status"] == "accepted"
                    and proof["probe_updates"] == count and proof["audit_probe_updates"] == 0
                    and proof["updates_by_path"] == path_updates
                    and proof["exact_continuous_split_and_completed"] is True
                    and proof["query_labels_accessed"] is proof["raw_gt_deserialized"] is False
                    and set(proof["layers"]) == set(reg.LAYERS) and not proof["numerical_failures"],
                    "Historical independent review differs from frozen role")
        results[name] = proof
    reg.require(protected == expected_pins and set(review["outputs"]) == {
        str(paths["review"].parent/name) for name in ("summary.json", "metrics.csv", "paired-differences.csv",
            *(name+".json" for name in results))}, "Historical review file coverage differs")
    pins = {**protected, **review["outputs"], **{str(path): SHA256[key] for key, path in paths.items()}}
    identity = {"policy": POLICY, "commit": COMMIT, "source_hash": SOURCE, "probe_updates": review["probe_updates"],
                "sources": {key: {"path": str(path), "sha256": SHA256[key]} for key, path in paths.items()}}
    return identity, results, pins
