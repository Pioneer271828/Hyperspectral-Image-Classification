"""Independently compare a prepared resource scope with the original HongHu contract."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.research_config import digest
from research_runtime import code_snapshot, external_directory, write_json
from transfer import p4_independent_acceptance as acceptance, p4_independent_registration as resources
from transfer import p4_coverage_batch as batch, p4_three_arm_contract as fit


ARMS = ("none", "anchor_only", "binary", "continuous", "shuffled")


def compare(original, new):
    resources.require(original["sha256"] == fit.ADMISSION_SHA, "Original admission SHA differs")
    old, proposed = resources.read(original), resources.read(new)
    fresh = resources.validate(proposed, live_worker=True)
    resources.require(proposed["mode"] == resources.REAL and proposed["science"] == original
                      and set(old) == set(fresh) == set(proposed["science_fields"])
                      and all(old[key] == fresh[key] and proposed["science_fields"][key] == digest(old[key])
                              for key in old), "Original and prepared science fields differ")
    pins = old["protected_files"]
    registration_path = old["registration"]
    registration = resources.read({"path": registration_path, "sha256": pins[registration_path]})
    path = registration["exposure"]["plan"]
    exposure = resources.read({"path": path, "sha256": pins[path]})
    roles = {}
    for case in old["cases"]:
        key = (case["k"], case["method"])
        resources.require(case["support_seed"] == 2048 and case["model_seed"] == 512
                          and case["sampler_seed"] == 64 and case["protocol"] == old["protocol"],
                          "P4 role, seeds or arithmetic changed")
        roles.setdefault(key, []).append(case)
    expected = {(k, method) for k in (1, 5) for method in ("scratch", "shared_transfer")}
    resources.require(roles.keys() == expected, "Original paired roles differ")
    for (k, method), cases in roles.items():
        uuid = (resources.GPU1, resources.GPU6)[((k == 5) + (method == "shared_transfer")) % 2]
        resources.require(tuple(row["arm"] for row in cases) == ARMS
                          and {row["physical_uuid"] for row in cases} == {uuid}
                          and set(row["id"] for row in cases) <= set(proposed["slots"][uuid]["cases"]),
                          "Paired role arms or exact physical assignment differ")
    resources.require(set(old["observation_diagnostics"]) == set(fit.CONDITIONS)
                      and old["paths"] == {"full": [60], "split": [2, 30, 60]}
                      and old["budget"]["validation_steps"] == [30, 60]
                      and old["budget"]["selection"] == registration["coverage"]["selection"]
                      and registration["coverage"]["steps"] == exposure["steps"] == 60,
                      "Frozen views, paired paths or selection changed")
    batch.check_pins(pins)
    return {"status": "verified_preparation", "reference": original, "resource_scope": new,
            "science_field_sha256": proposed["science_fields"], "registration_content_sha256": {
                key: digest(value) for key, value in sorted(registration.items())},
            "exposure_plan_sha256": pins[path], "slot_budget": proposed["slots"],
            "selection": old["budget"]["selection"], "protocol": old["protocol"],
            "condition_names": sorted(old["observation_diagnostics"]),
            "model_config": registration["config"]["model"], "loss": registration["config"]["loss"],
            "training": registration["config"]["training"],
            "protected_science_inputs": len(pins), "old_qualification": {
                gpu: pins[path] for gpu, path in old["qualifications"].items()},
            "same_device_recoveries": proposed["physical_recoveries"],
            "optimizer_updates": 0, "query_accessed": False, "raw_gt_loaded": False}


def negative_cases(original, new, out, code):
    resource = resources.read(new)
    science = resources.read(original)
    result = {}

    def reject(name, callback, text):
        try:
            callback()
        except (ValueError, FileNotFoundError) as error:
            if text not in str(error):
                raise
            result[name] = str(error)
            return
        raise AssertionError(f"Unexpectedly accepted {name}")

    for field, replacement in {"seed": {"cases": [{**science["cases"][0], "model_seed": 64},
                                                    *science["cases"][1:]]},
                               "budget": {"budget": {**science["budget"], "planned_optimizer_updates": 2399}},
                               "numeric": {"protocol": "cpu_fp32_v1"}}.items():
        bad = out / (field + "-forged-science.json")
        write_json(bad, {**science, **replacement})
        bogus = {**resource, "science": resources.pin(bad),
                 "execution": str(out.parent / "forged-input-unused-execution")}
        bogus["science_fields"] = {key: digest(row) for key, row in resources.read(bogus["science"]).items()}
        reject(field, lambda: resources.validate(bogus), "exact original")
    forged_resource = out / "forged-worker.json"
    write_json(forged_resource, {**resource, "worker": {**resource["worker"], "source_hash": "b"*64}})
    reject("worker_source", lambda: resources.validate(resources.read(resources.pin(forged_resource))), "identity")
    bad_qual = {**resource["physical_recoveries"], resources.GPU6: resource["physical_recoveries"][resources.GPU1]}
    reject("qualification", lambda: resources.validate({**resource, "physical_recoveries": bad_qual},
                                                        live_worker=True), "Recovery UUID differs")

    # Use the real prepared scope and a fabricated summary in its own isolated
    # directory. A fresh fixture tests the release consumer without any training.
    execution = out.parent / (out.name + "-fabricated-execution")
    execution.mkdir()
    synthetic_resource = {**resource, "execution": str(execution)}
    replacement = out / "fabricated-resource.json"
    write_json(replacement, synthetic_resource)
    new_pin = resources.pin(replacement)
    receipt = execution / "independent-acceptance.json"
    staged = {"schema": acceptance.SCHEMA, "status": "passed", "synthetic": False,
              "cases": 20, "optimizer_updates": 2400}
    target = str(out.parent / "future-fit-unreleased")
    quals = fit.REFERENCE_QUALIFICATION_SHA
    gpus = [resources.GPU1, resources.GPU6]
    approval = {"schema": "wp05_three_arm_execution_release_v2", "decision": "approved",
                "source_commit": code["commit"], "source_hash": code["source_hash"],
                "research_issue": 470, "qualification_sha256": quals, "gpus": gpus,
                "execution": target, "planned_optimizer_updates": 13500,
                "admission_scope": new_pin, "admission_controller": resource["controller"]}
    approval_file = out / "synthetic-negative-release.json"

    def call(row):
        write_json(approval_file, {**approval, "admission_acceptance": resources.pin(receipt), **row})
        return fit.release_pins(resources.pin(approval_file), code, gpus, quals, target)

    for name, row, expected in (("partial_predecessor", {**staged, "cases": 10}, "Full 20-case"),
                                ("fabricated_poisoned_ancestor", staged, "resource-owner.json")):
        write_json(receipt, row)
        reject(name, lambda: call({}), expected)
    reject("controller_mismatch", lambda: call({"admission_controller": {
        "commit": "0"*40, "source_hash": "0"*64}}), "Expected separately approved")
    reject("unknown_release_version", lambda: call({"schema": "forged"}), "reviewed 13500-step")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    for key in ("original", "resource", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    resources.require(not code["dirty"] and code["commit"] == args.expected_commit, "Unexpected audit source")
    out = external_directory(args.out)
    out.mkdir()
    original, new = resources.pin(args.original), resources.pin(args.resource)
    result = compare(original, new)
    result["rejected_inputs"] = negative_cases(original, new, out, code)
    write_json(out / "preflight.json", result)
    print(resources.pin(out / "preflight.json"), flush=True)


if __name__ == "__main__":
    main()
