"""Execute the fixed continuous/recovery probe paths and audit their full grid."""
from pathlib import Path
import time

import torch

from evaluation import authenticated_probe_pack as packs, frozen_probe_audit as audit
from evaluation import frozen_probe_registry as registry, frozen_probe_summary as summary
from evaluation.linear_probe import advance, fingerprint
from g4_matched_target_evaluation import new_output
from g4_support_context import check_files
from joint_training import _atomic_torch_save
from research_runtime import hash_file, require_frozen_code, write_json


def sibling(out, launch, scope, cases, scope_path, *other):
    owners = [Path(scope[name]).parent for name in ("freeze", "analysis", "capture_index")]
    owners.extend(Path(case["capture_receipt"]).parent for case, _, _ in cases)
    return new_output(out, launch, (*owners, Path(scope_path).resolve().parent, *other))


def folder_at(parent, name):
    path = parent / name
    registry.require(path.resolve() == path, f"Probe output traverses a symlink: {path}")
    return path


def pins_for(folder, names):
    return {str(folder / name): hash_file(folder / name) for name in names}


def segment_name(path, until):
    return f"{path}-{until:04d}"


def schedule(scope):
    plan_steps = scope["probe_plan"]["steps"]
    for path in ("full", "split", "completed"):
        stops = scope["paths"][path]
        previous = None if path in ("full", "split") else segment_name("split", plan_steps)
        start = 0 if previous is None else plan_steps
        for until in stops:
            name = segment_name(path, until)
            yield path, start, until, previous, name
            start, previous = until, name


def read_state(path, sha):
    check_files({str(path): sha})
    state = torch.load(path, map_location="cpu", weights_only=True)
    registry.require(state["checksum"] == fingerprint({k: v for k, v in state.items() if k != "checksum"}),
                     "Saved probe checkpoint contents changed")
    return state


def terminal_pack_contract(case):
    return {"policy": packs.POLICY, "selected_model_sha256": case["primary_sha256"]}


def prepare_pack(folder, bound, case, local, scope_sha, *, writer=packs._pack, contract=terminal_pack_contract):
    check_files(local)
    pack = writer(bound, local_only=True)
    path = folder_at(folder, "pack.pt")
    _atomic_torch_save(pack, path)
    identity = fingerprint(pack)
    receipt = {**contract(case), "scope_sha256": scope_sha, "case": case["name"],
               "capture_receipt_sha256": case["capture_receipt_sha256"],
               "pack_identity": identity, "pack_sha256": hash_file(path), "local_source_files": local,
               "train_rows": len(pack["labels"]["train"]), "val_rows": len(pack["labels"]["val"]),
               "backbone_updates": 0, "query_labels_accessed": False}
    check_files(local)
    write_json(folder / "pack-receipt.json", receipt)
    return pack, receipt


def execute_segment(folder, pack, pack_receipt, plan, case, scope_sha, path, start, until, previous, written):
    started = time.monotonic()
    registry.require(not folder.exists(), "Probe segment output already exists")
    parent_sha = None
    state = None
    if previous is not None:
        previous_file, parent_sha = written[previous]
        state = read_state(previous_file, parent_sha)
        registry.require(state["cursor"] == start, "Probe resume cursor differs from the registered stop")
    stop_after = until-start or None
    state, report = advance(pack, plan, state=state, stop_after=stop_after)
    registry.require(state["cursor"] == until and report["probe_updates_this_call"] == len(registry.LAYERS)*(until-start)
        and report["backbone_updates"] == 0 and report["query_labels_accessed"] is False
        and report["input_identity"] == pack_receipt["pack_identity"] and report["input_unchanged"],
        "Actual probe update ledger differs from the fixed plan")
    folder.mkdir(exist_ok=False)
    file = folder / "state.pt"
    _atomic_torch_save(state, file)
    write_json(folder / "report.json", report)
    sha = hash_file(file)
    receipt = {"case": case["name"], "scope_sha256": scope_sha, "path": path, "start": start, "until": until,
               "previous": previous, "previous_state_sha256": parent_sha, "pack_sha256": pack_receipt["pack_sha256"],
               "scientific_checksum": state["checksum"], "state_sha256": sha,
               "report_sha256": hash_file(folder/"report.json"), "probe_updates": report["probe_updates_this_call"],
               "wall_seconds": time.monotonic()-started, "backbone_updates": 0, "query_labels_accessed": False}
    write_json(folder / "receipt.json", receipt)
    return file, sha, receipt


def run_case(folder, bound, case, local, scope, scope_sha, record_update, *, writer=packs._pack,
             contract=terminal_pack_contract):
    folder.mkdir(exist_ok=False)
    started = time.monotonic()
    pack, pack_receipt = prepare_pack(folder, bound, case, local, scope_sha, writer=writer, contract=contract)
    written, segments = {}, {}
    for path, start, until, previous, name in schedule(scope):
        state_path, sha, receipt = execute_segment(folder_at(folder, name), pack, pack_receipt, scope["probe_plan"],
            case, scope_sha, path, start, until, previous, written)
        written[name] = state_path, sha
        segments[name] = receipt
        record_update(receipt)
    steps = scope["probe_plan"]["steps"]
    names = [segment_name(path, steps) for path in ("full", "split", "completed")]
    full, split, completed = [read_state(*written[name]) for name in names]
    registry.require(full["checksum"] == split["checksum"] == completed["checksum"],
                     "Continuous, segmented or completed probe checkpoints differ")
    _atomic_torch_save(audit.endpoint_logp(pack, scope["probe_plan"], full), folder / "selected-logp.pt")
    check_files(local)
    probe_updates = sum(row["probe_updates"] for row in segments.values())
    files = pins_for(folder, ("pack.pt", "pack-receipt.json", "selected-logp.pt"))
    for name in segments:
        files.update(pins_for(folder / name, ("state.pt", "report.json", "receipt.json")))
    result = {"case": case["name"], "scope_sha256": scope_sha, "pack_sha256": pack_receipt["pack_sha256"],
              "scientific_checksum": full["checksum"], "probe_updates": probe_updates,
              "completed_resume_updates": segments[names[-1]]["probe_updates"], "files": files,
              "wall_seconds": time.monotonic()-started, "backbone_updates": 0, "query_labels_accessed": False}
    path = folder / "execution.json"
    write_json(path, result)
    return hash_file(path)


def run(scope_path, scope_sha, out, code):
    started = time.monotonic()
    scope, launch, cases, pins, auth_seconds = registry.load(scope_path, scope_sha, code)
    out = sibling(out, launch, scope, cases, scope_path)
    out.mkdir(exist_ok=False)
    rows = [{"case": row["name"], "status": "not_run", "probe_updates": 0} for row, _, _ in cases]
    state = {"policy": registry.POLICY, "scope_sha256": scope_sha, "commit": code["commit"], "cases": rows,
             "status": "running", "backbone_updates": 0, "bn_updates": 0, "query_labels_accessed": False,
             "input_authentication_seconds": auth_seconds}
    status_file = out / "execution.json"
    write_json(status_file, state)
    for status, (case, bound, local) in zip(rows, cases, strict=True):
        status["status"] = "running"

        def record_update(receipt):
            status["probe_updates"] += receipt["probe_updates"]
            write_json(status_file, state)

        try:
            receipt_sha = run_case(folder_at(out, case["name"]), bound, case, local, scope, scope_sha, record_update)
            status.update(status="executed", execution_sha256=receipt_sha)
            write_json(status_file, state)
        except Exception as error:
            status.update(status="execution_failed", error=repr(error))
            state["status"] = "execution_failed"
            break
    else:
        state["status"] = "executed"
    state["probe_updates"] = sum(row["probe_updates"] for row in rows)
    state["wall_seconds"] = time.monotonic()-started
    write_json(status_file, state)
    check_files(pins)
    require_frozen_code(code)
    state["protected_files_unchanged"] = True
    write_json(status_file, state)
    return status_file


def checked_execution(folder, expected_sha, scope, scope_sha, case, local, *, contract=terminal_pack_contract):
    execution = registry.pinned_json(folder / "execution.json", expected_sha)
    names = list(schedule(scope))
    files = pins_for(folder, ("pack.pt", "pack-receipt.json", "selected-logp.pt"))
    for *_, name in names:
        files.update(pins_for(folder_at(folder, name), ("state.pt", "report.json", "receipt.json")))
    registry.require(files == execution["files"] and execution["case"] == case["name"]
        and execution["scope_sha256"] == scope_sha and execution["completed_resume_updates"] == 0
        and execution["backbone_updates"] == 0 and execution["query_labels_accessed"] is False,
        "Executed probe segment files or scope differ")
    pack_path = folder/"pack.pt"
    pack_receipt = registry.pinned_json(folder/"pack-receipt.json", files[str(folder/"pack-receipt.json")])
    registry.require(pack_receipt["local_source_files"] == local and pack_receipt["case"] == case["name"]
        and pack_receipt["scope_sha256"] == scope_sha and pack_receipt["capture_receipt_sha256"] == case["capture_receipt_sha256"]
        and all(pack_receipt.get(key) == value for key, value in contract(case).items())
        and pack_receipt["pack_sha256"] == execution["pack_sha256"] == files[str(pack_path)]
        and pack_receipt["backbone_updates"] == 0 and pack_receipt["query_labels_accessed"] is False,
        "Authenticated pack receipt differs")
    pack = torch.load(pack_path, map_location="cpu", weights_only=True)
    registry.require(fingerprint(pack) == pack_receipt["pack_identity"], "Pack fingerprint differs")
    states, updates_by_path = {}, {path: 0 for path in scope["paths"]}
    for path, start, until, previous, name in names:
        segment = folder / name
        receipt_path, report_path, state_path = (segment / n for n in ("receipt.json", "report.json", "state.pt"))
        receipt = registry.pinned_json(receipt_path, files[str(receipt_path)])
        expected_parent = None if previous is None else files[str(folder/previous/"state.pt")]
        expected_updates = len(registry.LAYERS)*(until-start)
        registry.require((receipt["path"], receipt["start"], receipt["until"], receipt["previous"],
            receipt["previous_state_sha256"], receipt["state_sha256"], receipt["report_sha256"],
            receipt["pack_sha256"], receipt["scope_sha256"], receipt["case"], receipt["probe_updates"],
            receipt["backbone_updates"], receipt["query_labels_accessed"]) ==
            (path, start, until, previous, expected_parent, files[str(state_path)], files[str(report_path)],
             pack_receipt["pack_sha256"], scope_sha, case["name"], expected_updates, 0, False),
            "Resume path, update interval or phase receipt differs")
        state = read_state(state_path, receipt["state_sha256"])
        report = registry.pinned_json(report_path, receipt["report_sha256"])
        registry.require(state["cursor"] == until and state["checksum"] == receipt["scientific_checksum"]
            and report["plan"] == scope["probe_plan"] and report["input_identity"] == fingerprint(pack)
            and report["updates_this_call_per_probe"] == until-start
            and report["probe_updates_this_call"] == expected_updates and report["input_unchanged"] is True
            and report["backbone_updates"] == 0 and report["query_labels_accessed"] is False
            and set(state["layers"]) == set(report["layers"]) == set(registry.LAYERS),
            "Phase reports are not bound to the registered pack/plan/trajectory")
        states[path] = state, report
        updates_by_path[path] += expected_updates
    registry.require(len({state["checksum"] for state, _ in states.values()}) == 1
        and states["full"][0]["checksum"] == execution["scientific_checksum"]
        and sum(updates_by_path.values()) == execution["probe_updates"], "Path pairing or update accounting differs")
    recorded = torch.load(folder / "selected-logp.pt", map_location="cpu", weights_only=True)
    return pack, *states["full"], recorded, updates_by_path, files


def review(scope_path, scope_sha, run_path, run_sha, out, code):
    started = time.monotonic()
    scope, launch, cases, inputs, auth_seconds = registry.load(scope_path, scope_sha, code)
    run_path = Path(run_path).resolve(strict=True)
    run = registry.pinned_json(run_path, run_sha)
    rows = run["cases"]
    registry.require(run_path.name == "execution.json" and run["policy"] == registry.POLICY
        and run["commit"] == code["commit"] and run["scope_sha256"] == scope_sha
        and run["status"] in ("executed", "execution_failed") and run["protected_files_unchanged"] is True
        and run["backbone_updates"] == run["bn_updates"] == 0 and run["query_labels_accessed"] is False
        and [row["case"] for row in rows] == [case["name"] for case, _, _ in cases]
        and all(row["status"] in ("executed", "execution_failed", "not_run") for row in rows)
        and run["probe_updates"] == sum(row["probe_updates"] for row in rows),
        "Batch execution identity or role coverage differs")
    out = sibling(out, launch, scope, cases, scope_path, run_path.parent)
    out.mkdir(exist_ok=False)
    results, artifacts = [], {str(run_path): run_sha}
    for status, (case, bound, local) in zip(rows, cases, strict=True):
        if status["status"] != "executed":
            results.append({**status, "role": case})
            continue
        case_folder = folder_at(run_path.parent, case["name"])
        path = folder_at(out, case["name"]+".json")
        pack, state, report, recorded, counts, files = checked_execution(case_folder,
            status["execution_sha256"], scope, scope_sha, case, local)
        registry.require(status["probe_updates"] == sum(counts.values()), "Actual completed update tally differs")
        verified_pack = audit.audit_pack(pack, bound, scope, case, local)
        metrics = audit.audit_probes(pack, scope, state, report, recorded)
        result = {**metrics, **verified_pack, "role": case, "case": case["name"],
                  "status": "failed_numerical_gate" if metrics["numerical_failures"] else "accepted",
                  "exact_continuous_split_and_completed": True, "updates_by_path": counts,
                  "probe_updates": sum(counts.values()), "audit_probe_updates": 0,
                  "raw_gt_deserialized": False, "query_labels_accessed": False}
        write_json(path, result)
        results.append(result)
        artifacts.update({str(case_folder/"execution.json"): status["execution_sha256"], **files})
    complete = (all(r["status"] == "accepted" for r in results)
                and run["status"] == "executed" and run["probe_updates"] == scope["planned_probe_updates"])
    aggregate_files = summary.write_outputs(out, results)
    reviewed = [row for row in results if "layers" in row]
    failed_probes = sum(len({fail["layer"] for fail in row["numerical_failures"]}) for row in reviewed)
    all_pins = {**inputs, **artifacts}
    check_files(all_pins)
    require_frozen_code(code)
    receipt = {"policy": registry.POLICY, "status": "passed" if complete else "incomplete_or_failed",
               "commit": code["commit"], "scope_sha256": scope_sha, "execution_sha256": run_sha,
               "synthetic": scope["synthetic"], "cases": [{k: row[k] for k in ("case", "status", "probe_updates")} for row in results],
               "accepted_cases": sum(row["status"] == "accepted" for row in results),
               "strict_decisions": complete, "probe_updates": run["probe_updates"], "audit_probe_updates": 0,
               "reviewed_probes": len(registry.LAYERS)*len(reviewed),
               "accepted_probes": len(registry.LAYERS)*len(reviewed)-failed_probes,
               "failed_probes": failed_probes,
               "new_frozen_captures": sum(case["origin"] == "captured" for case, _, _ in cases),
               "reused_frozen_captures": sum(case["origin"] == "reused" for case, _, _ in cases),
               "backbone_updates": 0, "bn_updates": 0, "raw_gt_deserialized": False, "query_labels_accessed": False,
               "protected_files": all_pins, "protected_files_unchanged": True,
               "outputs": pins_for(out, aggregate_files + [r["case"]+".json" for r in results if "layers" in r]),
               "input_authentication_seconds": auth_seconds, "wall_seconds": time.monotonic()-started}
    result_path = out / "receipt.json"
    write_json(result_path, receipt)
    if complete:
        write_json(out/"acceptance.json", {"status": "passed", "policy": registry.POLICY, "scope_sha256": scope_sha,
            "independent_receipt_sha256": hash_file(result_path), "cases": 18, "probes": 72,
            "probe_updates": receipt["probe_updates"], "synthetic": scope["synthetic"]})
    return result_path
