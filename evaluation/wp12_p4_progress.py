"""Observe immutable P4 stage acknowledgments and the coordinator's current lease."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re

from research_runtime import hash_file
from transfer.development_blocks import read_pin


EXPECTED_OPERATIONS = (
    ("full", "advance", 0, 60), ("full", "completed", 60, None),
    ("split", "advance", 0, 2), ("split", "advance", 2, 30), ("split", "advance", 30, 60),
    ("split", "completed", 60, None), (None, "audit", 60, None),
)


@contextmanager
def observe_lock(path):
    try:
        stream = path.open("rb")
    except FileNotFoundError:
        yield None
        return
    with stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            yield True
            return
        yield False


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def case_stages(out, case, resource_sha, science_sha, slot):
    if not re.fullmatch(r"s[0-9]+-k[0-9]+-[a-z_]+-m[0-9]+-[a-z_]+", case):
        raise ValueError("Invalid frozen case name")
    folder = out / "stages" / case
    updates = phases = 0
    committed = []
    receipt_names = {path.name for path in folder.glob("*.receipt.json")}
    missing = False
    for i, (name, operation, start, until) in enumerate(EXPECTED_OPERATIONS):
        path = folder / f"{i:02d}.receipt.json"
        intent = folder / f"{i:02d}.intent.json"
        if path.name not in receipt_names:
            missing = True
            continue
        if missing:
            raise ValueError("An earlier acknowledgment is missing before a committed stage")
        saved = read(path)
        proof = Path(saved["evidence"]["path"])
        expected = out / "cases" / case
        if name is not None:
            expected /= name
        if operation == "advance":
            expected /= f"calls/{start:06d}-{until:06d}.json"
        else:
            expected /= "audit.json" if operation == "audit" else "completed-resume.json"
        if (set(saved) != {"attempt", "evidence"} or proof != expected or proof.resolve() != proof
                or hash_file(proof) != saved["evidence"]["sha256"] or saved["attempt"]["path"] != str(intent)
                or intent.resolve() != intent or hash_file(intent) != saved["attempt"]["sha256"]):
            raise ValueError("Partial progress stage fingerprint differs")
        binding = {"resource_sha256": resource_sha, "slot": slot, "case": case, "path": name,
                   "operation": operation, "start": start, "until": until}
        attempt, evidence = read(intent), read(proof)
        if any(attempt.get(key) != value for key, value in binding.items()) or evidence["scope_sha256"] != science_sha:
            raise ValueError("Partial progress does not bind the registered dispatch and science")
        if operation == "audit":
            if evidence["case"]["id"] != case or evidence["status"] != "passed" or evidence["optimizer_updates"] != 120:
                raise ValueError("Partial audit case evidence differs")
        else:
            if evidence["case"] != case or evidence["path"] != name:
                raise ValueError("Partial worker evidence case or path differs")
            if operation == "advance":
                if (evidence["start"], evidence["until"], evidence["new_optimizer_updates"]) != (
                        start, until, until-start):
                    raise ValueError("Partial worker update ledger differs")
                updates += until - start
            elif (evidence["cursor"], evidence["new_optimizer_updates"]) != (start, 0):
                raise ValueError("Completed recovery consumed an update")
        committed.append({"path": str(path), "sha256": hash_file(path)})
        phases += 1
    return updates, int(phases == len(EXPECTED_OPERATIONS)), committed


def status(resource_pin, root, *, approved_resource_sha):
    """Derive live and terminal states from locks, failures and validated stage evidence."""
    if resource_pin["sha256"] != approved_resource_sha:
        raise ValueError("The progress scope must match the reviewed digest")
    resource = json.loads(read_pin(resource_pin, root))
    out = Path(resource["execution"])
    if resource["schema"] != "wp05_p4_independent_slot_scope_v1" or not out.is_relative_to(root) or out.resolve() != out:
        raise ValueError("Progress scope or output identity differs")
    if not out.exists():
        return None
    with observe_lock(out / ".coordinator.lock") as running:
        owner_path = out / "resource-owner.json"
        has_owner = owner_path.is_file()
        if has_owner and read(owner_path)["resource"] != resource_pin:
            raise ValueError("Controller output owner differs")
        if not has_owner and (out / "stages").exists() and any((out / "stages").iterdir()):
            raise ValueError("Unowned controller stage evidence")
        result = {"status": "running" if running else "interrupted_or_unknown",
                  "resource_sha256": resource_pin["sha256"], "slots": {}, "waiting": {}, "errors": {}, "stage_pins": []}
        science_sha = resource["science"]["sha256"]
        for slot, spec in resource["slots"].items():
            updates, audited = 0, 0
            for case in spec["cases"]:
                trained, accepted, pins = case_stages(out, case, resource_pin["sha256"], science_sha, slot)
                updates += trained
                audited += accepted
                result["stage_pins"].extend(pins)
            result["slots"][slot] = {"planned_cases": len(spec["cases"]), "planned_updates": spec["updates"],
                                     "accepted_cases": audited, "optimizer_updates": updates}
        failure = out / "failure.json"
        final = out / "status.json"
        if failure.is_file():
            result["errors"] = read(failure)
            if not result["errors"] or not set(result["errors"]) <= set(result["slots"]):
                raise ValueError("Failure refers to an unregistered slot")
            result["status"] = "draining_after_failure" if running else "failed"
        elif not running and final.is_file():
            saved = read(final)
            if saved["resource_sha256"] != resource_pin["sha256"]:
                raise ValueError("Status belongs to another registered run")
            slots = {key: {k: v for k, v in row.items() if k != "state"}
                     for key, row in saved.get("slots", {}).items()}
            if saved["status"] in ("completed", "waiting_for_resources") and slots == result["slots"]:
                result["status"] = saved["status"]
                result["waiting"] = saved.get("waiting", {})
        return result
