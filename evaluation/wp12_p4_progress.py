"""Read P4 controller evidence without taking a worker lease or starting work."""
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


def active(path):
    try:
        stream = path.open("rb")
    except FileNotFoundError:
        return False
    with stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def status(resource_pin, root, *, approved_resource_sha):
    """Report committed partial work without inferring an accepted scientific batch."""
    if resource_pin["sha256"] != approved_resource_sha:
        raise ValueError("The progress scope must match the reviewed digest")
    resource = json.loads(read_pin(resource_pin, root))
    out = Path(resource["execution"])
    if resource["schema"] != "wp05_p4_independent_slot_scope_v1" or not out.is_relative_to(root) or out.resolve() != out:
        raise ValueError("Progress scope or output identity differs")
    if not out.exists():
        return None
    lock = out / ".coordinator.lock"
    running = active(lock)
    owner_file = out / "resource-owner.json"
    if running and not owner_file.exists():
        return {"status": "running", "slots": {
            name: {"planned_cases": len(row["cases"]), "planned_updates": row["updates"],
                   "optimizer_updates": 0, "accepted_cases": 0} for name, row in resource["slots"].items()}}
    owner = read(owner_file)
    if owner["resource"] != resource_pin:
        raise ValueError("Controller output owner differs")
    if not running:
        if (out / "status.json").is_file():
            saved = read(out / "status.json")
            if saved["resource_sha256"] != resource_pin["sha256"]:
                raise ValueError("Status belongs to another registered run")
            return saved
        return None
    result = {"status": "running", "slots": {}}
    for slot, spec in resource["slots"].items():
        updates, audited = 0, 0
        for case in spec["cases"]:
            if not re.fullmatch(r"s[0-9]+-k[0-9]+-[a-z_]+-m[0-9]+-[a-z_]+", case):
                raise ValueError("Invalid frozen case name")
            folder = out / "stages" / case
            # A phase that has no acknowledgment yet is still in flight.
            for i, (name, operation, start, until) in enumerate(EXPECTED_OPERATIONS):
                path = folder / f"{i:02d}.receipt.json"
                if not path.is_file():
                    break
                saved = read(path)
                proof = Path(saved["evidence"]["path"])
                expected = out / "cases" / case
                if name is not None:
                    expected /= name
                if operation == "advance":
                    expected /= f"calls/{start:06d}-{until:06d}.json"
                else:
                    expected /= "audit.json" if operation == "audit" else "completed-resume.json"
                if proof != expected or proof.resolve() != proof or hash_file(proof) != saved["evidence"]["sha256"]:
                    raise ValueError("Partial progress stage fingerprint differs")
                if operation == "advance":
                    updates += until - start
                elif operation == "audit":
                    audited += 1
        result["slots"][slot] = {"planned_cases": len(spec["cases"]), "planned_updates": spec["updates"],
                                 "accepted_cases": audited, "optimizer_updates": updates}
    return result
