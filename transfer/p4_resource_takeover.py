"""Durable transfer of P4 queue ownership under the original coordinator lease."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from transfer import p4_runtime_device as device


SCHEMA = "wp05_p4_resource_owner_taken_over_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def create(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path.parent)


def payload(row):
    return (json.dumps(row, indent=2, sort_keys=True) + "\n").encode("utf-8")


def untouched(execution, identifiers):
    require(not (execution / "failure.json").exists(), "Original failure or unconfirmed operations require review")
    for case_id in identifiers:
        for folder in ("cases", "stages"):
            path = execution / folder / case_id
            require(not path.exists() and not path.is_symlink(), f"Original case has state or a dispatch: {case_id}")
        require(not list((execution / "logs").glob(f"{case_id}-*")), f"Original case has a worker launch: {case_id}")


def files(record):
    directory = Path(record["control"])
    execution = Path(record["parent_execution"])
    require(directory == directory.resolve() and execution == execution.resolve() and directory.parent == execution.parent
            and directory != execution, "Separate immutable-origin and control directories required")
    return directory, execution, directory / "original-resource-owner.bytes", directory / "takeover-intent.json"


def verify(record, lease):
    directory, execution, archive, intent = files(record)
    device.check_inherited_lease(lease.fileno(), execution / ".coordinator.lock")
    require(archive.is_file() and digest(archive.read_bytes()) == record["prior_owner_sha256"],
            "Archived original owner differs")
    require(json.loads(intent.read_bytes()) == record, "Takeover intent differs")
    owner = execution / "resource-owner.json"
    require(not owner.is_symlink() and owner.read_bytes() == payload(record),
            "Original queue custody has not been transferred or has changed")
    return directory


def takeover(record, lease, validate):
    directory, execution, archive, intent = files(record)
    device.check_inherited_lease(lease.fileno(), execution / ".coordinator.lock")
    require(record["schema"] == SCHEMA and not intent.exists() and not archive.exists(),
            "Duplicate takeover or isolated intent requires independent review")
    owner = execution / "resource-owner.json"
    before = owner.read_bytes()
    require(not owner.is_symlink() and digest(before) == record["prior_owner_sha256"],
            "Original owner differs from the approved takeover")
    untouched(execution, record["cases"])
    validate()
    directory.mkdir(exist_ok=True)
    create(archive, before)
    create(intent, payload(record))
    # A crash before rename leaves the legacy owner and an isolated intent.
    # After rename, the owner itself is the complete, durable custody record.
    with NamedTemporaryFile(dir=execution, prefix=".resource-owner-takeover-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload(record))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        require(owner.read_bytes() == before, "Original owner changed before custody transfer")
        os.replace(temporary, owner)
        sync_directory(execution)
    finally:
        temporary.unlink(missing_ok=True)
    verify(record, lease)
