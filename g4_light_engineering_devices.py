"""Shared physical device leases for synthetic light engineering."""
import fcntl
import os
from pathlib import Path
import stat
import subprocess

from g4_light_engineering_registration import GPU_UUID

LEASE_ROOT = "/home/lbx/hsi-pioneer271828/runs/.p4_gpu_leases"
LEASE_FD = "G4_LIGHT_SHARED_LEASE_FD"


def path_for(uuid):
    root = Path(os.environ.get("HSI_GPU_LEASE_ROOT", LEASE_ROOT))
    if not GPU_UUID.fullmatch(uuid) or not root.is_dir() or root.resolve() != root:
        raise ValueError("Canonical shared physical GPU lease required")
    return root / (uuid + ".lock")


def open_lease(uuid):
    flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK
    return os.fdopen(os.open(path_for(uuid), flags, 0o660), "r+b")


def inventory(slot, uuid):
    result = subprocess.check_output(
        ["nvidia-smi", "-i", str(slot), "--query-gpu=uuid,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    identity, free, util = (v.strip() for v in result.strip().split(","))
    if identity != uuid:
        raise ValueError("Registered GPU index no longer denotes the physical UUID")
    return int(free), int(util)


def device_admitted(slot, uuid):
    free, util = inventory(slot, uuid)
    return free >= 12288 and util <= 60


def check_child(registration, slot):
    uuid = next(row.get("uuid") for row in registration["devices"] if row["slot"] == slot)
    if uuid is None:
        return
    try:
        fd = int(os.environ[LEASE_FD])
        actual = os.fstat(fd)
        expected = path_for(uuid).stat(follow_symlinks=False)
        if not stat.S_ISREG(actual.st_mode) or (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):
            raise ValueError("Wrong shared device lease")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError("Synthetic GPU worker needs its inherited shared lease") from exc
    inventory(slot,uuid)
