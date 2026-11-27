"""One physical GPU lease location, independent of artifact directories."""
from contextlib import contextmanager
import os
from pathlib import Path
import re
import stat
import subprocess


DEFAULT_ROOT = "/home/lbx/hsi-pioneer271828/runs/.p4_gpu_leases"


def canonical_gpu_uuid(value):
    """Normalize the physical UUID formats emitted by CUDA and nvidia-smi."""
    match = re.fullmatch(r"(?:GPU-)?([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})",
                         value, flags=re.IGNORECASE)
    if match is None:
        raise ValueError("Explicit full physical GPU UUID required")
    return "GPU-" + match[1].lower()


def lease_path(gpu):
    if re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", gpu) is None:
        raise ValueError("Explicit full physical GPU UUID required")
    root = Path(os.environ.get("HSI_GPU_LEASE_ROOT", DEFAULT_ROOT))
    if not root.is_absolute() or not root.is_dir() or root != root.resolve(strict=True):
        raise ValueError("Existing absolute GPU lease directory without symlinks required")
    path = root / f"{gpu}.lock"
    if path.is_symlink():
        raise ValueError("Shared GPU lease file cannot be a symlink")
    return path


@contextmanager
def dispatch_lease(gpu):
    """Hold the shared lock in a coordinator that never executes CUDA."""
    import fcntl
    from transfer import p4_runtime_device as p4

    path = lease_path(gpu)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o660)
    with os.fdopen(fd, "r+b") as lock:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Shared GPU lease must be a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise p4.ResourceUnavailable(f"Active lease: {path}") from exc
        yield lock


@contextmanager
def physical_lease(gpu, inherited_fd=None):
    """Hold the shared lock or verify the caller's inherited worker lock."""
    from transfer import p4_runtime_device as p4

    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise ValueError("Require the one physical GPU UUID in CUDA_VISIBLE_DEVICES")
    if inherited_fd is None:
        with dispatch_lease(gpu) as lock:
            yield lock
        return
    path = lease_path(gpu)
    if inherited_fd < 3 or not stat.S_ISREG(os.fstat(inherited_fd).st_mode):
        raise ValueError("An inherited lease descriptor must refer to a regular file")
    p4.check_inherited_lease(inherited_fd, path)
    yield


def indexed_snapshot(index):
    """Read a physical-index inventory row for legacy integer-index CLI arguments."""
    if type(index) is not int or index < 0:
        raise ValueError("A nonnegative physical GPU index is required")
    result = subprocess.check_output(["nvidia-smi", "-i", str(index),
        "--query-gpu=index,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    actual, physical, free, util = (field.strip() for field in result.strip().split(","))
    if int(actual) != index or int(free) < 0 or not 0 <= int(util) <= 100:
        raise ValueError("Invalid GPU inventory for the requested physical index")
    return {"gpu": index, "physical_uuid": canonical_gpu_uuid(physical),
            "free_mib": int(free), "utilization": int(util)}


@contextmanager
def indexed_physical_lease(index, *, inventory=None):
    """Use the same physical lease for integer-index workers and UUID workers."""
    row = indexed_snapshot(index) if inventory is None else inventory
    if row["gpu"] != index or os.environ.get("CUDA_VISIBLE_DEVICES") != str(index):
        raise ValueError("Expose only the registered physical GPU index")
    with dispatch_lease(row["physical_uuid"]):
        observed_cuda_device(row["physical_uuid"], visibility=str(index))
        yield row


def observed_cuda_device(gpu, *, visibility=None):
    """Bind the actual CUDA logical device to the full UUID of a held lease."""
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != (gpu if visibility is None else visibility) or canonical_gpu_uuid(gpu) != gpu:
        raise ValueError("Expose exactly the leased physical CUDA device")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.current_device() != 0:
        raise ValueError("Expected exactly one visible logical CUDA device at cuda:0")
    props = torch.cuda.get_device_properties(0)
    physical_uuid = canonical_gpu_uuid(str(getattr(props, "uuid", "")))
    if physical_uuid != gpu:
        raise ValueError(f"Actual CUDA UUID {physical_uuid} does not match leased UUID {gpu}")
    return {"physical_uuid": physical_uuid, "device": "cuda:0", "visible_devices": visible,
            "device_count": 1, "gpu_name": props.name, "total_memory_bytes": props.total_memory,
            "cuda_runtime": torch.version.cuda}
