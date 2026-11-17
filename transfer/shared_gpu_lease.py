"""One physical GPU lease location, independent of artifact directories."""
from contextlib import contextmanager
import os
from pathlib import Path
import re
import stat


DEFAULT_ROOT = "/home/lbx/hsi-pioneer271828/runs/.p4_gpu_leases"


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
def physical_lease(gpu, inherited_fd=None):
    """Hold the shared lock or verify that the caller passed its open descriptor."""
    import fcntl
    from transfer import p4_runtime_device as p4

    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise ValueError("Require the one physical GPU UUID in CUDA_VISIBLE_DEVICES")
    path = lease_path(gpu)
    if inherited_fd is not None:
        if inherited_fd < 3 or not stat.S_ISREG(os.fstat(inherited_fd).st_mode):
            raise ValueError("An inherited lease descriptor must refer to a regular file")
        p4.check_inherited_lease(inherited_fd, path)
        yield
        return

    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o660)
    with os.fdopen(fd, "r+b") as lock:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Shared GPU lease must be a regular file")
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise p4.ResourceUnavailable(f"Active lease: {path}") from exc
        yield lock
