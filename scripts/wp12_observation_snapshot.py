"""Read a verified private ZIP snapshot for opt-in WP12 engineering inference."""

from contextlib import contextmanager
import fcntl
from hashlib import sha256
import mmap
import os
from pathlib import Path
import stat

from .wp12_frozen54_deployment_contract import source


@contextmanager
def verified_snapshot(root, pin, *, max_bytes=None):
    """Yield a path to the sealed bytes that passed the original file digest."""
    path = source(root, pin)
    snapshot = os.memfd_create("wp12-observation", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
    try:
        original = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            status = os.fstat(original)
            if (not stat.S_ISREG(status.st_mode) or status.st_size <= 0
                    or Path(f"/proc/self/fd/{original}").readlink() != path):
                raise ValueError("Expected the pinned regular observation file")
            if max_bytes is not None and status.st_size > max_bytes:
                raise ValueError("Observation exceeds the registered snapshot size ceiling")
            offset = 0
            while offset < status.st_size:
                copied = os.sendfile(snapshot, original, offset, status.st_size - offset)
                if not copied:
                    raise ValueError("Observation was truncated while copying")
                offset += copied
            if os.fstat(original).st_size != status.st_size:
                raise ValueError("Observation size changed while copying")
        finally:
            os.close(original)
        fcntl.fcntl(snapshot, fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL)
        with mmap.mmap(snapshot, status.st_size, access=mmap.ACCESS_READ) as content:
            if sha256(content).hexdigest() != pin["sha256"]:
                raise ValueError("Pinned artifact content differs")
        yield Path(f"/proc/self/fd/{snapshot}")
    finally:
        os.close(snapshot)


def load_snapshot(path):
    """Use PyTorch's ZIP loader with CPU storage mapped from the verified snapshot."""
    import torch

    if torch.serialization.get_default_mmap_options() != mmap.MAP_PRIVATE:
        raise ValueError("Verified observation snapshots require a private PyTorch mapping")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
