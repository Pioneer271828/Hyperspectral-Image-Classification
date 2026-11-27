"""Process-local, verified CPU observation reuse for opt-in engineering."""
import stat

from .wp12_frozen54_deployment_contract import source
from .wp12_observation_snapshot import load_snapshot, verified_snapshot

CACHE_BYTES = 600 * 1024 * 1024
FIELDS = {"net_input", "Q", "A", "wavelengths", "fwhm", "superpixel_scale"}


def storage_bytes(item):
    import torch

    sizes = {}
    for tensor in item.values():
        if not torch.is_tensor(tensor):
            continue
        if tensor.layout == torch.sparse_coo:
            parts = (tensor._indices(), tensor._values())
        elif tensor.layout == torch.strided:
            parts = (tensor,)
        else:
            raise ValueError("Cache supports dense tensors and sparse COO only")
        for part in parts:
            storage = part.untyped_storage()
            sizes[storage.data_ptr()] = storage.nbytes()
    return sum(sizes.values())


def copy_observation(item):
    import torch

    return {key: value.detach().clone() if torch.is_tensor(value) else value for key, value in item.items()}


class DecodedObservationCache:
    """Retain one pinned observation and lend independent CPU copies per request."""

    def __init__(self, root, pin, validator, *, max_bytes=CACHE_BYTES):
        root = root.resolve(strict=True)
        self._path = source(root, pin)
        self._root = root
        self._pin = dict(pin)
        self._validator = validator
        self._limit = max_bytes
        self._stamp = self._file_stamp()
        self._item = None
        self.retained_bytes = 0

    def _file_stamp(self):
        if source(self._root, self._pin) != self._path:
            raise ValueError("Observation source path changed")
        value = self._path.stat(follow_symlinks=False)
        if not stat.S_ISREG(value.st_mode) or not 0 < value.st_size <= self._limit:
            raise ValueError("Expected a regular observation within the cache byte limit")
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def check_source(self):
        if self._file_stamp() != self._stamp:
            raise ValueError("Pinned observation changed during the cache session")

    def _opened_item(self):
        import torch

        with verified_snapshot(self._root, self._pin, max_bytes=self._limit) as path:
            item = load_snapshot(path)
            if (not isinstance(item, dict) or set(item) != FIELDS
                    or type(item["superpixel_scale"]) is not int
                    or any(not torch.is_tensor(value) or value.device.type != "cpu"
                           for key, value in item.items()
                           if key != "superpixel_scale" and not (key == "fwhm" and value is None))):
                raise ValueError("Only label-free observation fields and CPU tensors may be cached")
            self._validator(item)
            if storage_bytes(item) > self._limit:
                raise ValueError("Decoded observation exceeds the cache byte limit")
            owned = copy_observation(item)
        return owned

    def checkout(self):
        self.check_source()
        if self._item is None:
            item = self._opened_item()
            retained_bytes = storage_bytes(item)
            if retained_bytes > self._limit:
                raise ValueError("Owned observation exceeds the cache byte limit")
            self.check_source()
            self._item = item
            self.retained_bytes = retained_bytes
        request = copy_observation(self._item)
        self.check_source()
        return request
