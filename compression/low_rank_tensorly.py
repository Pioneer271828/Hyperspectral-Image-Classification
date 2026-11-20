"""Optional TensorLy SVD reference for the frozen WP10 point convolution."""

from importlib import import_module, util
from pathlib import Path
import subprocess

import torch

from compression.low_rank import factorize_model, matrix
from compression.low_rank_recovery import TARGET


TENSORLY_COMMIT = "7401f09e05812a0c4c3d673bbaf8419acd7b33b7"
TENSORLY_VERSION = "0.9.0"
RANKS = (8, 16, 32, 64)


class TensorLySVD:
    """Use the explicitly supplied upstream revision to attest installed sources."""

    def __init__(self, source):
        root = Path(source).resolve(strict=True)

        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args])

        if (Path(git("rev-parse", "--show-toplevel").decode().strip()) != root
                or git("rev-parse", "HEAD").decode().strip() != TENSORLY_COMMIT):
            raise ValueError("Pinned TensorLy source revision required")
        spec = util.find_spec("tensorly")
        if spec is None or spec.origin is None:
            raise RuntimeError("Pinned TensorLy installation required")
        package = Path(spec.origin).resolve().parent
        if (package.name != "tensorly" or package == root / "tensorly"
                or spec.submodule_search_locations != [str(package)]):
            raise ValueError("An isolated TensorLy installation is required")
        tracked = {
            Path(path)
            for path in git("ls-tree", "-rz", "--name-only", "HEAD", "--", "tensorly").decode().split("\0")
            if path.endswith(".py")
        }
        installed = {path.relative_to(package.parent) for path in package.rglob("*.py")}
        if installed != tracked:
            raise ValueError("Installed TensorLy module paths differ from the pinned revision")
        # Use immutable Git blobs to check the wheel. The checkout may contain build products.
        for path in sorted(tracked):
            if (package.parent / path).read_bytes() != git("show", f"HEAD:{path.as_posix()}"):
                raise ValueError(f"Installed TensorLy source mismatch: {path}")
        self.tensorly = import_module("tensorly")
        self.svd_module = import_module("tensorly.tenalg.svd")
        if (self.tensorly.__version__ != TENSORLY_VERSION
                or Path(self.tensorly.__file__).resolve() != package / "__init__.py"
                or Path(self.svd_module.__file__).resolve() != package / "tenalg/svd.py"):
            raise ValueError("TensorLy module origin or version differs from its verified sources")

    def decompose(self, layer, rank):
        kind, weight = matrix(layer)
        if kind != "conv1x1" or tuple(weight.shape) != (128, 64):
            raise ValueError("The WP10 128 by 64 point convolution is required")
        if type(rank) is not int or rank not in RANKS:
            raise ValueError("A registered WP10 rank is required")
        with self.tensorly.backend_context("numpy", local_threadsafe=True):
            u, s, vh = self.svd_module.truncated_svd(weight.cpu().double().numpy(), n_eigenvecs=rank)
        return tuple(torch.from_numpy(value.copy()) for value in (u, s, vh))

    def initialize(self, dense, rank, *, seed=64):
        u, s, vh = self.decompose(dense.get_submodule(TARGET), rank)
        root = s.sqrt()
        model = factorize_model(dense, {TARGET: rank}, init="random", seed=seed,
                                allow_expansion=rank == 64)
        target = model.get_submodule(TARGET)
        with torch.no_grad():
            target.left.copy_(u * root)
            target.right.copy_(root[:, None] * vh)
        if not all(torch.isfinite(parameter).all() for parameter in target.parameters()):
            raise ValueError("TensorLy factor conversion produced nonfinite parameters")
        model._low_rank_plan["init"] = "svd"
        return model
