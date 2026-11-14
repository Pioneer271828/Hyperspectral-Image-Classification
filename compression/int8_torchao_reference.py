"""Opt-in affine arithmetic oracle for frozen WP10 integer point convolutions."""

from importlib import import_module, metadata, util
from pathlib import Path
import subprocess

import torch
from torch import nn

from compression.int8 import IntegerMatrix, _finite_cpu


TORCHAO_COMMIT = "9338966da58ec44b60f0e0b173cabab08f942ed0"
TORCHAO_VERSION = "0.15.0+git9338966"
TORCH_VERSION = "2.7.1+cu128"


class TorchAOAffine:
    """Attest Python-only TorchAO before using its quantize/dequantize primitives."""

    def __init__(self, source):
        root = Path(source).resolve(strict=True)

        def git(*args):
            return subprocess.check_output(["git", "-C", str(root), *args])

        try:
            revision = git("rev-parse", "HEAD").decode().strip()
            repository = Path(git("rev-parse", "--show-toplevel").decode().strip())
        except subprocess.CalledProcessError as exc:
            raise ValueError("Pinned compatible TorchAO source revision required") from exc
        if repository != root or revision != TORCHAO_COMMIT:
            raise ValueError("Pinned compatible TorchAO source revision required")
        if str(torch.__version__) != TORCH_VERSION:
            raise ValueError("Frozen PyTorch 2.7.1+cu128 required")
        spec = util.find_spec("torchao")
        if spec is None or spec.origin is None:
            raise RuntimeError("Verified TorchAO installation required")
        dist = metadata.distribution("torchao")
        package = Path(spec.origin).resolve().parent
        if (dist.version != TORCHAO_VERSION or package != Path(dist.locate_file("torchao")).resolve()
                or package == root / "torchao"):
            raise ValueError("Isolated installation of the pinned TorchAO version required")
        installed = {p.relative_to(package.parent) for p in package.rglob("*.py")}
        tracked = {Path(name) for name in git("ls-tree", "-rz", "--name-only", "HEAD", "--", "torchao")
                   .decode().split("\0") if name.endswith(".py")}
        if not installed or not installed <= tracked or any(package.rglob("*.so")):
            raise ValueError("Unexpected TorchAO Python or native module paths")
        # A wheel can exclude packages. Every installed module must match an immutable Git blob.
        for path in sorted(installed):
            if (package.parent / path).read_bytes() != git("show", f"HEAD:{path.as_posix()}"):
                raise ValueError(f"Installed TorchAO source mismatch: {path}")
        self.ops = import_module("torchao.quantization.quant_primitives")
        if Path(self.ops.__file__).resolve() != package / "quantization/quant_primitives.py":
            raise ValueError("TorchAO primitive origin differs")
        self.version = dist.version

    def quantize(self, value, scale, zero, cap):
        value = value.contiguous()
        return self.ops.quantize_affine(value, tuple(value.shape), scale, zero, torch.uint8, 0, cap)

    def dequantize(self, value, scale, zero):
        value = value.contiguous()
        return self.ops.dequantize_affine(value, tuple(value.shape), scale, zero, torch.uint8, 0, 255,
                                          output_dtype=torch.float32)


def frozen_parameters(module):
    if (type(module) is not IntegerMatrix or module.kind != "conv1x1"
            or type(module.kernel) is not nn.quantized.Conv2d or module.training
            or module.engine != torch.backends.quantized.engine):
        raise ValueError("Frozen native CPU point convolution required")
    kernel = module.kernel
    w = kernel.weight()
    if (w.qscheme() != torch.per_channel_affine or w.dtype != torch.qint8 or w.q_per_channel_axis() != 0
            or tuple(w.shape[1:]) != (module.in_features, 1, 1) or w.q_per_channel_zero_points().count_nonzero()
            or (w.int_repr() == -128).any()):
        raise ValueError("Frozen per-output symmetric weight required")
    return dict(weight=w.int_repr()[:, :, 0, 0].contiguous(), weight_scales=w.q_per_channel_scales(),
                bias=kernel.bias(), input_scale=module.input_scale.detach().clone(),
                input_zero=module.input_zero_point.detach().clone(),
                output_scale=torch.tensor([kernel.scale], dtype=torch.float32),
                output_zero=torch.tensor([kernel.zero_point], dtype=torch.int64))


@torch.no_grad()
def compare_layer(module, x, oracle, p=None):
    """Use frozen qparams. The actual packed operator remains the deployment path."""
    _finite_cpu(x)
    if x.ndim != 4 or x.shape[1] != module.in_features:
        raise ValueError("Matching NCHW input required")
    p = frozen_parameters(module) if p is None else p
    scale, zero = float(p["input_scale"].item()), int(p["input_zero"].item())
    bounded = x.clamp(-zero*scale, (127-zero)*scale)
    q_native = torch.quantize_per_tensor(bounded, scale, zero, torch.quint8)
    q_ao = oracle.quantize(bounded, p["input_scale"], p["input_zero"], 127)
    # Compare a 1x1 convolution with independent unrolled pixel dot products.
    shape = (x.shape[0], x.shape[2], x.shape[3], p["weight"].shape[0])
    act = q_ao.permute(0, 2, 3, 1).reshape(-1, x.shape[1]).to(torch.int64) - zero
    accum = act @ p["weight"].to(torch.int64).T
    scaled = accum.double() * (scale * p["weight_scales"])
    if p["bias"] is not None:
        scaled += p["bias"].detach().double()
    scaled = scaled.reshape(shape).permute(0, 3, 1, 2)
    q_ref = oracle.quantize(scaled.float(), p["output_scale"], p["output_zero"], 255)
    decoded = oracle.dequantize(q_ref, p["output_scale"], p["output_zero"])
    native = module.kernel(q_native)
    return q_native.int_repr(), q_ao, accum, native.int_repr(), q_ref, native.dequantize(), decoded
