"""Explicit CPU/CUDA arithmetic, resource admission and worker leases for P4."""
from contextlib import contextmanager, nullcontext
import os
import platform
import re
import subprocess

import torch
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode


CPU = 'cpu_fp32_v1'
CUDA = 'cuda_deterministic_fp32_v1'
CUDA_AUDITED = 'cuda_deterministic_fp32_named_adam_gemm_v3'
CUDA_PROTOCOLS = (CUDA, CUDA_AUDITED)
WORKSPACE = ':4096:8'
UUID = re.compile(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}')
MIN_FREE_MIB, MAX_UTILIZATION = 12288, 60


class ResourceUnavailable(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def slots(protocol, gpus):
    require(protocol in (CPU, *CUDA_PROTOCOLS), 'Unknown numerical protocol')
    if protocol == CPU:
        require(not gpus, 'CPU registration has no physical GPU assignment')
        return [None]
    require(len(gpus) in (2, 3) and len(set(gpus)) == len(gpus)
            and all(UUID.fullmatch(gpu) for gpu in gpus), 'Register two or three distinct physical GPU UUIDs')
    return list(gpus)


def snapshot(gpu):
    require(bool(UUID.fullmatch(gpu)), 'Physical GPU UUID required')
    result = subprocess.check_output(['nvidia-smi', '-i', gpu,
        '--query-gpu=uuid,memory.free,utilization.gpu,driver_version', '--format=csv,noheader,nounits'],
        text=True).strip().split(',')
    identity, free, util, driver = (value.strip() for value in result)
    require(identity == gpu, 'Resource inventory returned a different GPU UUID')
    return {'physical_uuid': identity, 'free_mib': int(free), 'utilization': int(util), 'driver': driver}


def eligible(row):
    return row['free_mib'] >= MIN_FREE_MIB and row['utilization'] <= MAX_UTILIZATION


def simultaneous_inventory(gpus):
    result = subprocess.check_output(['nvidia-smi',
        '--query-gpu=uuid,memory.free,utilization.gpu,driver_version', '--format=csv,noheader,nounits'],
        text=True)
    rows = {}
    for line in result.splitlines():
        gpu, free, utilization, driver = (value.strip() for value in line.split(','))
        if gpu in gpus:
            rows[gpu] = {'physical_uuid': gpu, 'free_mib': int(free), 'utilization': int(utilization), 'driver': driver}
    require(set(rows) == set(gpus), 'Registered physical GPU missing from simultaneous inventory')
    return rows


def gate(gpu):
    row = snapshot(gpu)
    if not eligible(row):
        raise ResourceUnavailable(f'{gpu} has {row["free_mib"]} MiB free and {row["utilization"]}% utilization')
    return row


@contextmanager
def lease(path):
    """OS releases this lock on exit. Pass the descriptor to a live worker."""
    import fcntl

    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ResourceUnavailable(f'Active lease: {path}') from error
        yield stream


def check_inherited_lease(fd, path):
    import fcntl

    require(os.path.samestat(os.fstat(fd), path.stat()), f'Worker lease does not match {path}')
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise ResourceUnavailable(f'Worker did not inherit the active lease: {path}') from error


def _bin_weights(value, size, count):
    cells = torch.arange(count, device=value.device)
    indices = torch.arange(size, device=value.device)
    start = cells * size // count
    end = ((cells + 1) * size + count - 1) // count
    inside = (indices >= start[:, None]) & (indices < end[:, None])
    return inside.to(value.dtype) / (end - start)[:, None]


def adaptive_bin_mean(value, shape):
    """Average the same possibly overlapping bins as adaptive_avg_pool2d."""
    vertical = _bin_weights(value, value.shape[-2], shape[0])
    horizontal = _bin_weights(value, value.shape[-1], shape[1])
    return (vertical @ value) @ horizontal.T


def sparse_row_product(left, right):
    """Multiply a COO graph by dense features using deterministic index_add."""
    graph = left.coalesce()
    rows, columns = graph.indices()
    terms = graph.values()[:, None] * right[columns]
    return right.new_zeros((left.shape[0], right.shape[1])).index_add(0, rows, terms)


class _SparseRowProductGemm(torch.autograd.Function):
    """Use a deterministic FP32 matrix product for the COO weight gradient."""

    @staticmethod
    def forward(ctx, weights, features, rows, columns, height):
        ctx.save_for_backward(weights, features, rows, columns)
        return features.new_zeros((height, features.shape[1])).index_add(
            0, rows, weights[:, None] * features[columns])

    @staticmethod
    def backward(ctx, sensitivity):
        weights, features, rows, columns = ctx.saved_tensors
        # Gather only the original edges from dY @ X^T, including transposed graphs.
        weight_grad = (sensitivity @ features.T)[rows, columns] if ctx.needs_input_grad[0] else None
        feature_grad = (features.new_zeros(features.shape).index_add(0, columns, weights[:, None] * sensitivity[rows])
                        if ctx.needs_input_grad[1] else None)
        return weight_grad, feature_grad, None, None, None


def sparse_row_product_gemm(left, right):
    graph = left.coalesce()
    rows, columns = graph.indices()
    return _SparseRowProductGemm.apply(graph.values(), right, rows, columns, graph.shape[0])


class _CudaGraphKernels(TorchFunctionMode):
    """Use the registered FP32 graph operators only inside a P4 CUDA worker."""

    def __init__(self, sparse_product, calls=None):
        super().__init__()
        self.sparse_product = sparse_product
        self.calls = calls

    def __torch_function__(self, func, types, args=(), kwargs=None):
        options = kwargs or {}
        if len(args) == 2 and not options:
            if func is F.adaptive_avg_pool2d and args[0].is_cuda:
                if self.calls is not None:
                    self.calls['adaptive_avg_pool2d'] += 1
                size = args[1]
                return adaptive_bin_mean(args[0], (size, size) if isinstance(size, int) else size)
            if func is torch.sparse.mm and args[0].is_cuda and args[0].is_sparse:
                if self.calls is not None:
                    self.calls['sparse_mm'] += 1
                return self.sparse_product(*args)
        return func(*args, **options)


@contextmanager
def arithmetic(protocol, gpu=None, *, kernel_calls=None):
    """Apply a versioned FP32 protocol and restore process settings on exit."""
    require(protocol in (CPU, *CUDA_PROTOCOLS) and (gpu is None) == (protocol == CPU), 'Device/protocol mismatch')
    details = {'protocol': protocol, 'physical_uuid': gpu, 'device': 'cpu' if gpu is None else 'cuda:0',
               'torch': str(torch.__version__), 'torch_threads': 1,
               'machine': platform.machine(), 'host': platform.node()}
    if gpu is not None:
        # Gate before creating a CUDA context and require the coordinator's isolation.
        row = gate(gpu)
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == gpu and
                os.environ.get('CUBLAS_WORKSPACE_CONFIG') == WORKSPACE,
                'Expose only the assigned UUID with the registered cuBLAS workspace')
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
                'Expected one visible logical CUDA device')
        torch.cuda.set_device(0)
        props = torch.cuda.get_device_properties(0)
        graph_kernels = ('adaptive_bin_mean_and_sparse_row_product_weight_gemm_v3' if protocol == CUDA_AUDITED
                         else 'adaptive_bin_mean_and_sparse_row_product_v1')
        details.update(driver=row['driver'], cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       gpu_name=props.name, compute_capability=[props.major, props.minor],
                       graph_kernels=graph_kernels)
    old_threads = torch.get_num_threads()
    old_precision = torch.get_float32_matmul_precision()
    old_determinism = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    old_backend = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                   torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(gpu is not None, warn_only=False)
        torch.set_float32_matmul_precision('highest')
        if gpu is not None:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        details['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
        details['deterministic_warn_only'] = torch.is_deterministic_algorithms_warn_only_enabled()
        details['float32_matmul_precision'] = torch.get_float32_matmul_precision()
        sparse_product = sparse_row_product_gemm if protocol == CUDA_AUDITED else sparse_row_product
        with (_CudaGraphKernels(sparse_product, kernel_calls) if gpu is not None else nullcontext()):
            yield details
    finally:
        torch.use_deterministic_algorithms(old_determinism[0], warn_only=old_determinism[1])
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = old_backend
        torch.set_float32_matmul_precision(old_precision)
        torch.set_num_threads(old_threads)
