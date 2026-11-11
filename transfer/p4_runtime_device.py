"""Explicit CPU/CUDA arithmetic, resource admission and worker leases for P4."""
from contextlib import contextmanager
import os
import platform
import re
import subprocess

import torch


CPU = 'cpu_fp32_v1'
CUDA = 'cuda_deterministic_fp32_v1'
WORKSPACE = ':4096:8'
UUID = re.compile(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}')


class ResourceUnavailable(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def slots(protocol, gpus):
    require(protocol in (CPU, CUDA), 'Unknown numerical protocol')
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
    return row['free_mib'] >= 12288 and row['utilization'] <= 60


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


@contextmanager
def arithmetic(protocol, gpu=None):
    """Apply a versioned FP32 protocol and restore process settings on exit."""
    require(protocol in (CPU, CUDA) and (gpu is None) == (protocol == CPU), 'Device/protocol mismatch')
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
        details.update(driver=row['driver'], cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                       gpu_name=props.name, compute_capability=[props.major, props.minor])
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
        yield details
    finally:
        torch.use_deterministic_algorithms(old_determinism[0], warn_only=old_determinism[1])
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic) = old_backend
        torch.set_float32_matmul_precision(old_precision)
        torch.set_num_threads(old_threads)
