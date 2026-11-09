"""Canonical 2-bit weight-only storage. Decoding gives FP32, not a fast kernel.

Layout: TQW1, uint8 rank, rank uint32 LE dimensions, two FP32 LE magnitudes,
then four low-bit-first codes per byte: 0=zero, 1=positive, 2=negative.
Code 3 and nonzero padding are rejected. Bias/architecture are not in this file.
"""
import hashlib
import math
import os
from pathlib import Path
import struct
import tempfile

import numpy as np
import torch

from compression.ternary import assignments, scales


def encode(weight, positive, negative, ratio):
    scales(positive, negative)
    pos, neg = assignments(weight, ratio)
    shape = tuple(weight.shape)
    if len(shape) not in (2, 4) or any(n <= 0 or n >= 2**32 for n in shape):
        raise ValueError('matrix or 1x1 convolution weight shape required')
    if len(shape) == 4 and shape[2:] != (1, 1): raise ValueError('only 1x1 convolution')
    codes = (pos.to(torch.uint8) + 2 * neg.to(torch.uint8)).flatten().numpy()
    padded = np.pad(codes, (0, (-len(codes)) % 4)).reshape(-1, 4)
    packed = sum(padded[:, i] << (2*i) for i in range(4)).astype(np.uint8).tobytes()
    return (b'TQW1' + struct.pack('<B', len(shape)) + struct.pack('<'+'I'*len(shape), *shape)
            + struct.pack('<ff', float(positive.detach()), float(negative.detach())) + packed)


def decode(payload):
    if type(payload) is not bytes or len(payload) < 5 or payload[:4] != b'TQW1':
        raise ValueError('not a TQW1 weight payload')
    rank = payload[4]
    if rank not in (2, 4): raise ValueError('invalid rank')
    offset = 5 + 4*rank + 8
    if len(payload) < offset: raise ValueError('truncated header')
    shape = struct.unpack_from('<'+'I'*rank, payload, 5)
    if not all(shape) or (rank == 4 and shape[2:] != (1, 1)): raise ValueError('invalid shape')
    count = math.prod(shape)
    if len(payload) != offset + (count+3)//4: raise ValueError('payload length mismatch')
    p, n = struct.unpack_from('<ff', payload, offset-8)
    scales(torch.tensor(p), torch.tensor(n))
    packed = np.frombuffer(payload, dtype=np.uint8, offset=offset)
    codes = np.stack([(packed >> (2*i)) & 3 for i in range(4)], axis=1).reshape(-1)
    if (codes[:count] == 3).any() or codes[count:].any(): raise ValueError('invalid code or padding')
    values = np.array([0, p, -n], dtype=np.float32)[codes[:count]]
    return torch.from_numpy(values.copy().reshape(shape))


def save_weights(path, payload):
    decode(payload)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temp = Path(handle.name); handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.link(temp, path)
    finally:
        if temp is not None: temp.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def load_weights(path, expected_sha256):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256: raise ValueError('TQ weight hash mismatch')
    return decode(payload)
