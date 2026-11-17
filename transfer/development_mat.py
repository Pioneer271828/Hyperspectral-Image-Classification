"""Bounded reads from the uncompressed, previously audited guarded MAT v5 copy."""

import os
import struct

import numpy as np

from research_runtime import hash_file
from transfer.development_blocks import inside


def _header(stream, size, variable, shape):
    header = stream.read(128)
    if len(header) != 128 or header[124:] != b"\x00\x01IM":
        raise ValueError("Expected little-endian MAT v5")
    kind, matrix_bytes = struct.unpack("<II", stream.read(8))
    if kind != 14 or 136 + matrix_bytes != size:
        raise ValueError("Expected exactly one uncompressed guarded MAT variable")

    def element(expected, count=None):
        tag = stream.read(8)
        kind, length = struct.unpack("<II", tag)
        if kind != expected or length > 128 or (count is not None and length != count):
            raise ValueError("Unexpected MAT variable metadata")
        value = stream.read(length)
        stream.seek((-length) % 8, os.SEEK_CUR)
        return value

    flags, reserved = struct.unpack("<II", element(6, 8))
    dimensions = struct.unpack("<iii", element(5, 12))
    name = element(1).decode("ascii")
    kind, length = struct.unpack("<II", stream.read(8))
    if (flags != 7 or reserved != 0 or dimensions != shape or name != variable or kind != 7
            or length != 4 * int(np.prod(shape)) or stream.tell() + length + (-length) % 8 != size):
        raise ValueError("MAT name, dimensions, dtype or data boundary differs")
    return stream.tell()


def open_guarded(pin, root, variable, shape, boxes):
    """Authenticate bytes without decoding the scene, then read only released column spans."""
    path = inside(pin["path"], root)
    if hash_file(path) != pin["sha256"]:
        raise ValueError("Guarded source file hash differs")
    with path.open("rb") as stream:
        offset = _header(stream, os.fstat(stream.fileno()).st_size, variable, shape)
    height, width, bands = shape
    allowed = {row["id"]: row for row in boxes}

    def read(row):
        if allowed.get(row["id"]) != row:
            raise ValueError("Guarded payload request escapes the admitted roles")
        y0, y1, x0, x1 = row["box"]
        rows = y1 - y0
        value = np.empty((rows, x1 - x0, bands), dtype=np.float32)
        with path.open("rb") as stream:
            for band in range(bands):
                for column in range(x0, x1):
                    stream.seek(offset + 4 * (y0 + height * (column + width * band)))
                    data = stream.read(rows * 4)
                    if len(data) != rows * 4:
                        raise ValueError("Truncated guarded observation")
                    value[:, column - x0, band] = np.frombuffer(data, dtype="<f4")
        return value

    return read
