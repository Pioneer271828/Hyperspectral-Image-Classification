"""Optional v2 exact-byte checkpoint CAS. No historical writer is modified.

Derived from the project's whole-file SHA CAS and fsync/rename transaction
patterns. Compression is reversible bytes; no tensor or pickle rewriting.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import uuid
import zipfile
import zlib

import numpy as np

CHUNK = 65536
MAX_FILE = 64*1024**2
MAX_DEPTH = 2
NAME = re.compile(r'[A-Za-z0-9_-]{1,100}\Z')


def require(ok, message):
    if not ok: raise ValueError(message)


def hashed(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def create(path, raw):
    with Path(path).open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    fsync_dir(Path(path).parent)


def xor(a,b):
    require(len(a)==len(b),'XOR operands differ')
    return np.bitwise_xor(np.frombuffer(a,dtype=np.uint8),np.frombuffer(b,dtype=np.uint8)).tobytes()


def segments(raw, name, layout):
    """Cover every original byte, including ZIP padding/descriptors/directory."""
    require(0<len(raw)<=MAX_FILE,'File size outside registered codec limit')
    spans=[]
    if layout=='zip-xor' and name.endswith('.pt'):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            infos=sorted(z.infolist(),key=lambda i:i.header_offset)
            require(len({i.filename for i in infos})==len(infos),'Duplicate ZIP members')
            cursor=0
            for i in infos:
                h=i.header_offset
                require(i.compress_type==zipfile.ZIP_STORED and h>=cursor and raw[h:h+4]==b'PK\x03\x04','Unsupported ZIP storage/header')
                nl,el=struct.unpack_from('<HH',raw,h+26);start=h+30+nl+el;end=start+i.compress_size
                require(cursor<=start<=end<=len(raw) and i.compress_size==i.file_size,'ZIP member escapes file')
                if start>cursor:spans.append((f'gap:{i.filename}',cursor,start))
                if end>start:spans.append((f'data:{i.filename}',start,end))
                cursor=end
            if cursor<len(raw):spans.append(('tail',cursor,len(raw)))
    else:
        spans=[('raw',0,len(raw))]
    output=[]
    for slot,start,end in spans:
        for j,offset in enumerate(range(start,end,CHUNK)):
            output.append((f'{slot}:{j}',offset,raw[offset:min(offset+CHUNK,end)]))
    require(sum(len(x[2]) for x in output)==len(raw),'Segment coverage differs')
    return output


