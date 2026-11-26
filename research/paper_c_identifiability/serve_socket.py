"""Serve exactly one pinned opaque archive over a private Unix socket."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
from time import perf_counter


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('archive','socket','sha256'):p.add_argument('--'+n,required=True)
    p.add_argument('--bytes',type=int,required=True);p.add_argument('--offset',type=int,required=True);a=p.parse_args()
    source=Path(a.archive).resolve();endpoint=Path(a.socket).absolute();before=source.stat();start=perf_counter()
    if endpoint.exists() or not 0<=a.offset<a.bytes or before.st_size!=a.bytes:raise ValueError('New socket and exact source/offset required')
    if endpoint.parent.stat().st_mode&0o077:raise ValueError('Socket parent must be private to the authorized account')
    if len(a.sha256)!=64 or sha(source)!=a.sha256:raise ValueError('Original archive SHA differs')
    os.umask(0o077)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as listener:
        listener.bind(str(endpoint));os.chmod(endpoint,0o600);listener.listen(1);listener.settimeout(900)
        print(json.dumps({'status':'READY_PRIVATE_SOCKET_STREAM','archive':source.name,'bytes':a.bytes,'offset':a.offset}),flush=True)
        client,_=listener.accept()
        with client,source.open('rb') as f:
            client.settimeout(120);f.seek(a.offset);remaining=a.bytes-a.offset
            while remaining:
                raw=f.read(min(1024**2,remaining))
                if not raw:raise EOFError('Source archive truncated')
                client.sendall(raw);remaining-=len(raw)
    after=source.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('Source stat changed')
    print(json.dumps({'status':'SENT_PINNED_ARCHIVE_BYTES','bytes':a.bytes-a.offset,'elapsed_seconds':perf_counter()-start,
                      'source_stat_unchanged':True,'target_whole_hash_verification':'SEPARATE_RECEIPT_REQUIRED','tensor_decodes':0}),flush=True)


if __name__=='__main__':main()
