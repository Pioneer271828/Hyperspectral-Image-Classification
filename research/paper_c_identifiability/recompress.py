"""Losslessly re-encode a sealed gzip tar with a long dictionary; no tensor decode."""
import argparse
import gzip
import hashlib
import json
import lzma
import os
from pathlib import Path
from time import perf_counter


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def encode(source,out,expected):
    source=Path(source).resolve();out=Path(out).resolve();pending=out.with_suffix(out.suffix+'.incoming')
    if out.exists() or pending.exists() or out==source:raise ValueError('New output required; retain prior files')
    start=perf_counter();before=source.stat()
    if sha(source)!=expected:raise ValueError('Original compressed archive changed')
    h=hashlib.sha256();total=0;reported=0
    filters=[{'id':lzma.FILTER_LZMA2,'dict_size':128*1024**2,'mode':lzma.MODE_FAST,'mf':lzma.MF_HC3,'nice_len':32,'depth':4}]
    # The tar byte stream is unchanged; individual tensor files are never opened.
    with gzip.open(source,'rb') as src,lzma.open(pending,'xb',format=lzma.FORMAT_XZ,check=lzma.CHECK_SHA256,filters=filters) as dst:
        for raw in iter(lambda:src.read(1024**2),b''):
            h.update(raw);total+=len(raw);dst.write(raw)
            if total-reported>=268435456:
                print(json.dumps({'uncompressed_tar_bytes':total,'output_bytes_so_far':pending.stat().st_size}),flush=True);reported=total
    after=source.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):raise ValueError('Original archive changed during encoding')
    # Independent decoder verifies every byte of the original tar stream.
    check=hashlib.sha256();decoded=0
    with lzma.open(pending,'rb') as f:
        for raw in iter(lambda:f.read(1024**2),b''):check.update(raw);decoded+=len(raw)
    if decoded!=total or check.hexdigest()!=h.hexdigest():raise ValueError('Lossless tar-stream verification failed')
    with pending.open('rb') as f:os.fsync(f.fileno())
    if out.exists():raise ValueError('Concurrent output appeared')
    pending.rename(out)
    value={'status':'VERIFIED_LOSSLESS_TAR_STREAM_REENCODING','source_archive':source.name,'source_archive_sha256':expected,
        'archive':out.name,'archive_bytes':out.stat().st_size,'archive_sha256':sha(out),'uncompressed_tar_bytes':total,
        'uncompressed_tar_sha256':h.hexdigest(),'elapsed_seconds':perf_counter()-start,'original_stat_unchanged':True,
        'tensor_or_observation_decodes':0,'new_neural_calls':0,'new_optimizer_updates':0,'target_restore_verification':'PENDING'}
    with out.with_suffix('.receipt.json').open('x') as f:json.dump(value,f,indent=2)
    print(json.dumps(value),flush=True);return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','out','source-sha256'):p.add_argument('--'+n,required=True)
    a=p.parse_args();encode(a.source,a.out,a.source_sha256)


if __name__=='__main__':main()
