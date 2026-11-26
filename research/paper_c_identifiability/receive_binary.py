"""Receive raw SSH bytes into durable hashed frames; publish only whole-SHA proof."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True);p.add_argument('--bytes',type=int,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--frame-bytes',type=int,default=16777216);p.add_argument('--status',action='store_true');a=p.parse_args()
    if a.bytes<=0 or not 1<=a.frame_bytes<=16777216 or len(a.sha256)!=64 or any(c not in '0123456789abcdef' for c in a.sha256):
        raise ValueError('Positive archive/frame bounds and full SHA256 required')
    os.umask(0o077);final=Path(a.out).resolve();part=final.with_suffix(final.suffix+'.partial')
    journal=final.with_suffix(final.suffix+'.progress.json');lock=final.with_suffix(final.suffix+'.lock')
    with lock.open('a+b') as guard:
        fcntl.flock(guard,fcntl.LOCK_NB|(fcntl.LOCK_SH if a.status else fcntl.LOCK_EX))
        if final.exists():
            if final.stat().st_size!=a.bytes or sha(final)!=a.sha256:raise ValueError('Published archive differs; never overwrite')
            print(json.dumps({'complete':True,'offset':a.bytes,'archive_sha256':a.sha256}),flush=True);return
        state={'archive_bytes':a.bytes,'archive_sha256':a.sha256,'offset':0,'chunks':[]}
        if journal.exists():
            state=json.loads(journal.read_text())
            if state['archive_bytes']!=a.bytes or state['archive_sha256']!=a.sha256:raise ValueError('Original resume identity differs')
            if not part.exists() or part.stat().st_size!=state['offset']:raise ValueError('Uncommitted partial bytes; retain and investigate')
            cursor=0
            with part.open('rb') as f:
                for row in state['chunks']:
                    if row['offset']!=cursor or not 0<row['bytes']<=16777216:raise ValueError('Noncontiguous frame history')
                    raw=f.read(row['bytes'])
                    if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('Committed prefix corrupt')
                    cursor+=row['bytes']
            if cursor!=state['offset'] or not 0<=cursor<=a.bytes:raise ValueError('Resume cursor differs')
        elif part.exists():raise ValueError('Orphan partial retained; no implicit overwrite')
        if a.status:
            print(json.dumps({'complete':False,'offset':state['offset'],'archive_sha256':a.sha256,
                              'prefix_frame_hashes_verified':True,'whole_source_agreement_verified':False}),flush=True);return
        def commit():
            tmp=journal.with_name(journal.name+'.'+uuid.uuid4().hex)
            with tmp.open('x') as f:json.dump(state,f,indent=2);f.flush();os.fsync(f.fileno())
            os.replace(tmp,journal);fsync_directory(journal.parent)
        last_report=state['offset']
        with part.open('ab' if journal.exists() else 'xb') as f:
            while state['offset']<a.bytes:
                wanted=min(a.frame_bytes,a.bytes-state['offset']);raw=bytearray()
                while len(raw)<wanted:
                    data=sys.stdin.buffer.read(wanted-len(raw))
                    if not data:raise EOFError('SSH ended inside an uncommitted in-memory frame; source retained')
                    raw.extend(data)
                f.write(raw);f.flush();os.fsync(f.fileno())
                state['chunks'].append({'offset':state['offset'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
                state['offset']+=len(raw);commit()
                if state['offset']-last_report>=67108864:
                    print(json.dumps({'committed_offset':state['offset'],'whole_source_agreement_verified':False}),flush=True)
                    last_report=state['offset']
        if sha(part)!=a.sha256:raise ValueError('Complete source SHA differs; do not publish')
        if final.exists():raise ValueError('Concurrent publication detected')
        part.rename(final);fsync_directory(final.parent);state['complete']=True;commit()
        print(json.dumps({'complete':True,'offset':a.bytes,'archive_sha256':a.sha256,
                          'whole_source_agreement_verified':True,'new_optimizer_updates':0}),flush=True)


if __name__=='__main__':main()
