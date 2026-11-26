"""Receive checksummed base64 frames over SSH stdin; payload is never printed."""
import argparse,base64,hashlib,json,os,uuid
from pathlib import Path
import sys


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--bytes',type=int,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--frame-bytes',type=int,choices=(1048576,16777216),default=1048576)
    p.add_argument('--status',action='store_true');p.add_argument('--one-frame',action='store_true')
    a=p.parse_args();final=Path(a.out).resolve()
    part=final.with_suffix(final.suffix+'.partial');journal=final.with_suffix(final.suffix+'.progress.json')
    if final.exists():
        if final.stat().st_size!=a.bytes or sha(final)!=a.sha256:raise ValueError('Existing final file differs; never overwrite')
        print(json.dumps({'complete':True,'offset':a.bytes,'archive_sha256':a.sha256}),flush=True);return
    state={'archive_bytes':a.bytes,'archive_sha256':a.sha256,'offset':0,'chunks':[]}
    if journal.exists():
        state=json.loads(journal.read_text())
        if state['archive_bytes']!=a.bytes or state['archive_sha256']!=a.sha256:raise ValueError('Resume contract differs')
        if not part.exists() or part.stat().st_size!=state['offset']:raise ValueError('Uncommitted partial bytes; retain and use new attempt')
        with part.open('rb') as f:
            for row in state['chunks']:
                if hashlib.sha256(f.read(row['bytes'])).hexdigest()!=row['sha256']:raise ValueError('Committed partial chunk differs')
    elif part.exists():raise ValueError('Uncommitted partial archive; no implicit overwrite')
    if a.status:
        print(json.dumps({'complete':False,'offset':state['offset'],'archive_sha256':a.sha256}),flush=True);return
    def commit():
        tmp=journal.with_name(journal.name+'.'+uuid.uuid4().hex)
        with tmp.open('x') as f:json.dump(state,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(tmp,journal)
    with part.open('ab' if journal.exists() else 'xb') as f:
        for line in sys.stdin.buffer:
            if line.rstrip()==b'END':break
            offset,digest,payload=line.rstrip(b'\r\n').split(b'|',2)
            raw=base64.b64decode(payload,validate=True);offset=int(offset);digest=digest.decode('ascii')
            if offset!=state['offset'] or not 0<len(raw)<=a.frame_bytes or offset+len(raw)>a.bytes:
                raise ValueError('Frame offset/size differs')
            if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Frame hash differs')
            f.write(raw);f.flush();os.fsync(f.fileno())
            state['chunks'].append({'offset':offset,'bytes':len(raw),'sha256':digest})
            state['offset']+=len(raw);commit()
            print(json.dumps({'offset':state['offset'],'chunk_sha256':digest}),flush=True)
            if a.one_frame:break
    if a.one_frame and state['offset']<a.bytes:return
    if state['offset']!=a.bytes or sha(part)!=a.sha256:raise ValueError('Final archive bytes/SHA differ')
    if final.exists():raise ValueError('Final publication appeared concurrently')
    part.rename(final);state['complete']=True;commit()
    print(json.dumps({'complete':True,'offset':a.bytes,'archive_sha256':a.sha256}),flush=True)


if __name__=='__main__':main()
