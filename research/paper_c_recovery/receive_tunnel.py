"""Read immutable source bytes through one authenticated loopback SSH tunnel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    p.add_argument('--bytes',type=int,required=True);p.add_argument('--sha256',required=True);p.add_argument('--known-hosts',required=True)
    p.add_argument('--port',type=int,default=32294);p.add_argument('--receipt',required=True);a=p.parse_args()
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+',a.source) or not re.fullmatch('[a-f0-9]{64}',a.sha256):raise ValueError('Controlled immutable source path and complete SHA required')
    final=Path(a.out).resolve();part=final.with_suffix(final.suffix+'.partial');journal=final.with_suffix(final.suffix+'.progress.json')
    if Path(a.receipt).exists():raise ValueError('Existing receipt preserved')
    if final.exists():
        if final.stat().st_size!=a.bytes or sha(final)!=a.sha256:raise ValueError('Existing archive differs')
        print(json.dumps({'complete':True,'offset':a.bytes,'archive_sha256':a.sha256}));return
    state={'archive_bytes':a.bytes,'archive_sha256':a.sha256,'offset':0,'chunks':[]}
    if journal.exists():
        state=json.loads(journal.read_text())
        if state['archive_bytes']!=a.bytes or state['archive_sha256']!=a.sha256 or not part.exists() or part.stat().st_size!=state['offset']:
            raise ValueError('Uncommitted partial bytes or changed contract; preserve and use new attempt')
        with part.open('rb') as f:
            for row in state['chunks']:
                if hashlib.sha256(f.read(row['bytes'])).hexdigest()!=row['sha256']:raise ValueError('Committed prefix differs')
    elif part.exists():raise ValueError('Uncommitted orphan archive preserved')
    offset=state['offset']
    command=['ssh','-T','-p',str(a.port),'-o','StrictHostKeyChecking=yes','-o','HostKeyAlgorithms=ssh-ed25519',
        '-o','UserKnownHostsFile='+str(Path(a.known_hosts).resolve()),'-o','PubkeyAuthentication=no',
        '-o','PreferredAuthentications=password','-o','ConnectTimeout=15','-o','ServerAliveInterval=15',
        'lbx@127.0.0.1',f'tail -c +{offset+1} {a.source}']
    # SSH reads its password from the inherited controlling TTY. Binary source
    # stdout goes only into this private receiver, never to that TTY or Windows.
    child=subprocess.Popen(command,stdout=subprocess.PIPE)
    def commit():
        temp=journal.with_name(journal.name+'.'+uuid.uuid4().hex)
        with temp.open('x') as f:json.dump(state,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(temp,journal)
    try:
        with part.open('ab' if journal.exists() else 'xb') as f:
            while state['offset']<a.bytes:
                wanted=min(16777216,a.bytes-state['offset']);pieces=[];n=0
                while n<wanted:
                    chunk=child.stdout.read(wanted-n)
                    if not chunk:raise EOFError('Source SSH ended before next complete checkpointed block')
                    pieces.append(chunk);n+=len(chunk)
                raw=b''.join(pieces);h=hashlib.sha256(raw).hexdigest()
                f.write(raw);f.flush();os.fsync(f.fileno())
                state['chunks'].append({'offset':state['offset'],'bytes':n,'sha256':h});state['offset']+=n;commit()
                print(json.dumps({'verified_bytes':state['offset']}),flush=True)
        if child.stdout.read(1):raise ValueError('Source exceeds sealed archive length')
        if child.wait()!=0:raise ValueError('Source SSH exit failure')
        if part.stat().st_size!=a.bytes or sha(part)!=a.sha256:raise ValueError('Complete target bytes/SHA differ from source seal')
        if final.exists():raise ValueError('Concurrent archive publication appeared')
        part.rename(final);state['complete']=True;commit()
        receipt={'status':'VERIFIED_PRIVATE_LOOPBACK_SSH_BINARY_TRANSFER','archive_bytes':a.bytes,'archive_sha256':a.sha256,
            'resume_start':offset,'verified_offset':a.bytes,'windows_payload_files':0,'credential_files':0,
            'source_authentication':'interactive SSH only; strict preverified host key; target loopback forwarding only'}
        with Path(a.receipt).open('x') as f:json.dump(receipt,f,indent=2)
        print(json.dumps(receipt),flush=True)
    finally:
        if child.poll() is None:child.terminate();child.wait()


if __name__=='__main__':main()
