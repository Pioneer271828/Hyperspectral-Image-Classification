"""Verify every sealed chunk and publish the original archive without overwrites."""
import argparse,hashlib,json
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--chunks',required=True)
    p.add_argument('--out',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    root=Path(a.chunks).resolve();index=json.loads((root/'manifest.json').read_text());out=Path(a.out)
    for i,row in enumerate(index['parts']):
        if row['name']!=f'chunk-{i:04d}.bin':raise ValueError('Explicit chunk order differs')
        file=root/row['name']
        if file.stat().st_size!=row['bytes'] or sha(file)!=row['sha256']:raise ValueError('Chunk bytes/SHA differ')
    if out.exists():
        if out.stat().st_size!=index['archive_bytes'] or sha(out)!=index['archive_sha256']:raise ValueError('Existing archive differs')
    else:
        partial=out.with_name(out.name+'.building')
        with partial.open('xb') as f:
            for row in index['parts']:
                with (root/row['name']).open('rb') as source:
                    for block in iter(lambda:source.read(1024**2),b''):f.write(block)
        if partial.stat().st_size!=index['archive_bytes'] or sha(partial)!=index['archive_sha256']:
            raise ValueError('Reassembled archive differs from original')
        partial.rename(out)
    receipt={k:index[k] for k in ('archive_bytes','archive_sha256','payload_files','payload_bytes')}
    receipt.update(status='VERIFIED_ALL_CHUNKS_AND_ORIGINAL_ARCHIVE',chunks=len(index['parts']),raw_bank_arrays_copied=False)
    with Path(a.receipt).open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt))


if __name__=='__main__':main()
