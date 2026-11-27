"""Seal only the authorized public-intake snapshot on Linux for private backup."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from catalog import guard


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for raw in iter(lambda:f.read(1048576),b''):h.update(raw)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=guard(a.expected_commit,a.out);root=Path(a.run).resolve()
    if out.exists():raise ValueError('New sealed backup path required')
    directories=('ovh-official-metadata02','ovh-training-catalog01','public-train-intake01','paired-train-intake01')
    index=[]
    for name in directories:
        if not (root/name/'manifest.json').is_file():raise ValueError('Completed intake/metadata manifest required before seal')
        for path in sorted((root/name).rglob('*')):
            if path.is_symlink():raise ValueError('Snapshot symlinks not admitted')
            if path.is_file():index.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':file_sha(path)})
    with tarfile.open(out,'x:gz',compresslevel=1) as t:
        for row in index:t.add(root/row['path'],arcname=row['path'],recursive=False)
        raw=(json.dumps({'producer':a.expected_commit,'files':index,'attribution':'IGN FLAIR, Garioud et al. NeurIPS2023, DOI10.48550/arXiv.2310.13336; dataset Etalab Open Licence2.0. Exact official source URLs retained in metadata manifests.','scope':'Private authorized backup; includes public training samples/labels and geospatial metadata; never place on Windows or public Git.'},indent=2)+'\n').encode()
        import io
        info=tarfile.TarInfo('snapshot-index.json');info.size=len(raw);info.mode=0o600;t.addfile(info,io.BytesIO(raw))
    receipt={'schema':'paper_c_private_intake_snapshot_v1','producer':a.expected_commit,'archive_bytes':out.stat().st_size,'archive_sha256':file_sha(out),'source_files':len(index),'source_decoded_bytes':sum(x['bytes'] for x in index),'off_host_verified':False,'attribution':'IGN FLAIR, Etalab Open Licence2.0','payload_files_on_Windows':0,'new_optimizer_updates':0}
    out.with_suffix(out.suffix+'.receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))


if __name__=='__main__':main()
