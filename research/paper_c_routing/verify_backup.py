"""Extract a sealed C pilot backup into a new directory and verify every byte."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',required=True)
    p.add_argument('--out',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    # Pilot and routing archives have different manifests; both contain only
    # explicitly enumerated regular files, never filesystem-wide traversal.
    archive=Path(a.archive).resolve();out=Path(a.out).resolve();before=(archive.stat().st_size,sha(archive))
    if out.exists():raise ValueError('New extraction directory required; no overwrite')
    with tarfile.open(archive,'r:gz') as t:
        members=t.getmembers()
        if any(not m.isfile() or PurePosixPath(m.name).is_absolute() or '..' in PurePosixPath(m.name).parts for m in members):
            raise ValueError('Unsafe or non-file archive member')
        names={m.name for m in members}
        index_name='pilot-index01.json' if 'pilot-index01.json' in names else 'routing-index01.json'
        index=json.load(t.extractfile(index_name))
        if names!={x['name'] for x in index['files']}|{index_name}:
            raise ValueError('Manifest does not cover every archive member')
        out.mkdir()
        for m in members:
            path=(out/m.name).resolve()
            if not path.is_relative_to(out):raise ValueError('Backup extraction escape')
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as f:
                source=t.extractfile(m)
                for block in iter(lambda:source.read(1024**2),b''):f.write(block)
    for row in index['files']:
        path=out/row['name']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('Extracted backup differs from sealed source')
    models=[]
    for row in index['models']:
        path=out/row['checkpoint']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('Selected model differs')
        if not (path.parent/'result.json').is_file():raise ValueError('Selected model lost its recovery metadata')
        models.append({k:row[k] for k in ('case','sha256','bytes','selected_step','optimizer_RNG_state')})
    if (index_name=='pilot-index01.json' and len(models)!=9) or (archive.stat().st_size,sha(archive))!=before:
        raise ValueError('Selected roster or archive changed during verification')
    result={'status':'VERIFIED_SOURCE_MANIFEST_AND_EXTRACTED_FILES','archive_name':archive.name,
            'archive_bytes':before[0],'archive_sha256':before[1],'verified_files':len(index['files']),
            'verified_selected_models':models,'payload_bytes':sum(x['bytes'] for x in index['files']),
            'recovery_index_present':True,'new_neural_forwards':0,'optimizer_updates':0,
            'raw_data_or_bank_arrays_transferred':False,
            'remaining_dependencies':['exact development banks and observation package; cross-host data authorization pending',
                                      'pinned project/bank code and sparse author code checkout',
                                      'compatible Python/PyTorch environment; exact numerical replay not yet exercised'],
            'recovery_limit':'Original pilot saved selected model state and BN buffers, not optimizer/RNG continuation state.',
            'manifest_schema':index['schema']}
    with Path(a.receipt).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('status','archive_bytes','archive_sha256','verified_files','payload_bytes')}))


if __name__=='__main__':main()
