"""Restore a byte-deduplicated private backup and verify every logical file."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path,PurePosixPath


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',required=True)
    p.add_argument('--out',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    archive=Path(a.archive);out=Path(a.out).resolve()
    with tarfile.open(archive,'r:gz') as t:
        members=t.getmembers();index=json.load(t.extractfile('object-index01.json'))
        expected={f'objects/{h}' for h in index['objects']}|{'object-index01.json'}
        if {m.name for m in members}!=expected or any(not m.isfile() for m in members):raise ValueError('Unlisted archive member')
        out.mkdir()
        for m in members:
            dest=(out/m.name).resolve()
            if not dest.is_relative_to(out):raise ValueError('Object archive path escape')
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as f:shutil.copyfileobj(t.extractfile(m),f)
    for h,row in index['objects'].items():
        file=out/'objects'/h
        if file.stat().st_size!=row['bytes'] or sha(file)!=h:raise ValueError('Stored object differs')
    logical=out/'restored';logical.mkdir()
    for row in index['files']:
        name=PurePosixPath(row['name']);file=(logical/row['name']).resolve()
        if name.is_absolute() or '..' in name.parts or not file.is_relative_to(logical):raise ValueError('Logical path escape')
        file.parent.mkdir(parents=True,exist_ok=True)
        with (out/'objects'/row['sha256']).open('rb') as source,file.open('xb') as target:shutil.copyfileobj(source,target)
        if file.stat().st_size!=row['bytes'] or sha(file)!=row['sha256']:raise ValueError('Restored logical file differs')
    result={'status':'VERIFIED_ALL_UNIQUE_OBJECTS_AND_RESTORED_LOGICAL_FILES','archive_bytes':archive.stat().st_size,
            'archive_sha256':sha(archive),'logical_files':len(index['files']),'logical_bytes':sum(x['bytes'] for x in index['files']),
            'unique_objects':len(index['objects']),'unique_bytes':sum(x['bytes'] for x in index['objects'].values()),
            'raw_input_GT_query_or_bank_arrays_copied':False,'new_neural_forwards':0,'optimizer_updates':0,
            'recovery_limit':'Output/PDF/code artifacts verified; external raw banks and exact original runtime/path restoration remain dependencies.'}
    with Path(a.receipt).open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
