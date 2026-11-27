"""Fetch the original IGN-hosted metadata snapshot; never fetch test payloads."""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import subprocess
import zipfile

from fetch_metadata import fetch

BASE='https://storage.gra.cloud.ovh.net/v1/AUTH_366279ce616242ebb14161b7991a8461/defi-ia/flair_data_2/'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2];out=Path(a.out).resolve()
    if os.name!='posix' or out.is_relative_to(repo):raise ValueError('Authorized external Linux artifact root required')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=a.expected_commit or subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():raise ValueError('Clean expected commit required')
    out.mkdir(mode=0o700);rows={};members={}
    for remote,local in (('flair_2_aerial_metadata.zip','camera-metadata'),('flair_2_centroids_sp_to_patch.zip','centroid-matching')):
        rows[remote]=fetch(BASE+remote,out,remote,8*1024**2)
        with zipfile.ZipFile(out/remote) as z:
            candidates=[i for i in z.infolist() if not i.is_dir() and i.filename.endswith('.json')]
            if len(candidates)!=1:raise ValueError('Exactly one metadata JSON required')
            info=candidates[0];path=PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts or info.file_size>32*1024**2:raise ValueError('Unreviewed member path/size')
            raw=z.read(info);json.loads(raw)
            with (out/(local+'.json')).open('xb') as f:f.write(raw)
            members[local]={'archive':remote,'member':info.filename,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'crc32':info.CRC}
    value={'schema':'paper_c_original_ign_metadata_snapshot_v1','producer':a.expected_commit,'official_source':'https://ignf.github.io/FLAIR/FLAIR2/flair_2.html',
        'archives':rows,'members':members,'snapshot_limit':'Original OVH-hosted snapshot; not silently equated to later combined HF55-domain version.',
        'new_pixel_or_label_payload_reads':0,'new_neural_calls':0,'new_optimizer_updates':0}
    with (out/'manifest.json').open('x') as f:json.dump(value,f,indent=2)
    camera=json.loads((out/'camera-metadata.json').read_text());example=next(iter(camera.values()))
    print(json.dumps({'status':'ORIGINAL_OFFICIAL_METADATA_PINNED','records':len(camera),'field_names':list(example),'member_receipts':members}))


if __name__=='__main__':main()
