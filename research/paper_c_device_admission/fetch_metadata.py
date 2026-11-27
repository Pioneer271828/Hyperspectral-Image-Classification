"""Bounded official metadata acquisition; no model, pixel or label decoder."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from time import perf_counter
from urllib.parse import quote,urlsplit,urlunsplit

import requests

HOST='https://huggingface.co'
REPO='IGNF/FLAIR-1-2'


def public_url(url):
    p=urlsplit(url);return urlunsplit((p.scheme,p.netloc,p.path,'',''))


def fetch(url,root,name,limit):
    final=root/name;pending=root/(name+'.incoming')
    failure=root/(name+'.failure.json')
    if final.exists() or pending.exists() or failure.exists():raise ValueError('New file identity required; earlier attempts retained')
    begin=perf_counter();h=hashlib.sha256();size=0
    try:
        with requests.get(url,stream=True,timeout=(15,60),headers={'Accept-Encoding':'identity','User-Agent':'Research-metadata-intake/1'}) as response:
            response.raise_for_status()
            if int(response.headers.get('Content-Length','0'))>limit:raise ValueError('Declared source exceeds per-file cap')
            with pending.open('xb') as f:
                for raw in response.iter_content(65536):
                    size+=len(raw)
                    if size>limit:raise ValueError('Streaming source exceeds cap; incoming retained')
                    f.write(raw);h.update(raw)
                f.flush();os.fsync(f.fileno())
            receipt={'url':public_url(url),'final_url':public_url(response.url),'bytes':size,'sha256':h.hexdigest(),
                'etag':response.headers.get('ETag'),'content_type':response.headers.get('Content-Type'),
                'last_modified':response.headers.get('Last-Modified'),'elapsed_seconds':perf_counter()-begin}
    except Exception as err:
        failure.write_text(json.dumps({'url':public_url(url),'error_type':type(err).__name__,'partial_file_bytes':pending.stat().st_size if pending.exists() else 0,'elapsed_seconds':perf_counter()-begin,'signed_redirect_queries_recorded':False},indent=2))
        raise RuntimeError(type(err).__name__+': bounded metadata request failed; see sanitized failure receipt') from None
    pending.rename(final);return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2]
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=a.expected_commit or subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():raise ValueError('Clean expected full source SHA required')
    if os.name!='posix':raise ValueError('Authorized Linux execution only')
    out=Path(a.out).resolve()
    if out.is_relative_to(repo):raise ValueError('Checkout-external intake path required')
    out.mkdir(mode=0o700);rows={};used=0
    rows['dataset-api.json']=fetch(HOST+'/api/datasets/'+REPO,out,'dataset-api.json',4*1024**2)
    source=json.loads((out/'dataset-api.json').read_text());revision=source['sha']
    if not re.fullmatch('[a-f0-9]{40}',revision):raise ValueError('Full immutable dataset revision required')
    resources={'README.md':'README.md','aux-data/load_data.py':'upstream-load-data.txt',
        'aux-data/flair_aerial_metadata.json':'aerial-metadata.json',
        'aux-data/flair-2_centroids_sp_to_patch.json':'centroid-matching.json'}
    used=sum(r['bytes'] for r in rows.values())
    try:
        for path,name in resources.items():
            remaining=67108864-used
            rows[name]=fetch(HOST+'/datasets/'+REPO+'/resolve/'+revision+'/'+quote(path,safe='/'),out,name,min(32*1024**2,remaining))
            used+=rows[name]['bytes'];print(json.dumps({'file':name,'bytes':rows[name]['bytes'],'sha256':rows[name]['sha256']}),flush=True)
    finally:
        manifest={'schema':'paper_c_official_metadata_manifest_v1','source_commit':a.expected_commit,'dataset_repository':REPO,
            'dataset_revision':revision,'files':rows,'successful_bytes':used,'retained_partial_bytes':sum(x.stat().st_size for x in out.glob('*.incoming')),'public_train_pixel_reads':0,'test_pixel_or_label_reads':0,
            'new_optimizer_updates':0,'new_neural_calls':0,'label_decoder_imported':False,
            'license_scope':'HF official card Etalab2 data; code license must be checked separately; paper license is not a data grant.',
            'scope':'Official text/metadata only, no archive or pixel payload. Signed redirect query parameters are not persisted.'}
        with (out/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2,allow_nan=False)
    print(json.dumps({'status':'OFFICIAL_METADATA_PINNED','revision':revision,'bytes':used,'new_pixel_reads':0}))


if __name__=='__main__':main()
