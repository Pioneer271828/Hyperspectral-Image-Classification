"""Inspect only original official training ZIP directories through bounded byte ranges."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

from ovh_metadata import BASE
from range_zip import RangeFile,validate_member

TRAIN_DOMAINS=tuple('D006_2020 D007_2020 D008_2019 D009_2019 D013_2020 D016_2020 D017_2018 D021_2020 D023_2020 D030_2021 D032_2019 D033_2021 D034_2021 D035_2020 D038_2021 D041_2021 D044_2020 D046_2019 D049_2020 D051_2019 D052_2019 D055_2018 D060_2021 D063_2019 D070_2020 D072_2019 D074_2020 D078_2021 D080_2021 D081_2020 D086_2020 D091_2021'.split())
ARCHIVES={'aerial':'flair_aerial_train.zip','labels':'flair_labels_train.zip','sentinel':'flair_sen_train.zip'}


def guard(expected,out):
    repo=Path(__file__).resolve().parents[2];out=Path(out).resolve()
    if os.name!='posix' or out.is_relative_to(repo):raise ValueError('External authorized Linux artifact directory required')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=expected or subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():raise ValueError('Clean expected full commit required')
    return out


def domain_of(name):
    import re
    domains=re.findall(r'(?<![A-Za-z0-9])D\d{3}_\d{4}(?![A-Za-z0-9])',name)
    if len(set(domains))!=1:raise ValueError('Exactly one explicit domain required')
    return domains[0]


def train_member(name):
    if domain_of(name) not in TRAIN_DOMAINS:raise ValueError('Pixel member is not in the frozen official training whitelist')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=guard(a.expected_commit,a.out);out.mkdir(mode=0o700);receipts={}
    for key,name in ARCHIVES.items():
        rf=RangeFile(BASE+name,budget=16*1024**2)
        try:
            with zipfile.ZipFile(rf) as z:
                members=[]
                for i in z.infolist():
                    validate_member(i,cap=2**40)
                    if i.is_dir():continue
                    members.append({'name':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,'crc32':i.CRC,'compression':i.compress_type,'header_offset':i.header_offset})
                raw=(json.dumps(members,sort_keys=True,separators=(',',':'))+'\n').encode()
                (out/(key+'.json')).write_bytes(raw)
                receipts[key]={'url':BASE+name,'archive_bytes':rf.size,'etag':rf.etag,'received_bytes':rf.received,'ranges':rf.ranges,'catalog_sha256':hashlib.sha256(raw).hexdigest(),'members':len(members),'example_names':[r['name'] for r in members[:5]],'payload_read':False}
                print(json.dumps({'archive':key,**{k:v for k,v in receipts[key].items() if k not in ('ranges','url')}}),flush=True)
        finally:
            (out/'manifest.json').write_text(json.dumps({'producer':a.expected_commit,'archives':receipts,'test_or_query_archives_requested':0,'full_archive_sha256':None,'identity_limit':'ETag/length/range hashes pin partial reads only; no full archive checksum claim.'},indent=2))


if __name__=='__main__':main()
