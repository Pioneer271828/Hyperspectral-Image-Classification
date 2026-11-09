"""Freeze then export registered development diagnostics; no query release."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from evaluation.registered_spatial_diagnostics import freeze_diagnostics,export_diagnostics


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--expected-commit',required=True)
    sub=parser.add_subparsers(dest='mode',required=True)
    freeze=sub.add_parser('freeze');export=sub.add_parser('export')
    for p in (freeze,export):p.add_argument('--out',type=Path,required=True)
    for name in ('registration','completed','requests'):
        freeze.add_argument('--'+name,type=Path,required=True);freeze.add_argument('--'+name+'-sha256',required=True)
    export.add_argument('--freeze',type=Path,required=True);export.add_argument('--freeze-sha256',required=True)
    a=parser.parse_args();torch.set_num_threads(1)
    def pin(name):return dict(path=str(getattr(a,name).resolve()),sha256=getattr(a,name+'_sha256'))
    if a.mode=='freeze':result=freeze_diagnostics(pin('registration'),pin('completed'),pin('requests'),out=a.out,expected_commit=a.expected_commit)
    else:result=export_diagnostics(pin('freeze'),out=a.out,expected_commit=a.expected_commit)
    print(json.dumps(result))


if __name__=='__main__':main()
