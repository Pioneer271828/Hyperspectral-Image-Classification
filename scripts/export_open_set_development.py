"""Freeze all nine open-set development controls, then release development labels."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.registered_open_set import freeze_open_set,export_open_set


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True)
    sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('freeze');b=sub.add_parser('export')
    for parser,name in ((a,'registration'),(b,'freeze')):
        parser.add_argument('--'+name,type=Path,required=True)
        parser.add_argument('--'+name+'-sha256',required=True);parser.add_argument('--out',type=Path,required=True)
    args=p.parse_args();name='registration' if args.mode=='freeze' else 'freeze'
    pin=dict(path=str(getattr(args,name).resolve()),sha256=getattr(args,name+'_sha256'))
    fn=freeze_open_set if args.mode=='freeze' else export_open_set
    print(json.dumps(fn(pin,out=args.out,expected_commit=args.expected_commit)))


if __name__=='__main__':main()
