"""Produce authenticated three-head development arrays from a fixed M8 checkpoint."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluation.open_set_producer import produce_open_set


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('expected-commit','registration','registration-sha256','out'):p.add_argument('--'+k,required=True)
    a=p.parse_args()
    print(json.dumps(produce_open_set(dict(path=str(Path(a.registration).resolve()),sha256=a.registration_sha256),
        out=a.out,expected_commit=a.expected_commit)))


if __name__=='__main__':main()
