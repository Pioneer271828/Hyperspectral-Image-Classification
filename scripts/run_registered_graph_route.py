"""Execute only a pre-registered E3 development segment; never release query."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from transfer.registered_development import execute_route


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--registration-sha256', required=True)
    p.add_argument('--to-step', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--resume-sha256')
    a = p.parse_args()
    if (a.resume is None) != (a.resume_sha256 is None): p.error('Resume path and external hash must be paired')
    torch.set_num_threads(1)
    result = execute_route(dict(path=str(a.registration.resolve()), sha256=a.registration_sha256),
        out=a.out, to_step=a.to_step, expected_commit=a.expected_commit,
        resume=None if a.resume is None else dict(path=str(a.resume.resolve()), sha256=a.resume_sha256))
    print(json.dumps({k: result[k] for k in ('status','completed_updates','new_updates','checkpoint','query')}))


if __name__ == '__main__': main()
