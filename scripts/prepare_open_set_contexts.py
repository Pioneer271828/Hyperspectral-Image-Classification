"""Prepare strict known-only source/calibration inputs without evaluation labels."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.known_context_preparation import prepare_known_contexts


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('expected-commit','registration','registration-sha256','out'):
        p.add_argument('--'+key, required=True)
    a=p.parse_args()
    print(json.dumps(prepare_known_contexts(dict(path=str(Path(a.registration).resolve()),sha256=a.registration_sha256),
                                         out=a.out,expected_commit=a.expected_commit)))
