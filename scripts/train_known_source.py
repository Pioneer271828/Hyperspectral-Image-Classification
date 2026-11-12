"""Train only the registered known-class source rectangle for WP09."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.known_source_training import train_known_source


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('expected-commit', 'registration', 'registration-sha256', 'out'):
        parser.add_argument('--' + flag, required=True)
    args = parser.parse_args()
    registration = dict(path=str(Path(args.registration).resolve()), sha256=args.registration_sha256)
    print(json.dumps(train_known_source(registration, out=args.out, expected_commit=args.expected_commit)))
