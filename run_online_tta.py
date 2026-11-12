"""Run or resume an authenticated observation-only WP09 stream on CPU."""
import argparse
import json
from pathlib import Path

from transfer.online_tta_run import execute
from transfer.online_tta_stream import optional_pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--to-frame', type=int)
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--admission-sha256')
    args = parser.parse_args()
    pin = dict(path=str(args.registration.resolve()), sha256=args.registration_sha256)
    admission = optional_pin(args.admission, args.admission_sha256)
    print(json.dumps(execute(pin, args.out, to_frame=args.to_frame, admission=admission), sort_keys=True))


if __name__ == '__main__':
    main()
