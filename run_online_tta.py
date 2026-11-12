"""Run or resume an authenticated observation-only WP09 stream on CPU."""
import argparse
import json
from pathlib import Path

from transfer.online_tta_run import execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--registration-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--to-frame', type=int)
    args = parser.parse_args()
    pin = dict(path=str(args.registration.resolve()), sha256=args.registration_sha256)
    print(json.dumps(execute(pin, args.out, to_frame=args.to_frame), sort_keys=True))


if __name__ == '__main__':
    main()
