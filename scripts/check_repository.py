"""Run in the remote training container; never trains or reads real datasets."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repository_delivery import STATUS_PATH, SUMMARY_PATH, render_status, check_repository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit')
    parser.add_argument('--write-status', action='store_true', help='Only render the status Markdown; no checks')
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Run code/checks in the remote Linux training environment')
    if args.write_status:
        value = json.loads((ROOT / STATUS_PATH).read_text(encoding='utf-8'))
        (ROOT / SUMMARY_PATH).write_text(render_status(value), encoding='utf-8')
    else:
        print(json.dumps(check_repository(ROOT, args.expected_commit), ensure_ascii=False))


if __name__ == '__main__':
    main()
