"""Freeze original joint benchmark inputs or run a bounded three-dataset loop."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joint_baseline import load_config, prepare, short_loop, verify_prepared
from loadData.dataset_registry import file_hash
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'short-loop'])
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--config', default='configs/experiments/base97_joint.json')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--prepared-dir')
    parser.add_argument('--prepared-manifest-hash')
    parser.add_argument('--cycles', type=int, choices=[1,2], default=2)
    args = parser.parse_args(argv)
    source = code_snapshot()
    if source['dirty'] or source['commit'] != args.expected_commit or sys.platform != 'linux':
        raise RuntimeError('Run only synchronized, committed, clean code in the remote Linux environment')
    config = load_config(args.config)
    out = external_directory(args.out_dir)
    if out.exists():
        raise ValueError('Fresh output directory required; this entry does not resume training')
    try:
        if args.phase == 'prepare':
            prepare(config, args.data_root, out, source)
            require_frozen_code(source)
            print(json.dumps({'status': 'prepared', 'manifest_sha256': file_hash(out/'manifest.json')}), flush=True)
        else:
            if not args.prepared_dir or not args.prepared_manifest_hash:
                raise ValueError('Pinned prepared manifest path and SHA256 required')
            if out == Path(args.prepared_dir).resolve() or Path(args.prepared_dir).resolve() in out.parents:
                raise ValueError('Run must be separate from immutable prepared data')
            items, manifest = verify_prepared(args.prepared_dir, args.prepared_manifest_hash,
                                              config, args.data_root, source)
            result = short_loop(config, items, out, cycles=args.cycles,
                                provenance={'code': source,
                                            'prepared_manifest_sha256': args.prepared_manifest_hash})
            require_frozen_code(source)
            result.update(code=source, prepared_manifest_sha256=args.prepared_manifest_hash,
                          config=config.to_dict(), data_root=str(Path(args.data_root).resolve()))
            write_json(out/'result.json', result)
            print(json.dumps({'status': 'passed', 'optimizer_steps': result['runtime']['actual_optimizer_steps'],
                              'selected_cycle': result['selected_cycle'],
                              'elapsed_seconds': result['elapsed_seconds']}), flush=True)
    except Exception as error:
        if out.is_dir():
            write_json(out/'failure.json', {'status': 'failed', 'type': type(error).__name__,
                                          'error': str(error), 'code': source})
            if (out/'result.json').is_file():
                failed = json.loads((out/'result.json').read_text(encoding='utf-8'))
                failed.update(status='failed', error=str(error))
                write_json(out/'result.json', failed)
        raise


if __name__ == '__main__':
    main()
