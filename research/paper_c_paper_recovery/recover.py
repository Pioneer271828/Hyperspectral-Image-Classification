"""Reconcile and restore frozen evidence on the authorized LXC, without fitting."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import socket
import subprocess
import tarfile
from time import perf_counter

BASE = Path('/home/lbx/hsi-pioneer271828')
REC = BASE / 'backups/paper-c-recovery-20261126-01'
ARCHIVES = {
    'baseline': (REC / 'engineering3-priority01.tar.gz', 106360985,
                 'dc3dd8c12eb3d9c9bfb083c11b7736f27fb0b6171837ff9b82bb1dd90d8c01b7'),
    'transfer': (REC / 'transfer3-priority01.tar.gz', 17655194,
                 '24c7a57188f2185f1f6eba7344702fecdb0653e638c7222ec511c08c69813101'),
    'routes': (REC / 'routing-objects01.tar.gz', 161206780,
               '4684a7ee159b400aa56322614b77089bfcf1e4ddfee15fca16365b4d4923d6d6'),
    'pavia': (BASE / 'backups/paper-a-three-scene-source-20261127/PaviaU-capsule.tar.gz',
              21719117, 'dd91359a38dd86509d4c6c917e5c10e661e650b1345b044fbf7ac00ff46deb0e'),
    'roles': (BASE / 'backups/paper-a-three-scene-source-20261127/roles.tar.gz',
              109681972, 'c61a2b0223fdefb6debbcf4a8091688c7aec7ae0d1ee73b57f441f81fe6995fee'),
}
SUITES = {'baseline': BASE / 'runs/paper-c-recovery-20261126',
          'transfer': BASE / 'runs/paper-c-transfer-engineering-20261126',
          'routes': BASE / 'runs/paper-c-routes-engineering-20261126'}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def relative(name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and '..' not in p.parts
            and '\\' not in name and ':' not in name, 'Unsafe archive/logical path')
    return p


def restore(archive, length, checksum, target):
    require(archive.stat().st_size == length and sha(archive) == checksum,
            'Frozen archive identity differs: ' + archive.name)
    target.mkdir(mode=0o700)
    with tarfile.open(archive, 'r:gz') as tar:
        first = tar.next()
        require(first.isfile() and first.name in ('index.json', 'object-index01.json'), 'Missing first index')
        raw_index = tar.extractfile(first).read()
        (target / 'archive-index.json').write_bytes(raw_index)
        index = json.loads(raw_index)
        files = index['files']
        rows = files if isinstance(files, list) else [dict(v, name=k) for k, v in files.items()]
        logical = {r['name']: r for r in rows}
        require(len(logical) == len(rows), 'Duplicate logical path')
        for name in logical:
            relative(name)
        objects = index.get('objects')
        expected = dict(objects) if objects is not None else dict(logical)
        seen = set()
        while (member := tar.next()) is not None:
            relative(member.name)
            require(member.isfile() and not member.issym() and not member.islnk(), 'Only regular files admitted')
            key = member.name.removeprefix('objects/') if objects is not None else member.name
            require(key in expected and key not in seen, 'Unexpected or duplicate object')
            pin = expected[key]
            require(member.size == pin['bytes'], 'Archive object size differs')
            path = target / member.name
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with tar.extractfile(member) as source, path.open('xb') as dest:
                shutil.copyfileobj(source, dest)
            require(sha(path) == (key if objects is not None else pin['sha256']), 'Archive object SHA differs')
            seen.add(key)
        require(seen == set(expected), 'Incomplete archive')
    root = target / 'restored'
    root.mkdir(mode=0o700)
    for name, row in logical.items():
        source = target / ('objects/' + row['sha256'] if objects is not None else name)
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source, dest)
        require(dest.stat().st_size == row['bytes'] and sha(dest) == row['sha256'], 'Restored logical bytes differ')
    write(target / 'index.json', index)
    return index, logical


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    checkout = Path(__file__).resolve().parents[2]
    git = lambda *args: subprocess.check_output(['git', '-C', str(checkout), *args], text=True).strip()
    require(platform.system() == 'Linux' and socket.gethostname() == 'lbx', 'Authorized LXC required')
    require(re.fullmatch('[a-f0-9]{40}', a.expected_commit) and git('rev-parse', 'HEAD') == a.expected_commit
            and not git('status', '--porcelain'), 'Published SHA and clean checkout required')
    out = a.out.resolve()
    require(out.is_relative_to(BASE / 'backups') and not out.exists(), 'New authorized external backup required')
    out.mkdir(mode=0o700)
    started = perf_counter()
    public = out / 'review'
    public.mkdir(mode=0o700)
    available = {}
    indexes = {}
    archive_rows = []
    for key, (archive, length, checksum) in ARCHIVES.items():
        index, logical = restore(archive, length, checksum, out / key)
        indexes[key] = index
        for name, row in logical.items():
            available.setdefault(row['sha256'], {'path': str(out / key / 'restored' / name),
                                                 'archive': str(archive), 'logical': name, 'bytes': row['bytes']})
        unchanged = 0
        if key in SUITES:
            for name, row in logical.items():
                if name == 'recovery/full-source-history-index.json':
                    continue
                source = SUITES[key] / name
                require(source.is_file() and source.stat().st_size == row['bytes'] and sha(source) == row['sha256'],
                        'Frozen current source differs: ' + name)
                unchanged += 1
        archive_rows.append({'id': key, 'bytes': length, 'sha256': checksum, 'logical_files': len(logical),
                             'logical_bytes': sum(r['bytes'] for r in logical.values()),
                             'current_source_files_verified': unchanged, 'restored_file_differences': 0})
        print(json.dumps({'restored': key, 'files': len(logical)}), flush=True)

    # Historical off-host verification is a recorded fact, never a current host probe.
    history = []
    state_objects = {}
    for key in ('baseline', 'transfer'):
        source = read(out / key / 'restored/recovery/full-source-history-index.json')
        omitted = source['omitted_intermediate_states']
        missing = {}
        for row in omitted:
            original = SUITES[key] / row['name']
            require(original.stat().st_size == row['bytes'] and sha(original) == row['sha256'], 'Historical source state differs')
            state_objects.setdefault(row['sha256'], row['bytes'])
            if row['sha256'] not in available:
                missing.setdefault(row['sha256'], row['bytes'])
        history.append({'id': key, 'omitted_logical_states': len(omitted),
                        'omitted_logical_bytes': sum(r['bytes'] for r in omitted),
                        'unique_hashes_absent_from_restored_critical_and_role_capsules': len(missing),
                        'unique_bytes_absent_from_restored_critical_and_role_capsules': sum(missing.values()),
                        'all_original_omitted_state_hashes_currently_verified': True})
    original = REC / 'engineering3-objects01.tar.gz'
    require(original.stat().st_size == 8337564720 and sha(original) ==
            'b9cf9adc398d79214c23abd034e4178ff4e82c2fdd0fb2ba81fd5ac092fdb045', 'Original full archive differs')

    # Preserve only previously frozen dependency paths; do not enumerate raw datasets.
    reg = read(SUITES['baseline'] / 'engineering3-execution01/registration.json')
    pilot_path = Path(reg['pilot'])
    pilot = read(pilot_path)
    pins = dict(pilot['protected_inputs'])
    pins[str(pilot_path)] = sha(pilot_path)
    for scene in pilot['scenes'].values():
        receipt = scene['receipt']
        pins[receipt['path']] = receipt['sha256']
        registered = Path(read(receipt['path'])['root']) / 'registration.json'
        pins[str(registered)] = sha(registered)
    dependency = read(BASE / 'backups/paper-c-lxc-20261126-01/recovery-dependency-index01.json')
    for row in dependency['files']:
        pins[str(BASE / row['source_relative'])] = row['sha256']
    # Source, licenses, model-sidecar identities and geometry explicitly protected by contracts.
    for key in ('baseline', 'transfer'):
        for case in indexes[key]['recovery']:
            folder = 'engineering3-execution01' if key == 'baseline' else 'execution01'
            c = read(out / key / 'restored' / folder / case['case'] / 'contract.json')
            pins.update(c.get('protected', {}))
    obj = out / 'increment/objects'
    obj.mkdir(parents=True, mode=0o700)
    manifest = []
    missing_paths = []
    for name, checksum in sorted(pins.items()):
        source = Path(name)
        require(source.is_relative_to(BASE), 'Outside authorized project dependency')
        if not source.is_file():
            missing_paths.append({'path': name, 'sha256': checksum})
            continue
        require(not source.is_symlink() and sha(source) == checksum, 'Frozen dependency source differs')
        size = source.stat().st_size
        prior = available.get(checksum)
        if prior is None:
            dest = obj / checksum
            if not dest.exists():
                shutil.copyfile(source, dest)
            require(dest.stat().st_size == size and sha(dest) == checksum, 'Increment source/destination differ')
            available[checksum] = {'path': str(dest), 'archive': 'C11_INCREMENT', 'bytes': size}
        manifest.append({'path': name, 'sha256': checksum, 'bytes': size,
                         'backup': available[checksum], 'new_object': prior is None})
    write(out / 'increment/manifest.json', {'files': manifest, 'missing_sources': missing_paths})
    restored = out / 'increment-restored'
    restored.mkdir(mode=0o700)
    for row in manifest:
        dest = restored / row['sha256']
        if not dest.exists():
            shutil.copyfile(row['backup']['path'], dest)
        require(dest.stat().st_size == row['bytes'] and sha(dest) == row['sha256'], 'Increment fresh restore differs')
    counts = Counter(Path(r['path']).suffix for r in manifest if r['new_object'])
    receipt = {'schema': 'paper_c_paper_recovery_v1', 'producer': a.expected_commit,
               'archives': archive_rows, 'historical_states': history,
               'historical_full_archive': {'bytes': original.stat().st_size, 'sha256': sha(original),
                    'current_LXC_hash_verified': True, 'historical_offhost_verified_prefix_bytes': 535822336,
                    'full_offhost_completion_evidence': False},
               'increment': {'frozen_dependency_paths': len(pins), 'verified_source_paths': len(manifest),
                    'new_unique_objects': sum(r['new_object'] for r in manifest),
                    'new_unique_bytes': sum(r['bytes'] for r in manifest if r['new_object']),
                    'new_object_suffixes': dict(counts), 'missing_source_paths': len(missing_paths),
                    'fresh_restore_differences': 0},
               'historically_verified_offhost_cases': 33, 'current_offhost_connections': 0,
               'current_new_backup_scope': 'SAME_LXC_HOST_ONLY', 'optimizer_updates': 0,
               'new_neural_calls': 0, 'query_reads': 0, 'private_artifacts_exported_to_Windows': False,
               'elapsed_seconds': perf_counter()-started}
    write(public / 'reconciliation.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
