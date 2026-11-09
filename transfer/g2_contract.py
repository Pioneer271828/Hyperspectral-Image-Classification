"""Validate a declared G2 holdout and lineage before wiring a training runner."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json

WHU_HI = frozenset({'whu-honghu', 'whu-hanchuan', 'WHUL'})
ACCESS = {
    'fit_statistics': ('train', False),
    'train_graph': ('train', False),
    'source_train': ('train', True),
    'validation_graph': ('val', False),
    'source_select': ('val', True),
}


def fields(record, expected, name):
    if not isinstance(record, dict) or record.keys() != set(expected.split()):
        raise ValueError(f'Invalid {name} fields')


def sha(value):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError('Expected a complete SHA256')
    return value


def distinct(values, name):
    if (not isinstance(values, list) or not values
            or any(not isinstance(v, str) or not v.strip() or v.strip() != v for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(f'Expected distinct nonempty {name}')
    return set(values)


def validate_contract(catalog, contract):
    fields(catalog, 'policy scenes', 'catalog')
    fields(contract, 'policy catalog_sha256 target sources exclude semantic_classes semantic_evidence_sha256 '
                     'semantic_mapping unmapped_policy source_releases selection', 'contract')
    if (catalog['policy'] != 'g2_family_inventory_v1' or contract['policy'] != 'g2_closed_set_v1'
            or contract['selection'] != 'source_validation_only'
            or contract['unmapped_policy'] != 'exclude_before_sampling_and_scoring'):
        raise ValueError('Expected strict G2 source-only policy')
    sha(contract['catalog_sha256'])
    sha(contract['semantic_evidence_sha256'])
    scenes = catalog['scenes']
    if not isinstance(scenes, dict) or not WHU_HI.issubset(scenes):
        raise ValueError('Complete WHU-Hi family inventory required')
    for name, row in scenes.items():
        fields(row, 'family evidence_level artifact_sha256 family_evidence_sha256 class_table_sha256 class_count', 'scene')
        if type(row['class_count']) is not int or row['class_count'] < 2:
            raise ValueError(f'Invalid class count for {name}')
        if row['evidence_level'] not in ('verified_local_binding', 'nominal_only', 'unresolved'):
            raise ValueError(f'Unknown evidence level for {name}')
        family = row['family']
        if family is not None and (not isinstance(family, str) or not family.strip() or family.strip() != family):
            raise ValueError(f'Invalid family for {name}')
    families = {scenes[n]['family'] for n in WHU_HI}
    if len(families) != 1 or None in families:
        raise ValueError('WHU-Hi scenes must share a canonical family')
    target = contract['target']
    if not isinstance(target, str) or target not in scenes:
        raise ValueError('Unknown target scene')
    sources = distinct(contract['sources'], 'source scene names')
    participating = sources | {target}
    if target in sources or not participating.issubset(scenes):
        raise ValueError('Unknown or overlapping source/target scene')
    for n in sorted(participating):
        row = scenes[n]
        if row['evidence_level'] != 'verified_local_binding' or row['family'] is None:
            raise ValueError(f'Local family binding missing for {n}')
        for key in ('artifact_sha256', 'family_evidence_sha256', 'class_table_sha256'):
            sha(row[key])
    excluded = {n for n, row in scenes.items() if row['family'] == scenes[target]['family']}
    if distinct(contract['exclude'], 'excluded scenes') != excluded or sources & excluded:
        raise ValueError('The whole target family must be excluded from source roles')
    semantics = distinct(contract['semantic_classes'], 'ordered semantic classes')
    if len(semantics) < 2:
        raise ValueError('Closed set requires at least two semantic classes')
    mapping = contract['semantic_mapping']
    if not isinstance(mapping, dict) or mapping.keys() != participating:
        raise ValueError('Explicit semantic mapping required for each participating scene')
    for n, row in mapping.items():
        allowed = {str(k) for k in range(1, scenes[n]['class_count']+1)}
        if (not isinstance(row, dict) or len(row) != len(semantics)
                or not row.keys() <= allowed
                or any(not isinstance(v, str) for v in row.values()) or set(row.values()) != semantics):
            raise ValueError(f'Semantic mapping is incomplete or ambiguous for {n}')
    releases = contract['source_releases']
    if not isinstance(releases, dict) or releases.keys() != sources:
        raise ValueError('Source-only releases required')
    for partitions in releases.values():
        fields(partitions, 'train val', 'source release')
        for row in partitions.values():
            fields(row, 'observation_sha256 labels_sha256', 'partition release')
            sha(row['observation_sha256'])
            sha(row['labels_sha256'])
        if (partitions['train']['observation_sha256'] == partitions['val']['observation_sha256']
                or partitions['train']['labels_sha256'] == partitions['val']['labels_sha256']):
            raise ValueError('Source training and selection releases must differ')
    return excluded


def validate_lineage(contract, lineage, contract_sha256):
    fields(lineage, 'policy contract_sha256 selected records', 'lineage')
    if lineage['policy'] != 'g2_source_lineage_v1' or lineage['contract_sha256'] != sha(contract_sha256):
        raise ValueError('Lineage is not bound to the reviewed G2 contract')
    selected = sha(lineage['selected'])
    records = lineage['records']
    if not isinstance(records, dict) or selected not in records:
        raise ValueError('Selected checkpoint receipt missing')
    sources = set(contract['sources'])
    used = set()
    visiting = set()
    seen_phases = set()

    def visit(checkpoint):
        if checkpoint in visiting or checkpoint not in records:
            raise ValueError('Cyclic or incomplete checkpoint ancestry')
        if checkpoint in used:
            return
        visiting.add(checkpoint)
        sha(checkpoint)
        record = records[checkpoint]
        fields(record, 'parents access head', 'checkpoint receipt')
        parents = record['parents']
        if not isinstance(parents, list):
            raise ValueError('Malformed checkpoint parents')
        parents = [sha(parent) for parent in parents]
        if len(set(parents)) != len(parents):
            raise ValueError('Duplicate checkpoint parents')
        for parent in parents:
            visit(parent)
        access = record['access']
        if not isinstance(access, list) or not access:
            raise ValueError('Missing checkpoint access receipt')
        for row in access:
            fields(row, 'scene phase partition observation_sha256 labels_sha256', 'phase access')
            scene, phase = row['scene'], row['phase']
            if scene not in sources or phase not in ACCESS:
                raise ValueError('Target, non-source or unregistered phase access')
            partition, with_labels = ACCESS[phase]
            release = contract['source_releases'][scene][partition]
            if (row['partition'] != partition or row['observation_sha256'] != release['observation_sha256']
                    or row['labels_sha256'] != (release['labels_sha256'] if with_labels else None)):
                raise ValueError('Source release or label access differs from allowed phase')
            seen_phases.add(phase)
        visiting.remove(checkpoint)
        used.add(checkpoint)

    visit(selected)
    if used != records.keys() or not {'source_train', 'source_select'}.issubset(seen_phases):
        raise ValueError('Incomplete training/selection lineage or unused checkpoint receipts')
    head = records[selected]['head']
    fields(head, 'kind ordered_classes state_sha256', 'selected head')
    if head['kind'] != 'shared_semantic' or head['ordered_classes'] != contract['semantic_classes']:
        raise ValueError('Selected checkpoint requires a shared head with exact semantic order')
    sha(head['state_sha256'])
    return used


def read_pinned(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha(expected):
        raise ValueError(f'Pinned document changed: {path.name}')
    return json.loads(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('catalog', 'contract', 'lineage'):
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    if code['commit'] != args.expected_commit:
        raise ValueError('Unexpected G2 validator commit')
    require_frozen_code(code)
    catalog = read_pinned(args.catalog, args.catalog_sha256)
    contract = read_pinned(args.contract, args.contract_sha256)
    lineage = read_pinned(args.lineage, args.lineage_sha256)
    if contract['catalog_sha256'] != args.catalog_sha256:
        raise ValueError('Contract catalog identity differs')
    excluded = validate_contract(catalog, contract)
    visited = validate_lineage(contract, lineage, args.contract_sha256)
    require_frozen_code(code)
    for name in ('catalog', 'contract', 'lineage'):
        if hash_file(getattr(args, name)) != getattr(args, name+'_sha256'):
            raise ValueError('A reviewed document changed during validation')
    out = external_directory(args.out)
    out.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'passed_declared_contract', 'commit': code['commit'],
               'catalog_sha256': args.catalog_sha256, 'contract_sha256': args.contract_sha256,
               'lineage_sha256': args.lineage_sha256, 'excluded_scenes': sorted(excluded),
               'checked_checkpoints': sorted(visited), 'training_executable': False,
               'query_accessed': False}
    write_json(out/'receipt.json', receipt)


if __name__ == '__main__':
    main()
