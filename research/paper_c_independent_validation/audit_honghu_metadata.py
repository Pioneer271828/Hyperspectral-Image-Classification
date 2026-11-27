"""Recheck C7 parents using only JSON in the already resident C6 migration archive.

No spectral archive member is extracted or decoded. This reuses ReleasedBank and
C7 selection, not an alternate reader, point selection or authorization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

from build_tables import ROOT, clean
sys.path.insert(0, str(ROOT/'research/paper_c_hsi_pair'))
from c6_bridge import ReleasedBank
from build_plan import select_scene


def metadata_summary(bank, ontology):
    selected = select_scene(bank, ontology)
    roles = []
    for role, points in selected['roles'].items():
        classes = []
        for common in (0, 1):
            group = [p for p in points if p['common_id'] == common]
            classes.append(dict(common_class=common, points=len(group), original_parents=len({p['parent_id'] for p in group}), independently_verified_parcels=None))
        roles.append(dict(original_role=role, total_points=len(points), selected_parents=len({p['parent_id'] for p in points}), selected_tiles=len({p['tile_id'] for p in points}), classes=classes))
    if bank.payload_reads:
        raise ValueError('Metadata audit opened spectral payload')
    return dict(scene=bank.scene, roles=roles, shared_parent_count_excluded=selected['excluded_shared_parent_count'], cross_role_parent_overlap=selected['selected_parent_overlap'], minimum_box_gap_pixels=selected['minimum_role_box_gap_pixels'], original_acquisition_images=1, independent_acquisition_replicates_established=False, independently_verified_parcels=None, untouched_confirmation_units_released=0, original_NPY_members_decoded=0, new_point_memberships=0, query_reads=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('expected-commit', 'archive', 'archive-receipt', 'out'):
        p.add_argument('--'+name, required=True)
    args = p.parse_args()
    clean(args.expected_commit)
    archive = Path(args.archive)
    pin = json.loads(Path(args.archive_receipt).read_text())
    expected = '57326bbac7f67907372db51bf91a494adc23a581b65dc01c825a5f14a3d1a5be'
    if pin['archive_sha256'] != expected or archive.stat().st_size != 55257438:
        raise ValueError('Existing original migration archive identity differs')
    h = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    if h.hexdigest() != expected:
        raise ValueError('Original migration bytes differ')
    out = Path(args.out).resolve()
    if out.exists() or out.is_relative_to(ROOT):
        raise ValueError('New external private metadata output required')
    out.mkdir(mode=0o700)
    extracted, skipped, size = [], 0, 0
    with tarfile.open(archive, 'r:gz') as tf:
        for member in tf:
            allowed = re.fullmatch(r'whu-honghu/bank/(?:registration|bank|tile-\d+)\.json', member.name) or member.name == 'whu-honghu/label-release.json'
            if not allowed:
                skipped += 1
                continue
            if not member.isfile() or member.size > 4*1024**2:
                raise ValueError('Invalid bounded metadata member')
            size += member.size
            if size > 8*1024**2:
                raise ValueError('Metadata-only extraction cap exceeded')
            target = out/member.name
            if not target.resolve().is_relative_to(out) or target.exists():
                raise ValueError('Metadata path escape/duplication')
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = tf.extractfile(member).read()
            json.loads(raw)
            target.write_bytes(raw)
            extracted.append(dict(name=member.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    bank = ReleasedBank(out/'whu-honghu/bank', 'whu-honghu', label_release=out/'whu-honghu/label-release.json')
    ontology = json.loads((ROOT/'research/paper_c_hsi_pair/ontology.json').read_text())
    result = metadata_summary(bank, ontology)
    result.update(source=args.expected_commit, original_archive_sha256=expected, extracted_JSON_members=len(extracted), extracted_JSON_bytes=size, skipped_other_archive_members=skipped, archive_note='Compressed archive is scanned; spectral members are skipped without member extraction/array interpretation', native_full_class_role_points={'devtrain':110, 'devval':110}, source_metadata_manifest=extracted)
    (out/'aggregate.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'source_metadata_manifest'}))


if __name__ == '__main__':
    main()
