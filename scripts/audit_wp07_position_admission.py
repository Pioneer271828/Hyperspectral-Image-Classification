"""Read-only source/role/view evidence and pre-score position qualification. No new real view is materialized."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from model.research_config import digest
from research_runtime import code_snapshot, require_frozen_code
from scripts.run_wp07_condition_integration import table
from transfer import registered_development as dev
from transfer import wp07_exact_positions as positions
from transfer import wp07_observed_conditions as summary
from transfer.wp07_position_real import RegisteredPositionProvider
from transfer.wp07_graph_range import require


PREFLIGHT = {
    'paviaU': ('wp02-paviau-final-nov18/preflight/paviaU.json', '61a5264dc021979803cda0d7bd279d9644f8a50505f256aeded876ab39a7b77d'),
    'xuzhou': ('wp07-native-duty01/duties-preflight/xuzhou.json', 'bab1f7881873fe8aef0c50646a32c384a70536b6dff68647450e5ba4dc490238'),
}


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    args = parser.parse_args()
    code = code_snapshot(); require_frozen_code(code)
    require(code['commit'] == args.expected_commit, 'Wrong frozen admission source')
    args.out.mkdir()
    evidence, roster, qualification = [], [], []
    for scene, (relative, expected) in PREFLIGHT.items():
        path = args.root/relative
        raw = path.read_bytes(); require(hashlib.sha256(raw).hexdigest() == expected, 'Preflight changed')
        preflight = json.loads(raw)
        gate = RegisteredPositionProvider(args.root, preflight, 64)
        bank = gate.provider._bank
        identity = bank.identity
        n = bank.release.shape[-1]
        proposal = positions.proposed_real_positions(n)
        # Qualify on index representations only, before any prediction or new real view.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(27183)
            old = summary.IndexConditionEncoder()
            encoder = positions.PositionEncoder(old)
        ref = encoder.exact_mask[0].weight
        for name, indices in proposal.items():
            vector = positions.mask_features(indices, n)
            packet = summary.IndexPacket(None, indices, n)
            with torch.no_grad():
                encoder.representation = 'summary8'; small = encoder(packet, None, None, ref)
                encoder.representation = 'exact_mask'; exact = encoder(packet, None, None, ref)
            qualification.append(dict(scene=scene, view=name, master_count=n, retained_count=len(indices),
                minimum=indices[0], maximum=indices[-1], span=indices[-1]-indices[0],
                positions_sha256=digest(indices), mask_sha256=digest(vector),
                summary8=digest(small.tolist()), exact_mask_encoding=digest(exact.tolist()),
                new_view_materialized=False, prediction_scores_accessed=False))
        require(qualification[-3]['summary8'] == qualification[-1]['summary8']
                and qualification[-3]['exact_mask_encoding'] != qualification[-1]['exact_mask_encoding'],
                'Internal position proposal not identifiable under its intended representation')
        require(qualification[-3]['exact_mask_encoding'] != qualification[-2]['exact_mask_encoding'], 'Shift not identifiable')
        roles = {}
        for role in ('devtrain', 'devval'):
            rows = bank.role_rows(role)
            roles[role] = dict(blocks=len(rows), labelled_blocks=sum(bool(row['samples']) for row in rows),
                               released_points=sum(len(row['samples']) for row in rows),
                               representation_counts=dict(Counter(row['representation'] for row in rows)))
            tile = min(row['id'] for row in rows if row['samples'])
            before = bank.reads
            rejected = 0
            for name, indices in proposal.items():
                event = dict(tile=tile, band='full', factor=1, profile=None, condition='missing')
                try:
                    gate.request(role, event, code['commit'], positions=indices)
                except ValueError:
                    rejected += 1
            for bad in (dict(tile=tile, band='odd', factor=1), dict(tile=tile, band='full', factor=1)):
                requested_role = role if bad['band'] == 'odd' else 'confirmation'
                try:
                    gate.request(requested_role, bad, code['commit'])
                except ValueError:
                    rejected += 1
            require(rejected == 5 and bank.reads == before, 'Unadmitted view or role read payload')
            for band in ('full', 'even_retained_positions'):
                event = dict(tile=tile, band=band, factor=1, profile=None, condition='missing')
                start = time.perf_counter()
                actual, labels, points = gate.materialize(role, event, code['commit'])
                elapsed = time.perf_counter()-start
                roster.append(dict(scene=scene, role=role, tile=tile, band=band,
                    spectral=actual.receipt['spectral'], observation=actual.receipt['observation'],
                    positions=digest(actual.receipt['source']['positions']), band_metadata=actual.view.bands.fingerprint,
                    retained_count=len(actual.view.bands.band_ids), released_points=len(points.points),
                    released_set=points.summary()['point_set'], materialization_seconds=elapsed,
                    verified_historical_preflight=True))
            roles[role]['negative_requests_rejected_before_payload'] = rejected
        bands = identity['bands']
        evidence.append(dict(scene=scene, preflight={'path':str(path),'sha256':expected}, bank=bank.pin,
            publication=preflight['prior_production'], shape=identity['shape'], roles=roles,
            custody=identity['custody'], source=identity['source'],
            band_order=identity['band_order'], processed_column_ids=list(bands['band_ids']),
            original_sensor_column_mapping_authenticated=False, wavelength_quality=bands['wavelength_quality'],
            fwhm_present=bands['fwhm'] is not None,
            technical_sampling_API='Bank.materialize accepts sample configurations; this capability is not a new experimental view release',
            current_protocol_views=sorted({row['band'] for row in preflight['views']}),
            new_position_views_admitted=False, real_optimizer_release=False,
            new_position_proposal={k:list(v) for k,v in proposal.items()},
            permission_conclusion='Published development blocks and original roles available; new controlled positions are metadata-only proposals until separately admitted. Odd remains unreleased.'))
    table(args.out/'existing-provider-reads.csv', roster)
    table(args.out/'pre-score-identifiability.csv', qualification)
    dev._json_new(args.out/'source-role-authority.json', evidence)
    dev._json_new(args.out/'summary.json', dict(status='passed_existing_real_provider_and_pre_score_gates',
        producer=code['commit'], source_hash=code['source_hash'], admitted_existing_reads=len(roster),
        rejected_unadmitted_requests=20, rejected_request_payload_reads=0,
        proposal_representation_checks=len(qualification), new_real_views_materialized=0,
        real_optimizer_updates=0, classifier_forwards=0, raw_GT_opened=False, query_accessed=False,
        ready_for_real_training=False, block='separate controlled-view admission and optimizer registration required'))


if __name__ == '__main__':
    main()
