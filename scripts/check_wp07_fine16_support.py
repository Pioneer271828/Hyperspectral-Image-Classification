"""Independent integer-bitset reach, positive-weight precision, and prior pilot replay."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, os.environ['WP07_GRAPH_SOURCE'])
import numpy as np
import torch
from transfer.native_scale_campaign import read, packet
from transfer import registered_development as dev
from transfer.wp07_graph_range import require, require_checks


def bits(tensor):
    value = tensor.coalesce()
    result = [0]*value.shape[0]
    for a, b, v in zip(*value.indices().tolist(), value.values().tolist()):
        if v > 0:
            result[a] |= 1 << b
    return result


def compose(left, right):
    result = []
    for mask in left:
        merged = 0
        while mask:
            one = mask & -mask
            merged |= right[one.bit_length()-1]
            mask -= one
        result.append(merged)
    return result


def numeric_bits(value):
    packed = np.packbits(value.numpy() > 0, axis=1, bitorder='little')
    return [int.from_bytes(row.tobytes(), 'little') for row in packed]


def analyze(entry):
    q, a, b = (entry[k].coalesce() for k in ('q', 'near', 'wide'))
    require(all(t.dtype == torch.float32 for t in (q, a, b)), 'Source precision differs')
    qr, qt, ar, br = bits(q), bits(q.T), bits(a), bits(b)
    aa, bb = compose(ar, ar), compose(br, br)
    near, wide = compose(compose(qr, aa), qt), compose(compose(qr, bb), qt)
    Q, A, B = (t.to_dense().double() for t in (q, a, b))
    R = Q.T/Q.sum(0).clamp_min(1e-6)[:, None]
    X, Y = (Q@(A@A))@R, (Q@(B@B))@R
    require(near == numeric_bits(X) and wide == numeric_bits(Y), 'Integer and numeric support differ')
    require(all((x & ~y) == 0 for x, y in zip(near, wide)), 'Near support is not contained in wide')
    changed = sum(x != y for x, y in zip(near, wide))
    require(changed > 0, 'Degenerate graph-input pixel-feature coverage')
    require(float((X.sum(1)-1).abs().max()) < 1e-5 and float((Y.sum(1)-1).abs().max()) < 1e-5, 'Mass differs')
    return {'native_cells': q.shape[0], 'regions': q.shape[1],
        'region_twohop_support_changed_rows': sum(x != y for x, y in zip(aa, bb)),
        'Q_twohop_native_footprint_changed_cells': changed,
        'near_min_count': min(v.bit_count() for v in near), 'near_max_count': max(v.bit_count() for v in near),
        'wide_min_count': min(v.bit_count() for v in wide), 'wide_max_count': max(v.bit_count() for v in wide),
        'minimum_positive_proxy_weight': min(float(X[X>0].min()), float(Y[Y>0].min())),
        'max_linear_proxy_weight_delta': float((X-Y).abs().max()),
        'mean_linear_proxy_row_l1_delta': float((X-Y).abs().sum(1).mean())}


def main():
    require_checks()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    decision = read(dev._pin(args.geometry), args.root)
    with args.pilot.open(newline='') as stream:
        pilot = {(r['protocol'], r['role'], r['tile'], r['band']): r for r in csv.DictReader(stream)}
    rows = []
    for scene, count in [('paviaU', 336), ('xuzhou', 156)]:
        protocol = read(decision['protocols'][f'{scene}-factor1-scale16'], args.root)
        cache = packet(protocol['cache'], args.root)
        require(len(cache['entries']) == count, 'View denominator differs')
        for key, entry in cache['entries'].items():
            role, tile, band = key.split('/')
            value = analyze(entry)
            prior = pilot[f'{scene}-factor1-scale16-diagnostic', role, tile, band]
            for name in ['native_cells', 'region_twohop_support_changed_rows', 'Q_twohop_native_footprint_changed_cells']:
                require(value[name] == int(prior[name]), 'Original initial-geometry pilot differs')
            require(abs(value['max_linear_proxy_weight_delta']-float(prior['linear_proxy_max_abs_weight_delta'])) < 1e-12,
                    'Original linear proxy response differs')
            rows.append({'scene': scene, 'role': role, 'tile': int(tile), 'band': band, **value})
    args.out.mkdir()
    with (args.out/'views.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'status': 'passed', 'views': len(rows), 'geometry': dev._pin(args.geometry),
        'pilot_csv_sha256': hashlib.sha256(args.pilot.read_bytes()).hexdigest(),
        'program_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'strict_positive_threshold': 0, 'source_dtype': 'float32', 'analysis_dtype': 'float64',
        'float64_normal_min': torch.finfo(torch.float64).tiny,
        'minimum_positive_proxy_weight': min(r['minimum_positive_proxy_weight'] for r in rows),
        'labels_scored': 0, 'classifier_heads_evaluated': 0, 'optimizer_updates': 0,
        'scope': 'graph-input pixel-feature path support and scalar linear proxy, not full-network raw-input receptive field',
        'elapsed_seconds': time.perf_counter()-started}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
