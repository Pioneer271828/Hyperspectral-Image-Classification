"""Standard-library-only cost helpers for additive historical-runtime workers.

Keeps the worker independent of post-training inference/metric modules. Timing
is the same inclusive-quantile reducer used by evaluation.dynamic_curve.
"""

import hashlib
import json
import math
from pathlib import Path
import statistics


def pin(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def fields(value, names):
    if not isinstance(value, dict) or value.keys() != set(names.split()):
        raise ValueError("Expected fields: " + names)


def read_pinned(path, expected_hash):
    data = Path(path).read_bytes()
    if not pin(expected_hash) or hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("Input SHA-256 differs: " + str(path))
    return json.loads(data)


def timing_summary(samples, *, repeats, blocks):
    if (not isinstance(samples, list) or len(samples) != repeats or repeats < 2
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in samples)):
        raise ValueError("Complete finite positive timing repeats required")
    return dict(wall_ms=list(samples), p50_ms=statistics.median(samples),
                p95_ms=statistics.quantiles(samples, n=100, method="inclusive")[94],
                mean_ms=statistics.mean(samples), sample_sd_ms=statistics.stdev(samples),
                blocks_per_second=blocks * 1000 / statistics.median(samples))
