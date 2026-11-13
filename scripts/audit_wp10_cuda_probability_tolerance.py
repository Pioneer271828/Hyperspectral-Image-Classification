"""Independently recompute CUDA repeat tolerance from the authenticated full arrays."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from research_runtime import external_directory, hash_file, tensor_hash, write_json
from transfer import matched_route_admission as admission


HEADS = ("fused", "graph", "cnn")


def audit(pin):
    receipt = admission.read(pin)
    data = receipt["repeat_probability_arrays"]
    path = Path(data["path"])
    if path != Path(pin["path"]).parent / "repeat-probabilities.pt" or hash_file(path) != data["sha256"]:
        raise ValueError("Raw comparison arrays differ")
    pairs = torch.load(path, map_location="cpu", weights_only=True)
    expected = {f"{name}/{phase}" for name, row in receipt["cases"].items()
                for phase in ("repeat", "post_timing") if phase in row}
    if pairs.keys() != expected or len(expected) != 38 or len(receipt["cases"]) != 28:
        raise ValueError("Original 28-case comparison matrix required")
    results = {}
    for key, pair in pairs.items():
        name, phase = key.split("/")
        baseline = receipt["cases"][name]
        repeat = baseline[phase]
        summary = {}
        for head_index, head in enumerate(HEADS):
            report = dict(elements=0, exceeded=0, max_abs=0., maximum_ratio=0.)
            for index, (old, new) in enumerate(zip(pair["reference"], pair["actual"], strict=True)):
                ref, got = old[head_index], new[head_index]
                if (tensor_hash(ref) != baseline["output_hashes"][index][head]
                        or tensor_hash(got) != repeat["output_hashes"][index][head]):
                    raise ValueError("Repeat arrays are not the recorded output")
                a, b = ref.numpy().astype(np.float64), got.numpy().astype(np.float64)
                if a.shape != (784, 9) or b.shape != a.shape or not np.isfinite([a, b]).all():
                    raise ValueError("Unexpected five-block V5 output")
                close = np.isclose(b, a, atol=0.000001, rtol=0.00001, equal_nan=False)
                delta = np.abs(b-a)
                report["elements"] += a.size
                report["exceeded"] += np.count_nonzero(~close)
                report["max_abs"] = max(report["max_abs"], float(delta.max()))
                report["maximum_ratio"] = max(report["maximum_ratio"], float((delta/(0.000001+0.00001*np.abs(a))).max()))
            report["exceeded"] = int(report["exceeded"])
            if report != repeat["tolerance"][head] or report["elements"] != 3920*9:
                raise ValueError("Independent elementwise results differ")
            summary[head] = report
        results[key] = summary
    return dict(status="passed_independent_audit", pairs=len(results),
                elements=sum(head["elements"] for row in results.values() for head in row.values()),
                exceeded=sum(head["exceeded"] for row in results.values() for head in row.values()),
                max_ratio=max(head["maximum_ratio"] for row in results.values() for head in row.values()),
                max_abs=max(head["max_abs"] for row in results.values() for head in row.values()),
                receipt=pin, raw_arrays=data, rows=results)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", nargs=2, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = external_directory(args.out)
    if out.exists():
        raise FileExistsError(out)
    write_json(out, audit(dict(zip(("path", "sha256"), args.receipt, strict=True))))
