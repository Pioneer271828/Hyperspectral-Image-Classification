"""Trace the fixed PaviaU class 2/6 guard through the frozen readout components."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from evaluation.frozen_band_region import HEADS
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_frozen_band_region import _read, _write, check, pin
from scripts.run_frozen_mechanisms import read_bundle
from transfer.support_readout import SupportReadout, apply_distance, distance

FOILS = {2: 6, 6: 2}


def audit(folder, output, producer, expected_receipt):
    wp12 = pin(folder / "wp12/receipt.json", expected_receipt)
    receipt = _read(wp12)
    check(receipt["status"] == "accepted_development_only" and receipt["real_updates"] == 0
          and receipt["raw_gt_loaded"] is receipt["query_accessed"] is False,
          "Expected the exact accepted readout receipt")
    cells = []
    breakdown = Counter()
    for case, sha in receipt["case_receipts"].items():
        if not case.startswith("paviaU__"):
            continue
        path = folder / case
        case_receipt = _read(pin(path / "receipt.json", sha))
        result = case_receipt["result"]
        check(result["model"]["seed"] in (64, 512, 2048), "Unexpected model seed")
        _, arrays = read_bundle(pin(path / "arrays.npz", case_receipt["arrays"]))
        labels, aliases = (result["population"][name] for name in ("labels", "aliases"))
        for window in result["windows"]:
            features = {h: arrays[f"{window}__validation_features__{h}"] for h in HEADS}
            base = {h: arrays[f"{window}__native__{h}"] for h in HEADS}
            for group in result["groups"]:
                key = f"{window}__{group}"
                item = result["witnesses"][key]
                model = SupportReadout.from_state(torch.load(pin(path / f"{key}.pt", item["sha256"])["path"],
                                                            weights_only=True, map_location="cpu"), item["binding"])
                distances = {h: distance(model.centers[h], features[h]) for h in HEADS}
                alone = {h: apply_distance(base[h], distances[h], model.counts, model.scales[h]) for h in HEADS}
                pooled = model.weights[0] * alone["cnn"] + model.weights[1] * alone["graph"]
                final = arrays[f"{key}__candidate__fused"]
                check(np.allclose(final, ((alone["fused"] + pooled) * .5).numpy(), atol=2e-12, rtol=2e-12),
                      "Frozen component readout does not reproduce WP12")
                new = (("native", base["fused"]), ("centroid", arrays[f"{key}__centroid__fused"]),
                       ("single_head", alone["fused"].numpy()), ("candidate", final))
                for index, (alias, label) in enumerate(zip(aliases, labels, strict=True)):
                    if label not in FOILS:
                        continue
                    rival = FOILS[label]
                    pred = {arm: int(p[index].argmax()) + 1 for arm, p in new}
                    native_ok, single_ok, combined_ok = (pred[arm] == label for arm in ("native","single_head","candidate"))
                    status = ("preserved" if combined_ok else "lost") if native_ok else (
                        "gained" if combined_ok else "both_wrong")
                    source = ("changed_in_single_head" if not single_ok else "changed_at_branch_pool"
                              ) if status == "lost" else "other"
                    breakdown[f"{group}/{status}/{source}"] += 1
                    cells.append(dict(case=case, window=window, support=group, alias=alias, true_class=label,
                        foil=rival, status=status, loss_source=source, **pred,
                        support_true=int(model.counts[label-1]), support_foil=int(model.counts[rival-1]),
                        cnn_weight=float(model.weights[0]), graph_weight=float(model.weights[1]),
                        cnn_candidate=int(alone["cnn"][index].argmax()) + 1,
                        graph_candidate=int(alone["graph"][index].argmax()) + 1,
                        fused_pair_distance=float(distances["fused"][index,rival-1]-distances["fused"][index,label-1]),
                        graph_pair_distance=float(distances["graph"][index,rival-1]-distances["graph"][index,label-1]),
                        cnn_pair_distance=float(distances["cnn"][index,rival-1]-distances["cnn"][index,label-1]),
                        native_margin=float(base["fused"][index,label-1]-base["fused"][index,rival-1]),
                        single_margin=float(alone["fused"][index,label-1]-alone["fused"][index,rival-1]),
                        branch_margin=float(pooled[index,label-1]-pooled[index,rival-1]),
                        candidate_margin=float(final[index,label-1]-final[index,rival-1])))
    check(len(cells) == 360, "Expected 3 seeds × 4 windows × 3 support views × 10 protected points")
    output.mkdir()
    path = output / "paviau-class2-6-attribution.csv"
    with path.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, tuple(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    return _write(output / "receipt.json", dict(receipt=wp12["sha256"], producer=producer, cases=3, rows=len(cells),
                   artifact=pin(path)["sha256"], counts=dict(breakdown), real_updates=0,
                   query_accessed=False, raw_gt_loaded=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    check(code["commit"] == args.expected_commit, "Unexpected review checkout")
    root = external_directory(args.directory)
    print(audit(root, root / "paviau-review", code["commit"], args.receipt))
    require_frozen_code(code)


if __name__ == "__main__":
    main()
