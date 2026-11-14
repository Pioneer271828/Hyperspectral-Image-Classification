"""Verify the registered equal-capacity controls through native Run phases."""
import argparse
from pathlib import Path

import torch

from research_runtime import code_snapshot
from transfer import native_released_capacity as capacity, native_released_training as data
from transfer import native_scale_training as training, registered_development as receipt
from transfer.native_scale_campaign import packet
from transfer.xuzhou_multiscale_audit import _exact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    assert code["commit"] == args.expected_commit and not code["dirty"]
    torch.set_num_threads(1)
    args.out.mkdir()
    rows = []
    for classes, factor in ((9, 2), (22, 4)):
        for arm, intervention in (("joint", "matched"), ("joint", "missing"), (capacity.CONSTANT, capacity.CONSTANT)):
            spec = capacity.experiment(classes, 64, arm, factor, intervention, code["commit"])
            folder = args.out / f"{classes}-{arm}-{intervention}"
            folder.mkdir()
            receipts, states = {}, {}
            for phase in ("full", "prefix", "resume", "completed"):
                prior = states.get({"resume": "prefix", "completed": "resume"}.get(phase))
                path = folder / f"{phase}.pt"
                run = training.execute(spec, phase, path, code["commit"], prior, released=data)
                receipts[phase] = {**run, "state": receipt._pin(path)}
                states[phase] = packet(receipts[phase]["state"], args.out)
            assert _exact(states["full"], states["resume"]) and _exact(states["full"], states["completed"])
            assert receipts["completed"]["updates"] == 0
            keys = [name for name in spec["initialization"]["fresh"] if "condition_encoder" in name or "range_encoder" in name]
            assert keys
            observed = [step["fresh"][name] for step in states["full"]["history"] for name in keys]
            if intervention == "missing":
                assert all(item["gradient_l1"] is None and item["changed"] is False for item in observed)
            else:
                assert any(item["gradient_l1"] and item["changed"] for item in observed)
            rows.append({"classes": classes, "arm": arm, "intervention": intervention, "factor": factor,
                         "parameters": spec["parameters"], "paired_updates": 12, "stages": receipts,
                         "selected_step": states["full"]["selected"]["step"], "condition_updates": observed})
    print(receipt._json_new(args.out / "capacity-recovery.json", {
        "schema": capacity.SCHEMA, "producer": code["commit"], "synthetic": True,
        "rows": rows, "optimizer_updates": 72, "real_optimizer_updates": 0, "query_authorized": False}), flush=True)


if __name__ == "__main__":
    main()
