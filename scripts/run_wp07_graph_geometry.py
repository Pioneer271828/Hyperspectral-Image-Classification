"""Enumerate existing WP07 inputs without evaluating labels or classifier heads."""
import argparse
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from research_runtime import code_snapshot, require_frozen_code
from transfer import wp07_graph_range as graph, registered_development as dev
from transfer.development_bank import _atomic_new
from transfer.development_block_probe import VIEWS
from transfer.g3_training import state_hash
from transfer.native_released_real import ReleasedBankValidation, registered_case
from transfer.native_scale_campaign import read
from transfer.native_scale_observation import from_bank


def enumerate_geometry(root, out, code):
    graph.require_checks()
    require_frozen_code(code)
    graph.require(root in out.parents and not out.exists(), "Use a new external geometry directory")
    sources = {"paviaU": root / "wp02-paviau-final-nov18/preflight/paviaU.json",
               "xuzhou": root / "wp07-native-duty01/duties-preflight/xuzhou.json"}
    out.mkdir()
    records, manifests = [], {}
    # This order is frozen before any enumeration or new score access.
    protocol_order = [("paviaU", 1, 50), ("xuzhou", 1, 100), ("paviaU", 2, 50), ("xuzhou", 2, 100)]
    for scene, factor, scale in protocol_order:
        source = dev._pin(sources[scene])
        receipt = read(source, root)
        case = registered_case(receipt, f"{scene}/64/1/off", engineering=False)
        admitted = ReleasedBankValidation(root, receipt, case)
        baseline = admitted.baseline(64)
        model = graph.initialize(baseline, 64+7321)
        initial = state_hash(model)
        items, entries = [], {}
        for role in ("devtrain", "devval"):
            # Only signed role/tile identities are consumed, never samples,
            # targets, the preflight's stored scores, or classifier outputs.
            for row in admitted._bank.role_rows(role):
                tile = row["id"]
                for band, spec in VIEWS.items():
                    actual, _ = from_bank(admitted._bank, role, tile, spec, factor, code["commit"])
                    entry = graph.capture_geometry(model, actual, code["commit"], role, scale)
                    key = f"{role}/{tile}/{band}"
                    comparison = graph.compare_supports(entry["near"], entry["wide"])
                    item = {"role": role, "tile": tile, "band": band, "factor": factor, "scale": scale,
                            "q": graph.sparse_digest(entry["q"]), "grid": list(entry["grid"]),
                            "identity": entry["identity"], "comparison": comparison,
                            "near": graph.support_report(entry["near"], entry["grid"], entry["image_shape"], factor),
                            "wide": graph.support_report(entry["wide"], entry["grid"], entry["image_shape"], factor)}
                    items.append(item)
                    entries[key] = entry
        graph.require(state_hash(model) == initial, "Geometry changed the initial model")
        role_counts = {role: Counter(str(item["comparison"]["identifiable"]) for item in items if item["role"] == role)
                       for role in ("devtrain", "devval")}
        eligible = all(count.get("True", 0) > 0 for count in role_counts.values())
        name = f"{scene}-factor{factor}-scale{scale}"
        cache = out / (name + ".pt")
        with BytesIO() as buffer:
            torch.save({"entries": entries, "source": source, "bank": receipt["bank"], "scale": scale,
                        "factor": factor, "baseline_state": state_hash(baseline)}, buffer)
            _atomic_new(cache, buffer.getvalue())
        manifests[name] = dev._json_new(out / (name + ".json"), {
            "scene": scene, "factor": factor, "scale": scale, "preflight": source,
            "cache": dev._pin(cache), "role_counts": role_counts, "eligible": eligible,
            "all_views_distinct": all(x["comparison"]["identifiable"] for x in items), "views": items})
        records.append({"protocol": name, "eligible": eligible,
                        "all_views_distinct": all(x["comparison"]["identifiable"] for x in items)})
        print(json.dumps({"protocol": name, "role_counts": role_counts, "eligible": eligible}), flush=True)
    # Prefer an input where every view distinguishes the supports; otherwise
    # retain equivalent views in all scoring denominators and label them.
    choices = [r for r in records if r["all_views_distinct"]] or [r for r in records if r["eligible"]]
    chosen = choices[0]["protocol"] if choices else None
    result = dev._json_new(out / "geometry.json", {
        "schema": "wp07_b_label_free_graph_geometry_v1", "producer": code["commit"],
        "source_hash": code["source_hash"], "protocol_order": [r["protocol"] for r in records],
        "criterion": "first all-views-distinct protocol; otherwise first with distinct train and validation views",
        "selected_protocol": chosen, "protocols": manifests, "labels_scored": 0,
        "classifier_heads_evaluated": 0, "optimizer_updates": 0, "query_accessed": False, "raw_gt_loaded": False})
    return result


def main():
    graph.require_checks()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    code = code_snapshot()
    graph.require(code["commit"] == args.expected_commit, "Unexpected geometry source")
    torch.set_num_threads(1)
    print(json.dumps(enumerate_geometry(args.root.resolve(strict=True), args.out.resolve(), code)), flush=True)


if __name__ == "__main__":
    main()
