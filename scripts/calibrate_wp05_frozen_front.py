"""Measure one front-end population calibration on the four audited P4 cases."""
import argparse
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from evaluation import p4_bn_candidate as candidate, p4_bn_factors as bn
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, require_frozen_code, tensor_hash, write_json
from scripts.diagnose_wp05_bn_factors import protection
from scripts.diagnose_wp05_p4_fit import bounded_scene
from transfer import p4_coverage_batch as batch, p4_coverage_review as prior, p4_frozen_diagnostics as diagnostic
from transfer import p4_full_budget_audit as audit, p4_full_budget_campaign as campaign
from transfer import p4_full_budget_registration as registration, p4_independent_registration as scope
from transfer import p4_runtime_device as resource
from transfer.g3_training import _forward, state_hash
from transfer.p4_frozen_diagnostic_metrics import recompute
from transfer.p4_three_arm_contract import PROTECTION


SCHEMA = "wp05_p4_real_front_population_v1"
ROSTER = ("none", "anchor_only", "binary", "continuous")
STATES = ("initial", "selected", "last")
INPUTS = {
    "resource": ("wp05-independent-queue-real-prepared03/resource-scope.json",
                 "eb9cacef34c0cb5b10d654a52d8a8a6eccb645e4d4bd7908160ab2942cce9e43"),
    "release": ("wp05-independent-queue-real-authority01/release.json",
                "bec9247d9200234b5c88634f5be944c15c9e847a49f135f511f9ae6971358da7"),
    "independent": ("wp05-independent-queue-real-authority01/partial-audit02/review.json",
                    "a60134e6447550ce1d0455199a9b70f1f42648b860c61e87c72fe9ec0b8fa0fb"),
    "baseline": ("wp05-fit-sufficiency04/result/diagnostic.json",
                 "bbc23b724340a722306d6c2dd88f01bea8bbe78050984938c9bc7f59ebca21d6"),
}


def view_identity(tile, view, allowed, shape, expected_bands, plan_hash):
    obs = view.observation
    bn.require(allowed.get(tile["id"]) == tile and tile["partition"] == obs.partition == "train"
               and obs.plan_hash == plan_hash and view.bands.fingerprint == expected_bands,
               "Calibration requires original retained-band devtrain authority")
    y0,y1,x0,x1 = tile["box"]
    y,x = np.mgrid[y0:y1,x0:x1]
    coords = np.stack((y/max(shape[0]-1, 1), x/max(shape[1]-1, 1)), axis=-1).astype(np.float32)
    bn.require(np.array_equal(y*shape[1]+x, obs.global_ids) and np.array_equal(coords, obs.coordinates)
               and obs.spectral.shape == (*y.shape, len(view.bands.band_ids))
               and obs.spectral.dtype == np.float32 and np.isfinite(obs.spectral).all(),
               "Calibration coordinates or actual retained observations differ")
    return {"id": tile["id"], "box": tile["box"], "role": "devtrain", "plan_hash": plan_hash,
            "band_digest": view.bands.fingerprint, "spectral_digest": tensor_hash(obs.spectral),
            "coordinate_digest": tensor_hash(obs.coordinates), "pixel_digest": tensor_hash(obs.global_ids),
            "pixels": int(y.size)}


def check_views(contexts, expected, authority):
    rows = [view_identity(tile, view, **authority) for tile, view in contexts]
    bn.require(rows == expected and len({r["id"] for r in rows}) == len(rows) > 0,
               "Calibration context bytes, order or census differ")
    return [digest(row) for row in rows]


def fit(model, scene, contexts, expected, authority):
    pins = check_views(contexts, expected, authority)

    def forward(private, context):
        tile, view = context
        return _forward(private, scene, tile, observation_view=view)

    private, proof = candidate.calibrate_private(model, contexts, forward, role="devtrain", source_pins=pins)
    # Bind the fitted state to the exact normalized inputs before and after use.
    bn.require(check_views(contexts, expected, authority) == pins, "Calibration changed its inputs")
    base, calibrated = model.state_dict(), private.state_dict()
    bn.require(all(torch.equal(value, calibrated[key]) for key, value in base.items() if key not in proof["replacements"]),
               "Unexpected changes outside the two front running moments")
    bn.require(all(v["count"] == sum(r["pixels"] for r in expected) for v in proof["moments"].values()),
               "Population denominator differs from all authorized pixels")
    bn.require(all(torch.isfinite(t).all() for t in proof["replacements"].values()), "Nonfinite fitted population buffers")
    return private, proof


def load_base(root):
    items = {key: {"path": str(root / relative), "sha256": sha} for key, (relative, sha) in INPUTS.items()}
    res, reviewed, baseline = (scope.read(items[k]) for k in ("resource", "independent", "baseline"))
    sci = scope.validate(res, live_worker=True)
    scope.validate_release(items["resource"], res, items["release"])
    bn.require(reviewed["status"] == "partial" and reviewed["schema"] == "wp05_p4_real_selected_recompute_v1"
               and reviewed["resource"] == baseline["resource"] == items["resource"]
               and reviewed["release"] == baseline["release"] == items["release"]
               and reviewed["scientific_scope"] == res["science"] and reviewed["input_pins"] == sci["protected_files"]
               and reviewed["observed_cases"] == baseline["complete_cases"] == 4
               and tuple(r["case"]["arm"] for r in reviewed["cases"]) == ROSTER
               and [r["case"]["id"] for r in reviewed["cases"]] == [r["case"] for r in baseline["cases"]],
               "Expected the fixed four independently audited cases")
    return items, res, sci, reviewed, baseline


def resolve(root, code, loaded):
    items, res, sci, reviewed, baseline = loaded
    original = Path(res["execution"])
    sha = res["science"]["sha256"]
    src = sci["registration"]
    reg = scope.read({"path": src, "sha256": sci["protected_files"][src]})
    plan = scope.read({"path": reg["plan"], "sha256": reg["pins"][reg["plan"]]})
    path = reg["exposure"]["plan"]
    exposure = scope.read({"path": path, "sha256": reg["pins"][path]})
    geo, allowed, scene, frozen, image = bounded_scene(reg, plan, exposure, root)
    authority = {"allowed": {t["id"]: t for t in allowed["train"]}, "shape": scene.plan["shape"],
                 "expected_bands": frozen["records"]["train"]["selected_metadata_hash"],
                 "plan_hash": scene.plan["plan_hash"]}
    owner = campaign.read(original / "resource-owner.json")
    bn.require(owner["resource"] == items["resource"] and owner["release"] == items["release"],
               "Original P4 output owner differs")
    rows, payload = [], []
    source_cases = {c["id"]: c for c in sci["cases"]}
    for witness, old in zip(reviewed["cases"], baseline["cases"], strict=True):
        case = witness["case"]
        folder = campaign.bound_folder(original, case)
        bn.require(case == source_cases[case["id"]] and scope.pin(folder / "audit.json") == witness["audit"]
                   and image == old["bounded_observation"] and witness["audit_updates"] == 0,
                   "Wrong case ledger, model producer or bounded observation")
        win = campaign.accepted(folder, sci, case, sha)
        packet, *_ = audit.train_state(folder / "split", sci, reg, case, sha, win["numeric"], path_name="split")
        bn.require(packet["fingerprints"] == witness["paired"]["state"]
                   and packet["best"]["step"] == witness["paired"]["selected_step"] == old["selection_step"],
                   "Original selected checkpoint or independent fingerprints changed")
        model, init = registration.initialize_case(reg, case)
        bn.require(init == case["initialization"] == old["source_initialization"], "Wrong initial P4 state")
        model.eval()
        release = prior.released_roles(reg, plan, geo, allowed, case)
        role_data = {}
        for role, quota in (("train", case["k"]), ("val", 5)):
            layout, labels, population = registration.fixed_population(release[role], allowed, role, plan["shape"][1], quota)
            bn.require(population == old["roles"][role]["population"] == witness["populations"][role]
                       and [t["id"] for t, _ in layout] == old["roles"][role]["tiles"], "Released point population changed")
            condition = {"sampling": frozen["records"][role], "metadata": "nominal"}
            role_data[role] = diagnostic.fixed_views(scene, allowed, layout, condition, role), labels
        contexts = [(tile, view) for tile, _, view in role_data["train"][0]]
        fingerprints = [view_identity(tile, view, **authority) for tile, view in contexts]
        bn.require(len(fingerprints) == 22, "Actual fixed train-block census exceeds planned scope")
        source = {"case": case["id"], "k": case["k"], "arm": case["arm"], "selected_step": win["selected_step"],
                  "checkpoint": scope.pin(folder / "split/state.pt"), "audit": witness["audit"],
                  "population": witness["populations"], "contexts": fingerprints}
        rows.append(source)
        payload.append((source, model, packet, scene, role_data, contexts, old))
    count = sum(len(case["contexts"]) for case in rows)*len(STATES)*2
    evaluate = sum(sum(len(v) for v, _ in r[4].values()) for r in payload)*len(STATES)*2
    bn.require(count <= 528, "More population forwards than authorized")
    design = {"schema": SCHEMA, "scope": "real_frozen_p4_development_only",
              "producer": {k: code[k] for k in ("commit", "source_hash")}, "inputs": items, "cases": rows,
              "observation": image, "states": list(STATES), "comparison": ["original", "front_population"],
              "statistics_forward_limit": count, "evaluation_forward_limit": evaluate,
              "restore_evaluation_forward_limit": len(rows)*len(STATES),
              "statistics_buffer_write_limit": 4*len(rows)*len(STATES), "optimizer_updates": 0,
              "estimated_wall_seconds": [60, 180], "selection": "original_selection_frozen",
              "population": "all pixels of each registered devtrain tile; equal weight per pixel, correction 0",
              "protection": PROTECTION, "query_accessed": False, "raw_gt_loaded": False}
    return design, payload, authority


def readout(model, scene, role_data, arm):
    result = {}
    for role, (views, truth) in role_data.items():
        values, graphs = diagnostic.forward_rows(model, scene, views, arm)
        heads = {}
        for head, prob in values.items():
            metrics = diagnostic.score(prob, truth)
            check = recompute(prob.numpy(), truth)
            bn.require(all(metrics[k] == check[k] if k in ("confusion", "recall") else
                           abs(metrics[k]-check[k]) <= 1e-12 for k in ("oa", "aa", "kappa", "macro_f1", "nll", "recall", "confusion")),
                       "Independent P4 metric mismatch")
            support = np.bincount(truth, minlength=23)[1:]
            heads[head] = {"metrics": metrics, "support": support.tolist(), "probabilities": prob.tolist(),
                           "zero_recall": [i for i, recall in enumerate(metrics["recall"], 1) if recall == 0.]}
        result[role] = {"heads": heads, "graphs": graphs, "forwards": len(views), "labels": truth.tolist()}
    return result


def compare(current, base, k):
    def metrics(row):
        m = row["metrics"]
        return {"metrics": {**m, "per_class": [{"recall": value} for value in m["recall"]]}}
    output = {}
    for role in current:
        now, old = current[role], base[role]
        output[role] = {"protection": protection({h: metrics(v) for h, v in now["heads"].items()},
                                                  {h: metrics(v) for h, v in old["heads"].items()}, k),
                        "graph_tiles_changed": sum(a["q_sha256"] != b["q_sha256"] or a["a_sha256"] != b["a_sha256"]
                                                  for a, b in zip(now["graphs"], old["graphs"], strict=True))}
    return output


def execute(output, design_pin, design, payload, authority):
    started = perf_counter()
    output.mkdir()
    write_json(output / "claim.json", {"registration": design_pin, "optimizer_updates": 0})
    result = []
    for case, model, packet, scene, data, training, old in payload:
        original_weights = {k: v.clone() for k, v in model.state_dict().items()}
        row = {"case": case["case"], "checkpoint": case["checkpoint"], "selected_step": case["selected_step"], "states": {}}
        for name, weights in (("initial", original_weights), ("selected", packet["best"]["model"]), ("last", packet["model"])):
            tick = perf_counter()
            model.load_state_dict(weights, strict=True)
            original = state_hash(model)
            base = readout(model, scene, data, case["arm"])
            for role, value in base.items():
                for head, reading in value["heads"].items():
                    prior_metrics = old["frozen_cpu_readout"][name][role]["metrics"][head]
                    bn.require(reading["metrics"]["confusion"] == prior_metrics["confusion"]
                               and abs(reading["metrics"]["nll"]-prior_metrics["nll"]) < 1e-10,
                               "Baseline differs from already audited bounded readout")
            clone, proof = fit(model, scene, training, case["contexts"], authority)
            calibrated = readout(clone, scene, data, case["arm"])
            proof_file = output / f'{case["arm"]}-{name}-buffers.pt'
            with proof_file.open("xb") as stream:
                torch.save(proof, stream)
            saved = scope.pin(proof_file)
            restored, restored_result = candidate.restore_private(model, torch.load(proof_file, weights_only=True),
                                                                  check_views(training, case["contexts"], authority))
            first_view = data["train"][0][:1]
            # Exact graph and outputs after round-trip without any calibration forward.
            references, ref_graph = diagnostic.forward_rows(restored, scene, first_view, case["arm"])
            bn.require(state_hash(restored) == state_hash(clone) and ref_graph[0] == calibrated["train"]["graphs"][0]
                       and all(torch.equal(values, torch.tensor(calibrated["train"]["heads"][head]["probabilities"][:len(values)]))
                               for head, values in references.items())
                       and state_hash(model) == original, "Calibration recovery or protected source differs")
            row["states"][name] = {"original": base, "candidate": calibrated, **compare(calibrated, base, case["k"]),
                                    "buffers": saved, "proof": {k: v for k, v in proof.items() if k != "replacements"},
                                    "restore": {**restored_result, "evaluation_forwards": 1}, "seconds": perf_counter()-tick}
        bn.require(scope.pin(case["checkpoint"]["path"]) == case["checkpoint"], "Original checkpoint bytes changed")
        result.append(row)
    all_states = [s for c in result for s in c["states"].values()]
    stats = sum(s["proof"]["statistics_context_forwards"] for s in all_states)
    evaluation = sum(a[role]["forwards"] for s in all_states for a in (s["original"], s["candidate"]) for role in a)
    buffers = sum(s["proof"]["statistics_buffer_writes"] for s in all_states)
    bn.require((stats, evaluation, buffers, len(all_states)) == (design["statistics_forward_limit"],
               design["evaluation_forward_limit"], design["statistics_buffer_write_limit"],
               design["restore_evaluation_forward_limit"]), "Real evaluation census differs")
    selected_protected = all(x["passes"] for row in result for x in row["states"]["selected"]["val"]["protection"].values())
    selected_fused_help = all(row["states"]["selected"]["candidate"]["val"]["heads"]["fused"]["metrics"]["nll"] <
                              row["states"]["selected"]["original"]["val"]["heads"]["fused"]["metrics"]["nll"] for row in result)
    path = output / "calibration.json"
    write_json(path, {"schema": SCHEMA, "scope": design["scope"], "status": "measured",
                      "producer": design["producer"], "registration": design_pin, "cases": result,
                      "statistics_forwards": stats, "private_buffer_writes": buffers, "evaluation_forwards": evaluation,
                      "restore_evaluation_forwards": len(all_states), "restore_refits": 0,
                      "selected_all_heads_protected": selected_protected, "selected_fused_nll_improved_all": selected_fused_help,
                      "optimizer_updates": 0, "original_state_updates": 0, "validation_statistics_fit": False,
                      "raw_gt_loaded": False, "query_accessed": False, "confirmation_accessed": False,
                      "production_default_changed": False, "seconds": perf_counter()-started})
    return scope.pin(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("action", choices=("register", "evaluate"))
    parser.add_argument("--registration-sha256")
    args = parser.parse_args()
    code = code_snapshot()
    require_frozen_code(code)
    root, out = args.runs.resolve(strict=True), external_directory(args.out)
    bn.require(root in out.parents and code["commit"] == args.expected_commit, "Wrong experiment location or code")
    torch.set_num_threads(1)
    loaded = load_base(root)
    items, resources, science, *_ = loaded
    with resource.lease(Path(resources["execution"]) / ".coordinator.lock"):
        design, data, authorized = resolve(root, code, loaded)
        named = out / "registration.json"
        if args.action == "register":
            bn.require(args.registration_sha256 is None, "Register before pinning a case")
            out.mkdir()
            write_json(named, design)
            result = scope.pin(named)
        else:
            item = {"path": str(named), "sha256": args.registration_sha256}
            bn.require(scope.read(item) == design, "Registration no longer matches the actual context or source")
            result = execute(out / "result", item, design, data, authorized)
        batch.check_pins(science["protected_files"])
        for pin in items.values():
            scope.read(pin)
    require_frozen_code(code)
    print(result, flush=True)


if __name__ == "__main__":
    main()
