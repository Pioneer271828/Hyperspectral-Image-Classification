"""Replay just the missing same-point decisions from the frozen PaviaU producers."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

PRODUCERS = {"wp07":"ca22ac5ae52a4c1893f1bb340f1514be6eda6651",
             "wp08":"a67f6af84742d51da8d4a1abaf6808e91f946e4e"}
AUDITS = {
    "wp07": ("wp07-paviau-real-results-20261118.TC3eXtC5/aggregate/results/comparison-audit.json",
             "728076ffe4751662cea2580e66c3b915d3fc5b30cecc1308ff1d761a3b8cc1bb"),
    "wp08": ("wp08-paviau-gpu5-rerun-20261118/audit/comparison-audit.json",
             "aa786164e85638cda6b57b4b1af3c0708dd0db2b10c526aeff400e0d98cc061c"),
}
HEADS = ("fused","graph","cnn")
VIEWS = ("full","even_retained_positions")


def require(test, message):
    if not test:
        raise ValueError(message)


def get_code(root):
    def git(*args):
        return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()

    require(not git("status","--porcelain"),"The cross-backend consumer checkout is dirty")
    return git("rev-parse","HEAD")


def load_cases(root, kind, code):
    from transfer import native_released_campaign as campaign
    from transfer.native_scale_campaign import read

    name, sha = AUDITS[kind]
    audit_pin = {"path":str(root/name),"sha256":sha}
    audit = read(audit_pin,root)
    matrix, proofs = campaign.load(root,audit["matrix"],code,audit["release"],engineering=False)
    require(not audit["raw_gt_loaded"] and audit["audit_optimizer_updates"] == 0 and not audit["query_authorized"],
            "Historic record is not a released PaviaU development audit")
    if kind == "wp08":
        previous = {case["case"]:case for case in audit["cases"]}
    else:
        previous = {record["case"]:record for pin in audit["case_replays"] for record in [read(pin,root)]}
        plan = read(audit["plan"],root)
        require(plan["query_authorized"] is plan["raw_gt_authorized"] is False,
                "WP07 migration must bind the original development-only roles")
        expected = {case["id"] for case in matrix["cases"] if case["id"].split("/")[1] in plan["seed_gpus"]}
        require(expected == set(plan["case_releases"]), "Scope of the physical cohort migration differs")
        for key, link in plan["case_releases"].items():
            seed = key.split("/")[1]
            require(read(link,root) == {**read(proofs[key],root),"physical_uuid":plan["seed_gpus"][seed]},
                    "Physical migration changed a registered PaviaU input")
            proofs[key] = link
    require(set(previous) == {case["id"] for case in matrix["cases"]} == proofs.keys(),
            "Original replay and accepted matrix do not have the same case roster")
    return matrix["cases"], proofs, previous, audit_pin


def predictions(guard, spec, model, event, execution, producer):
    from transfer.native_scale_observation import forward

    actual, labels, targets = guard.materialize("devval",event,producer)
    guard.validate_observation(spec,actual,labels,targets,event)
    if not len(targets.points):
        return [],[]
    condition = guard.selection_condition(actual,spec) if hasattr(guard,"selection_condition") else None
    values, _ = forward(model,actual,producer,"devval",condition=condition,execution=execution,
                        graph_scale=spec.get("registered_graph_scale"))
    ids = [int(key) for key,_ in targets.points]
    truth = [int(y) for _,y in targets.points]
    items = {head:(value[list(targets.cells)].argmax(1)+1).tolist()
             for head,value in zip(HEADS,values[:3],strict=True)}
    return ids,[(target,*(items[h][i] for h in HEADS)) for i,target in enumerate(truth)]


def replay(root, kind, case, proof, expected, code, execution):
    from transfer import native_released_campaign as campaign, registered_development as storage
    from transfer.g3_training import state_hash
    from transfer.native_scale_campaign import digest, read
    from transfer.native_scale_training import initialize

    guard, place = campaign.authorized_case(root,case,proof,code,intake=None)
    registration = storage._pin(place/"registration.json")
    spec = read(registration,root)
    pin, receipt, packet = campaign.receipt(root,place,"resume",registration)
    require(pin == expected["stages"][1] and receipt["updates"] == 30
            and packet["registration"] == digest(spec) and packet["cursor"] == case["updates"] == 60
            and packet["selected"]["step"] == expected["selected_step"]
            and packet["selected"]["step"] == min(packet["validation"],key=lambda row:row["fused_nll"])["step"],
            "PaviaU weight recovery disagrees with its archived continuation or first-minimum")
    source = guard.baseline(spec["seed"])
    if kind == "wp08":
        model = guard.case_model(source,spec["arm"],spec["module_seed"])
    else:
        model = initialize(source,spec["arm"],spec["module_seed"])
    model.to(execution.device).eval().requires_grad_(False)
    model.backbone.adapter.cache_attention = False
    endpoints, reused = {}, {}
    for phase,state in (("selected",packet["selected"]["state"]),("last",packet["model"])):
        model.load_state_dict(state,strict=True)
        identity = state_hash(model)
        if identity in reused:
            endpoints[phase] = reused[identity]
            continue
        data = {view:{} for view in VIEWS}
        for event in spec["validation"]:
            require(event["band"] in VIEWS and event["factor"] == spec["factor"], "Unexpected archived observation")
            keys, rows = predictions(guard,spec,model,event,execution,code["commit"])
            items = data[event["band"]]
            require(not set(items).intersection(keys),"Duplicate released PaviaU point")
            items.update(zip(keys,rows,strict=True))
        require(all(len(v)==45 and Counter(row[0] for row in v.values())=={i:5 for i in range(1,10)}
                    for v in data.values()), "PaviaU replay changed the released class population")
        require(all({i:row[0] for i,row in data["full"].items()} == {i:row[0] for i,row in data[view].items()}
                    for view in VIEWS), "Band views did not reuse the same released points")
        endpoints[phase] = reused[identity] = data
    rows = {}
    for phase,data in endpoints.items():
        compare = expected["head_metrics"] if phase=="selected" else expected[
                                  "terminal_head_metrics" if kind=="wp08" else "last_head_metrics"]
        rows[phase] = {}
        for band,table in data.items():
            ordered = [table[i] for i in sorted(table)]
            matrices = {}
            for index,head in enumerate(HEADS,1):
                matrix = [[0]*9 for _ in range(9)]
                for item in ordered:
                    matrix[item[0]-1][item[index]-1] += 1
                matrices[head] = matrix == compare[head][band]["confusion_matrix"]
            rows[phase][band] = dict(point_order=digest(sorted(table)),values=ordered,
                                     confusion_match=matrices)
    return dict(checkpoint=receipt["checkpoint"],registration=registration,rows=rows,
                seed=spec["seed"],factor=spec["factor"],steps=spec["steps"],
                selected=packet["selected"]["step"],device=spec["numeric"]["physical_uuid"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--producer-checkout",type=Path,required=True)
    parser.add_argument("--archive", choices=tuple(PRODUCERS),required=True)
    parser.add_argument("--expected-commit",required=True)
    args = parser.parse_args()
    consumer = Path(__file__).resolve().parents[1]
    head = get_code(consumer)
    original = args.producer_checkout.resolve(strict=True)
    root, out = args.runs.resolve(strict=True), args.out.resolve()
    require(head == args.expected_commit and root in out.parents and not out.exists()
            and get_code(original) == PRODUCERS[args.archive], "Expected frozen checkouts and a new external receipt")
    sys.path.insert(0,str(original))
    import torch
    from research_runtime import code_snapshot, require_frozen_code
    from scripts.run_native_released_points import arithmetic
    from transfer.registered_development import _json_new

    code = code_snapshot(original)
    require_frozen_code(code)
    cases, proofs, previous, audit = load_cases(root,args.archive,code)
    started, result = perf_counter(), {}
    with arithmetic(root,None) as execution, torch.inference_mode():
        for case in cases:
            key = case["id"]
            result[key] = replay(root,args.archive,case,proofs[key],previous[key],code,execution)
            print(f"recovered {key}",flush=True)
    require_frozen_code(code)
    require(get_code(consumer)==head,"PaviaU replay consumer changed during execution")
    print(json.dumps(_json_new(out,dict(schema="wp12_paviau_archive_point_replay_v1",audit=audit,archive=args.archive,
        source=code["commit"],consumer=head,cases=result,numeric="cpu_frozen_replay",
        query_accessed=False,raw_gt_loaded=False,optimizer_updates=0,seconds=perf_counter()-started))),flush=True)


if __name__=="__main__":
    main()
