"""Review previously released comparisons without training or joining dissimilar protocols."""

from collections import Counter
import csv
from math import isclose
from time import perf_counter

import numpy as np
import torch

from evaluation.classification_metrics import metrics_from_confusion
from evaluation.frozen_training_stages import HEADS, json_pin, pinned, require
from evaluation.wp12_cross_backend import raw_pair
from loadData.spatial_tiles import observe_tile
from model.research_config import digest
from research_runtime import require_frozen_code, tensor_hash
from scripts.run_frozen_band_region import _write
from transfer import wp_cross_backend as backend
from transfer.g3_training import _forward
from transfer.registered_development import load_development

XZ = {
    "wp07": ("wp07-xuzhou-multiscale-real01",
             "711e90e6b08d9c4554a051bd2e5ae3d133e62a2fc6a897c362b7545ff1984604",
             "07b984a6e49a2c567c93f5713e1f9447db990acda24fab15e1760531da900281",
             "29ed4f67bf08f13070c20fec21ecb77f834a55778c5a090121737dcba0705a32",
             "vca-multiscale-run-v1"),
    "wp08": ("wp08-xuzhou-region-message-real01",
             "c19dfa3fe958b07935a17ad9ce45644a003cb396fbce927fa767c00c1b5cfd37",
             "5fb1eff61844f8f8c2b2688b261e797d49caf30dea4735ad3413ab4c0781ae08",
             "8df3b5c1d9c2e95a7ca04a713185a9af67187b3998e559b51762e242e3354f2e",
             "vca-region-message-run-v1"),
}
ARCHIVES = {
    "xz_directional": ("wp08-directional-gpu4-20261118/collected01/matrix-audit.json",
                       "88fdf89e327fe5ddc60fe3f67d22c98bf3520be58216584c0a5e3f784e6d185d"),
    "pavia_wp07": ("wp07-paviau-real-results-20261118.TC3eXtC5/aggregate/results/comparison-audit.json",
                   "728076ffe4751662cea2580e66c3b915d3fc5b30cecc1308ff1d761a3b8cc1bb"),
    "pavia_original": ("wp08-paviau-original-closeout-20261118/physical-audit.cSukhQLQ/per-case-comparison-audit.json",
                       "e7d59ee55331ce93c2c4af167cc3b76d7e5595c989256ff0b73b77eda865944f"),
    "pavia_gpu5": ("wp08-paviau-gpu5-rerun-20261118/audit/comparison-audit.json",
                   "aa786164e85638cda6b57b4b1af3c0708dd0db2b10c526aeff400e0d98cc061c"),
    "pavia_training_reproduction": ("wp08-paviau-original-closeout-20261118/reproduce-paired-final01/paired-summary.json",
                                    "1555e612664babc6025f44f1232e265274832ce49099ca91f72e3902163bc7be"),
    "pavia_frozen_device": ("wp06-wp08-paviau-frozen-diagnosis-20261118/summary-final01/summary.json",
                            "be1265f379a0e124cf4130cbee1bd917246be2ed965dfc139282876eb1b56116"),
    "xz_hh_spectral": ("wp06-native-pathways-20261120-qa/review6/pathways.json",
                       "d5682b35f4f6e78397dc13cfe725fcbdda77f363a5d7ae58357a4866c99dd126"),
}


def named_pin(root, path, sha):
    return {"path": str(root/path), "sha256": sha}


def provenance(root, family):
    path, reg_sha, done_sha, accepted_sha, _ = XZ[family]
    pins = {key: named_pin(root, path+"/"+key+".json", sha) for key, sha in
            (("registration", reg_sha), ("batch-complete", done_sha), ("audit", accepted_sha))}
    reg, done, audit = (json_pin(root, pins[key]) for key in pins)
    require(done["registration"] == audit["registration"] == pins["registration"] and audit["complete"] == pins["batch-complete"]
            and done["query"] == done["raw_gt"] == audit["query"] == audit["raw_gt"] == False
            and audit["independent_updates"] == 0 and audit["status"] == "passed"
            and set(reg["arm_order"]) == set(done["attempts"]) == set(audit["results"])
            and reg["budget"] == 256 and reg["recall_floor_delta"] == -.02,
            "Old registration or completed coverage no longer agrees with the accepted batch")
    return reg, done, audit, pins


def sealed_arm(root, family, arm, reg, complete):
    full = json_pin(root, complete["attempts"][arm]["full"])
    contract = reg["contracts"][arm]
    require(full["contract"] == contract and full["new_updates"] == reg["budget"], "Frozen full cursor differs")
    path = pinned(root, full["checkpoint"])
    packet = torch.load(path, map_location="cpu", weights_only=True)
    spec, history = packet["spec"], packet["history"]
    budget = reg["budget"]
    require(packet["format"] == XZ[family][-1] and packet["contract_hash"] == contract and digest(spec) == contract
            and spec["device"] == "cpu_fp32" and spec["updates"] == packet["progress"]["completed_updates"] == budget
            and spec["selection"] == "first_min_full_devval_fused_NLL_each_update"
            and [row["step"] for row in history] == list(range(1, budget+1))
            and packet["selected_step"] == min(history, key=lambda row: row["validation"]["nll"])["step"],
            "Checkpoint or saved first-minimum cursor is incompatible")
    family_stage = spec["architecture"]["backbone_kwargs"]
    kind = ("multiscale" if "cnn_multiscale" in family_stage else "region_message" if "region_attention" in family_stage
            else "aligned" if arm == "aligned" else "native")
    same_context = {key: spec[key] for key in ("data", "exposure_hash", "policy_hash", "updates", "sampler_seed",
                               "torch_seed", "lr", "optimizer_config", "sampler", "loss", "selection", "device", "scheduler")}
    same_context["exposure"] = digest([[row[k] for k in ("step", "tile_id", "anchor", "support_ids")] for row in history])
    return packet, kind, same_context, full["checkpoint"]


def dev_roster(scene, record):
    support = record["validation_support"]
    truth = dict(zip(support["indices"], support["labels"], strict=True))
    require(len(truth) == len(support["indices"]) == 45 and Counter(truth.values()) == {i:5 for i in range(1,10)},
            "Original Xuzhou V5 points or class population differs")
    seen, samples = [], []
    for row in record["roles"]:
        if row["role"] != "devval":
            continue
        y0,y1,x0,x1 = row["box"]
        width = scene.plan["shape"][1]
        points = sorted(i for i in truth if y0<=i//width<y1 and x0<=i%width<x1)
        cells = [(i//width-y0)*(x1-x0)+i%width-x0 for i in points]
        require(points and row["id"] == scene.plan["tiles"][row["id"]]["id"], "Bound V5 block has changed")
        samples.append((scene.plan["tiles"][row["id"]], cells, [truth[i] for i in points]))
        seen.extend(points)
    require(len(seen)==len(set(seen))==len(truth) and set(seen)==truth.keys(), "Historical evaluation points were reused")
    return samples, digest(seen)


@torch.no_grad()
def frozen_endpoints(scene, roster, packet, kind, audit, *, device="cpu", sync=lambda: None):
    spec, history = packet["spec"], packet["history"]
    results = {}
    for phase, step, state in (("selected",packet["selected_step"],packet["selected_state"]),
                               ("last",len(history),packet["state"])):
        start = perf_counter()
        model = backend.restore(spec["architecture"], kind, state, history[step-1]["state_hash"], device=device)
        sync()
        loaded = perf_counter()
        hashes, targets, prediction, cost, graph_sizes = [], [], {key:[] for key in HEADS}, [], []
        for tile, cells, truth in roster:
            block_start = perf_counter()
            outputs = _forward(model, scene, tile, multiscale_condition_only=True)
            sync()
            cost.append(perf_counter()-block_start)
            hashes.append([tensor_hash(value) for value in outputs[:3]])
            targets.extend(truth)
            for name, values in zip(HEADS, outputs[:3], strict=True):
                prediction[name].extend((values[cells].argmax(1)+1).tolist())
            info = outputs[-1]
            graph_sizes.append(dict(n=info["graph_nnz"], assignment=info["assignment_nnz"],
                                    range=len((info.get("multiscale_graph") or {}).get("nnz", [])),
                                    message=info.get("interaction", {}).get("region_message", {})))
        expected = audit["endpoints"][phase]
        require(step == expected["step"] and hashes == expected["scores"], "Original three-head probabilities did not replay")
        for h, guesses in prediction.items():
            cells = np.zeros((9,9),dtype=np.int64)
            np.add.at(cells,(np.asarray(targets)-1,np.asarray(guesses)-1),1)
            metrics = metrics_from_confusion(cells,list(range(1,10)))
            require(metrics["confusion_matrix"] == expected["metrics"][h]["confusion_matrix"],
                    "Original head point decisions differ")
        results[phase] = dict(step=step, state=history[step-1]["state_hash"], truth=targets, predictions=prediction,
                              original_metrics=expected["metrics"], inference_seconds=sum(cost),
                              recovery_seconds=loaded-start, graph=graph_sizes)
    return results


def xuzhou(root):
    started = perf_counter()
    registers = {family:provenance(root,family) for family in XZ}
    first, second = (registers[k][0] for k in XZ)
    require(all(first[k] == second[k] for k in ("source_state","source_hash","prior","exposure_hash",
                "exposure_prefix","covered_support","minimum_visits","budget","recall_floor_delta"))
            and all(first[k]["sha256"] == second[k]["sha256"] for k in ("source","sidecar")),
            "Historic campaigns cannot share a matched training cohort")
    for reg in (first,second):
        pinned(root,reg["source"])
        pinned(root,reg["sidecar"])
    prior = json_pin(root,first["prior"])
    pinned(root,prior["development"])
    scene, roles, _ = load_development(prior["development"])
    require(scene.dataset == "xuzhou" and scene.classes == 9, "Wrong released Xuzhou development package")
    roster, order = dev_roster(scene,roles)
    cases, contract = {}, None
    for family,(reg,complete,audit,pins) in registers.items():
        for arm in reg["arm_order"]:
            packet, kind, context, checkpoint = sealed_arm(root,family,arm,reg,complete)
            if contract is None:
                contract = context
            require(context == contract, "Two arms have incompatible training exposure, seed, or rules")
            info = audit["results"][arm]
            cases[f"{family}-{arm}"] = dict(backend=kind, checkpoint=checkpoint,
                points=order, scope="original_full_band_v5_development", budget=reg["budget"],
                endpoints=frozen_endpoints(scene,roster,packet,kind,info),
                training_cost={key:info[key] for key in ("per_stage","wall_seconds","updates")})
    require(all(all(cases["wp07-off"]["endpoints"][phase][key] == cases["wp08-off"]["endpoints"][phase][key]
                    for key in ("step","state","truth","predictions")) for phase in ("selected","last")),
            "The two original off controls are not identical")
    comparisons = [("wp07-off",name) for name in cases if name.startswith("wp07-") and name!="wp07-off"]
    comparisons.extend((f"wp08-{base}",f"wp08-{arm}") for pairs in second["comparisons"].values()
                       for arm,base in pairs)
    comparisons.extend((left,right) for left in ("wp07-graph_average","wp07-both_content")
                       for right in ("wp08-aligned","wp08-local","wp08-combined"))
    results = {}
    for left,right in comparisons:
        effects = {}
        for phase in ("selected","last"):
            a,b = (cases[k]["endpoints"][phase] for k in (left,right))
            require(a["truth"] == b["truth"] and (phase == "selected" or a["step"] == b["step"]),
                    "Pair changed the role, point order or fixed endpoint")
            effects[phase] = dict(left_step=a["step"],right_step=b["step"],
                heads={h:raw_pair(a["truth"],a["predictions"][h],b["predictions"][h],9,floor=5) for h in HEADS})
        results[f"{left}/{right}"] = effects
    for case in cases.values():
        for endpoint in case["endpoints"].values():
            del endpoint["truth"],endpoint["predictions"]
    return dict(pins={family:items[-1] for family,items in registers.items()}, track=contract,
                points=45, points_per_class=5, band_axis=scene.bands.fingerprint, point_order=order,
                seed=64, full_band=True, off_controls_equal=True, arms=cases, comparisons=results,
                wall_seconds=perf_counter()-started, added_optimizer_updates=0)

def directions(root, doc, *, execution=None):
    require(doc["status"] == "passed" and doc["optimizer_updates"] == 4608
            and doc["independent_audit_updates"] == doc["completed_reentry_updates"] == 0
            and doc["query"] == doc["raw_gt"] == 0, "Directional GPU campaign did not complete")
    wanted = {f"s{seed}--{direction}--{arm}" for seed in (64,512,2048)
              for direction in ("both","cnn_to_graph","graph_to_cnn") for arm in ("local","combined")}
    cases = doc["cases"]
    require(set(cases) == wanted and len(doc["per_seed_pairs"]) == 9
            and all(cases[key]["physical_uuid"] == cases[key.replace("--combined","--local")]["physical_uuid"]
                    for key in doc["per_seed_pairs"]), "The directional contrasts do not share their device")
    effects = None
    if execution is not None:
        prior = json_pin(root, named_pin(root, "wp06-xuzhou-real-route-campaign01/campaign.json",
                          "2dd3eff9c32300e1df6c330daaa7c79539301a699030c37dc2a4ae6ab8cfb6a4"))
        pinned(root, prior["development"])
        scene, roles, _ = load_development(prior["development"])
        roster, _ = dev_roster(scene,roles)
        outputs = {}
        for name, case in cases.items():
            require(case["physical_uuid"] == execution.numeric["physical_uuid"],
                    "Directional probability replay must use the original numerical device")
            entry = json_pin(root, case["receipts"]["resume"])
            packet = torch.load(pinned(root, entry["checkpoint"]), map_location="cpu", weights_only=True)
            require(digest(packet["spec"]) == packet["contract_hash"] == entry["contract"]
                    and packet["progress"]["completed_updates"] == 256
                    and len(packet["history"]) == 256
                    and packet["selected_step"] == min(packet["history"],
                        key=lambda row:row["validation"]["nll"])["step"],
                    "Directional checkpoint, progress or original first minimum disagree")
            outputs[name] = frozen_endpoints(scene, roster, packet, "region_message", case,
                                             device=execution.device,sync=execution.sync)
        effects = {}
        for name in doc["per_seed_pairs"]:
            effect = {}
            for phase in ("selected","last"):
                left,right = outputs[name.replace("--combined","--local")][phase],outputs[name][phase]
                require(left["truth"] == right["truth"], "Original direction reference changed point order")
                effect[phase] = {head:raw_pair(left["truth"],left["predictions"][head],
                    right["predictions"][head],9,floor=5) for head in HEADS}
                require(all(isclose(entry["delta_oa"],doc["per_seed_pairs"][name][phase][head]["delta"]["oa"],
                                    rel_tol=0,abs_tol=1e-12) for head,entry in effect[phase].items()),
                        "Replay cohort differs from original directional effects")
            effects[name] = effect
    return dict(pairs=doc["per_seed_pairs"],point_transitions=effects,interpretation="full_band_direction_only",
                budgets=dict(cases=18,steps=256,selected="first_min_V5"),
                original_costs={case:row["timings"] for case,row in cases.items()},
                independent_audit=doc["audits"], replay_device=None if execution is None else execution.numeric)


def pavia_points(root, scope_file):
    link = named_pin(root,"wp06-wp08-paviau-frozen-diagnosis-20261118/summary-final01/point-comparisons.csv",
                     "f286ae52f1b92bec4d022ce9d17c915ffa5922ae6bcddae848f6c2ab4350e8fa")
    predictions = {}
    with pinned(root,link).open(newline="",encoding="utf-8") as source:
        for item in csv.DictReader(source):
            key = (int(item["seed"]),item["band"],item["head"])
            predictions.setdefault(key,[]).append((item["point"],int(item["truth"]),int(item["local_prediction"]),
                                                     int(item["combined_prediction"])))
    require(len(predictions) == 18 and all(len(rows) == 45 and len({r[0] for r in rows})==45
            and Counter(r[1] for r in rows)=={i:5 for i in range(1,10)}
            for rows in predictions.values()), "Primary same-weight table has an incomplete roster")
    effects = {}
    for item in scope_file["primary"]:
        key = (item["seed"],item["band"],item["head"])
        rows = predictions[key]
        _, truth, first, last = map(list,zip(*rows,strict=True))
        result = raw_pair(truth,first,last,9,floor=5)
        require(isclose(result["delta_oa"],item["oa_delta"],abs_tol=1e-12)
                and result["protected"] is item["protected"], "Archived PaviaU paired point counts disagree")
        effects.setdefault(str(key[0]),{}).setdefault(key[1],{})[key[2]] = result
    return dict(receipt=link,selected_step=60,point_transitions=effects)


def paviau(root, archive):
    wp07, old, new, pair, frozen = (archive[key] for key in ("pavia_wp07","pavia_original","pavia_gpu5",
                                           "pavia_training_reproduction","pavia_frozen_device"))
    require(wp07["case_count"]==39 and wp07["optimizer_updates"]==2340
            and wp07["within_seed_same_physical_device"] and not (wp07["query_accessed"] or wp07["raw_gt_loaded"])
            and len(old["cases"])==len(new["cases"])==21 and old["optimizer_updates"]==new["optimizer_updates"]==1260
            and all(not row["raw_gt_loaded"] and not row["query_authorized"] and not row["audit_optimizer_updates"]
                    for row in (old,new)), "An archived PaviaU publication changes its released cohort")
    for link in wp07["files"]:
        pinned(root,link)
    paired_base = (root/ARCHIVES["pavia_training_reproduction"][0]).parent
    for name, sha in pair["tables"].items():
        pinned(root, {"path":str(paired_base/name), "sha256":sha})
    require(pair["all_same_selected_step"] and pair["endpoint_confusion_matches"]==252
            and pair["matched_nonempty_tile_heads"]==2142 and frozen["cpu_reference_disagreements"]==[],
            "Independent training and same-weight audit lost their original equivalence evidence")
    old_cases, new_cases = ({row["case"]:row for row in doc["cases"]} for doc in (old,new))
    require(old_cases.keys()==new_cases.keys(), "PaviaU independent training rerun used another roster")
    case_metrics = {}
    for case,row in new_cases.items():
        original = old_cases[case]
        require(row["optimizer_updates"] == original["optimizer_updates"] == 60
                and row["selected_step"] == original["selected_step"] == 60,
                "PaviaU case lost its original selection rule or budget")
        for field in ("head_metrics","terminal_head_metrics"):
            for head,column in row[field].items():
                require(set(column)=={"full","even_retained_positions"} and all(
                          value["confusion_matrix"]==original[field][head][band]["confusion_matrix"]
                          and [x["support"] for x in value["per_class"]]==[5]*9 for band,value in column.items()),
                        "PaviaU model cohort does not match the archived per-head class population")
        case_metrics[case] = dict(selected=row["head_metrics"],last=row["terminal_head_metrics"],
                                  original_device=old["per_case_devices"][case],rerun_device="GPU-5",
                                  costs=row["phase_costs"])
    same_weights = dict(representative_cases=len(frozen["cpu_validation"]),
                        disagreed=len(frozen["cpu_reference_disagreements"]),
                        comparisons=frozen["cpu_reference_comparisons"], table=pavia_points(root,frozen))
    training = {k:pair[k] for k in ("endpoint_confusion_matches","endpoint_class_correct_matches",
                 "matched_nonempty_tile_heads","max_abs_endpoint_aa_change","max_abs_endpoint_oa_change",
                 "max_abs_endpoint_nll_change","qualified_original_primary_rows")}
    return dict(multiscale=dict(cases=39,steps=60,views=wp07["bands"],tables=wp07["files"],
                                registered_budget=wp07["optimizer_updates"]),
                messages=dict(cases=case_metrics,recorded_pairs=new["comparisons"],
                              source_same_device_seeds=old["same_device_seeds"],mixed_source_seed=old["mixed_device_seed"],
                              independent_frozen_weight_replay=same_weights, independent_training=training),
                multiscale_vs_messages=None,
                exclusion="separate registries, optimizer exposure and scale/support construction; matching step budgets do not establish a matched architecture pair")


def review_history(root,out,code,*,directions_receipt=None):
    pins = {key:named_pin(root,*value) for key,value in ARCHIVES.items()}
    archive = {key:json_pin(root,pin) for key,pin in pins.items()}
    directional = directions(root,archive["xz_directional"])
    if directions_receipt is not None:
        extra = json_pin(root,directions_receipt)
        require(extra["source"] == pins["xz_directional"] and extra["producer"] == code["commit"]
                and extra["direction"]["pairs"] == directional["pairs"] and extra["optimizer_updates"] == 0,
                "Directional transitions are not from the same accepted study")
        directional = {**extra["direction"],"replay_receipt":directions_receipt}
    historic = archive["xz_hh_spectral"]
    require(historic["new_real_optimizer_updates"] == historic["authorized_real_interventions"] == 0
            and historic["query_accessed"] is historic["raw_gt_loaded"] is False,
            "Historical spectral results were promoted beyond frozen development")
    coverage = {key: dict(views=list(value["views"]),points=value["point_visits"])
                for key,value in historic["native"].items() if key.startswith("hh-")}
    require_frozen_code(code)
    findings = dict(scope="protocol_bound_real_development",sources=pins,producer=code["commit"],
        xuzhou_full=xuzhou(root),xuzhou_directions=directional,paviau=paviau(root,archive),
        xuzhou_central_gap=dict(receipt=pins["xz_hh_spectral"],scope="original_WP05_native_frozen_view",
                               structural_pairs=None,reason="the original WP07 and WP08 short tests authorized full-band V5"),
        honghu=dict(native_endpoints=coverage,focus_classes=[15,21],multiscale=None,region_message=None,
                    scope="single compatible original topology; no matching trained structural arm",
                    receipt=pins["xz_hh_spectral"]),
        independent_real=None,new_real_optimizer_updates=0,query_accessed=False,raw_gt_loaded=False,wp04_executed=False)
    require_frozen_code(code)
    return _write(out,findings)


def replay_directions(root,out,code,execution):
    link = named_pin(root,*ARCHIVES["xz_directional"])
    result = directions(root,json_pin(root,link),execution=execution)
    require_frozen_code(code)
    return _write(out,dict(source=link,producer=code["commit"],direction=result,optimizer_updates=0))
