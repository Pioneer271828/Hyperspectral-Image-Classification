"""Review previously released comparisons without training or joining dissimilar protocols."""

from collections import Counter
import csv
from math import isclose
from time import perf_counter

import numpy as np
import torch

from evaluation.classification_metrics import metrics_from_confusion
from evaluation.frozen_training_stages import HEADS, json_pin, pinned, probabilities, require
from evaluation.wp12_cross_backend import raw_pair
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
    "xz_directional_cpu": ("wp08-directional-matrix-20261118/collected03/matrix-audit.json",
                           "696b823117ac492772ffbd51cdcceeabe07aa2dc864941aa4fcdfa8f2fcf5f79"),
    "xz_scale_gap": ("wp05-wp06-wp07-band-scale-20261119/xuzhou.json",
                     "9501787b0e27fad18291059531c06c0d201f04c4af1a94ec0967ef0e923fa8f6"),
    "hh_scale_gap": ("wp05-wp06-wp07-band-scale-20261119/whu-honghu.json",
                     "c135dc35688627762d40cf16b32ec44669a6d2d9e9f83537854f4119d7775c1b"),
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
def frozen_endpoints(scene, roster, packet, kind, audit):
    spec, history = packet["spec"], packet["history"]
    results = {}
    for phase, step, state in (("selected",packet["selected_step"],packet["selected_state"]),
                               ("last",len(history),packet["state"])):
        start = perf_counter()
        model = backend.restore(spec["architecture"], kind, state, history[step-1]["state_hash"])
        loaded = perf_counter()
        hashes, targets, prediction, cost, graph_sizes = [], [], {key:[] for key in HEADS}, [], []
        for tile, cells, truth in roster:
            block_start = perf_counter()
            outputs = _forward(model, scene, tile, multiscale_condition_only=True)
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

def directions(root, gpu, cpu):
    require(gpu["status"] == "passed" and gpu["optimizer_updates"] == 864
            and gpu["independent_audit_updates"] == gpu["completed_reentry_updates"] == 0
            and gpu["query"] is gpu["raw_gt"] is False, "Directional GPU engineering campaign did not complete")
    matrix = json_pin(root,gpu["matrix"])
    require(matrix["inputs"] is None, "Directional scope differs from its engineering registration")
    wanted = {f"s{seed}--{direction}--{arm}" for seed in (64,512,2048)
              for direction in ("both","cnn_to_graph","graph_to_cnn") for arm in ("local","combined")}
    cases = gpu["cases"]
    require(set(cases) == wanted and len(gpu["per_seed_pairs"]) == 9
            and set(cpu["cases"]) == wanted and cpu["query"] is cpu["raw_gt"] is False
            and all(cases[key]["physical_uuid"] == cases[key.replace("--combined","--local")]["physical_uuid"]
                    for key in gpu["per_seed_pairs"]), "The directional contrasts do not share their registered scope")
    return dict(synthetic=dict(cases=18,devices={"cpu":cpu["protocol"],"cuda":gpu["protocol"]},
                               updates={"cpu":cpu["optimizer_updates"],"cuda":gpu["optimizer_updates"]},
                               pairs={"cpu":cpu["per_seed_pairs"],"cuda":gpu["per_seed_pairs"]}),
                real_notice="Legacy bidirectional Xuzhou local and combined are in the matched full-band cohort.",
                real_single_direction=None,
                exclusion="The archived device qualification matrix uses synthetic data; no released real single-direction matrix.")


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
    anchors = {}
    for link in wp07["case_replays"]:
        case = json_pin(root,link)
        if not case["case"].endswith("/1/off"):
            continue
        paired = old_cases[case["case"]]
        archived = { (row["band"],row["tile"]): row["heads"] for row in paired["validation"]["rows"] }
        current = { (row["band"],row["tile"]): row["heads"] for row in case["validation_views"]["selected"] }
        keys = ("source_state","initialization","released_train_exposures","selected_step","optimizer_updates")
        require(all(case[k]==paired[k] for k in keys) and current.keys()==archived.keys(),
                "Common PaviaU factor-one off does not bind the same source or released tiles")
        anchors[str(case["case"].split("/")[1])] = dict(shared_source=True, step=case["selected_step"],
             exact_tile_heads=sum(a[h]==archived[k][h] for k,a in current.items() for h in HEADS),
             compared_tile_heads=len(current)*len(HEADS), multiscale_device=case["actual_cuda_uuid"],
             message_device=old["per_case_devices"][case["case"]])
    require(set(anchors)=={"64","512","2048"}, "Missing PaviaU factor-one anchor")
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
                shared_factor_one_off=anchors,multiscale_vs_messages=None,
                exclusion="Only factor-one off is shared; trained WP07 candidates used factor 2 or 4 and WP08 trained only factor 1.")


def existing_scale_views(doc, *, scene, classes, focus):
    require(doc["scene"] == scene and doc["role"] == "devval" and doc["real_updates"] == 0
            and doc["query_accessed"] is doc["raw_gt_loaded"] is False,
            "Stored band comparison escapes its authorized development scope")
    groups = {}
    for row in doc["rows"]:
        event = row["event"]
        key = (event["factor"],event["band"],event["policy"])
        points = groups.setdefault(key, {})
        for i,(point,label) in enumerate(zip(row["points"],row["labels"],strict=True)):
            require(point not in points and doc["labels"][point] == label and label in focus,
                    "Historical scale views changed released point order or identity")
            points[point] = (label,{head:probabilities(row["probabilities"][head],len(row["points"]),classes)
                                    .argmax(1)[i].item()+1 for head in HEADS})
    results = {}
    for key, after in groups.items():
        factor,band,policy = key
        before = groups[(factor,"full","native")]
        require(after.keys() == before.keys() and sorted(after) == list(range(len(doc["labels"])))
                and all(before[i][0]==r[0] for i,r in after.items()), "Historical comparison is not same-point")
        index = sorted(before)
        effects = {}
        for h in HEADS:
            result = raw_pair([before[i][0] for i in index],[before[i][1][h] for i in index],
                              [after[i][1][h] for i in index],classes,floor=5)
            effects[h] = {str(label):result["transitions"][str(label)] for label in focus}
        results[f"factor{factor}/{band}/{policy}"] = effects
    return dict(cursor=doc["selected_step"], roles=doc["role"], policies=doc["supported_policies"],
                no_cross_reason=doc["no_cross_reason"], comparison_to_same_factor_full_native=results)


def review_history(root,out,code):
    pins = {key:named_pin(root,*value) for key,value in ARCHIVES.items()}
    archive = {key:json_pin(root,pin) for key,pin in pins.items()}
    historic = archive["xz_hh_spectral"]
    require(historic["new_real_optimizer_updates"] == historic["authorized_real_interventions"] == 0
            and historic["query_accessed"] is historic["raw_gt_loaded"] is False,
            "Historical spectral results were promoted beyond frozen development")
    coverage = {key: dict(views=list(value["views"]),points=value["point_visits"])
                for key,value in historic["native"].items() if key.startswith("hh-")}
    require_frozen_code(code)
    findings = dict(scope="protocol_bound_real_development",sources=pins,producer=code["commit"],
        xuzhou_full=xuzhou(root),xuzhou_directions=directions(root,archive["xz_directional"],archive["xz_directional_cpu"]),
        paviau=paviau(root,archive),
        xuzhou_central_gap=dict(native_receipt=pins["xz_hh_spectral"],
                               same_weight_scale_receipt=pins["xz_scale_gap"],
                               same_weight_scale=existing_scale_views(archive["xz_scale_gap"],
                                                                    scene="xuzhou",classes=9,focus=(5,6)),
                               matched_trained_structure_pairs=None,
                               reason="The original WP07 and WP08 matched short tests authorized full-band V5 only."),
        honghu=dict(native_endpoints=coverage,focus_classes=[15,21],multiscale=None,region_message=None,
                    scope="Single compatible original topology; no matching trained structural arm.",
                    same_weight_views=existing_scale_views(archive["hh_scale_gap"],
                                                           scene="whu-honghu",classes=22,focus=(15,21)),
                    receipt=pins["xz_hh_spectral"]),
        independent_real=None,new_real_optimizer_updates=0,query_accessed=False,raw_gt_loaded=False,wp04_executed=False)
    require_frozen_code(code)
    return _write(out,findings)
