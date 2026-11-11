"""Read-only diagnosis of the pinned full54 v1 probes and v2 normalization check."""
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from evaluation import frozen_probe_audit as audit, frozen_probe_campaign as campaign
from evaluation import frozen_probe_registry as reg, linear_probe, probe_decimal_reference as reference
from g4_support_context import check_files
from research_runtime import external_directory, hash_file, require_frozen_code, write_json


POLICY = "wp06_full54_v1_probe_diagnosis_and_v2_zero_update_v1"
V1_SHA = {
    "scope":"9e99db48a1df1bca898830d5b4e0826d9374b9b6da5dcaf8245c1d74f7677fa8",
    "execution":"50b8bf2834601c254f4cd7271aaf3960eb60dd4523960773d92641fd9f828d73",
    "review":"437ed795c09ce0289f5b1fd2fa68e0c1636bdb30661e159a757c8ce9f78f0cb3",
}


def errors(observed, expected):
    delta = np.abs(observed-expected)
    worst = np.unravel_index(int(delta.argmax()), delta.shape)
    nonzero = expected != 0
    return {"max_abs":float(delta[worst]), "max_relative_nonzero":float(
        np.max(delta[nonzero]/np.abs(expected[nonzero]), initial=0)),
        "nonzero_error_at_zero_reference":int(np.count_nonzero(delta[~nonzero])),
        "worst_index":[int(i) for i in worst], "worst_observed":float(observed[worst]),
        "worst_reference":float(expected[worst])}


def tolerance(observed, expected):
    ratio = np.abs(observed-expected)/(audit.PROB_ATOL+audit.PROB_RTOL*np.abs(expected))
    worst = np.unravel_index(int(ratio.argmax()), ratio.shape)
    return {"passes":bool(np.all(ratio <= 1)), "max_tolerance_ratio":float(ratio[worst]),
            "over_tolerance_cells":int(np.count_nonzero(ratio > 1)),
            "max_ratio_index":[int(i) for i in worst]}


def log_softmax(logits):
    shifted = logits-logits.max(1, keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(1, keepdims=True))


def layout(value):
    return {"dtype":str(value.dtype), "shape":list(value.shape),
            "strides":[int(s//value.itemsize) for s in value.strides],
            "c_contiguous":bool(value.flags.c_contiguous), "f_contiguous":bool(value.flags.f_contiguous)}


def fixed_head(plan):
    # Deterministic, untrained, and deliberately independent of the historical heads.
    classes, width = plan["classes"], plan["width"]
    slots = np.arange(classes*width, dtype=np.int64).reshape(classes,width)
    weight = (((slots+1)*8191 % 997)-498).astype(np.float64)/(997*width)
    bias = np.linspace(-0.1, 0.1, classes)
    return weight, bias


def compare_layer(pack, name, columns, plan, v1, v2, v2_meta, best, recorded, high_precision_roles):
    raw = {role:value.numpy() for role,value in pack["features"][name].items()}
    selected = {role:x.astype(np.float64)[:,columns] for role,x in raw.items()}
    t_selected = pack["features"][name]["train"].to(torch.float64)[:,columns]
    reg.require(np.array_equal(t_selected.numpy(), selected["train"]), "Torch/NumPy selected values differ")
    floor = plan["scale_floor"]
    mu_t, sd_t = t_selected.mean(0).numpy(), t_selected.std(0, unbiased=False).numpy()
    mu_n, sd_n = selected["train"].mean(0), selected["train"].std(0)
    scale_t, scale_n = np.where(sd_t > floor, sd_t, 1), np.where(sd_n > floor, sd_n, 1)
    contiguous = np.ascontiguousarray(selected["train"])
    c_mu, c_sd = contiguous.mean(0), contiguous.std(0)
    exact, exact_stats = reference.standardize(selected, floor)
    reg.require(v2_meta["scaled_columns"] == exact_stats["scaled_columns"],
                "v2 train-only scale-floor branches differ from Decimal")
    w, b = best["weight"], best["bias"]
    w_np, b_np = w.numpy(), b.numpy()
    fixed_w, fixed_b = fixed_head(plan)
    roles = {}
    for role, value in selected.items():
        t_x = v1[role]
        n_x = (value-mu_n)/scale_n
        t_logits = t_x @ w.T + b
        t_logp = F.log_softmax(t_logits, dim=1)
        reg.require(torch.equal(t_logp,recorded[role]), "Old selected authority differs byte-for-byte")
        t_l, t_log, t_prob = t_logits.numpy(), t_logp.numpy(), np.exp(t_logp.numpy())
        n_l = n_x @ w_np.T+b_np
        n_log = log_softmax(n_l)
        n_prob = np.exp(n_log)
        hybrids = {
            "numpy_using_torch_mean":np.exp(audit.numpy_logp((value-mu_t)/scale_n,w_np,b_np)),
            "numpy_using_torch_scale":np.exp(audit.numpy_logp((value-mu_n)/scale_t,w_np,b_np)),
            "numpy_using_torch_mean_and_scale":np.exp(audit.numpy_logp((value-mu_t)/scale_t,w_np,b_np)),
            "numpy_using_torch_x":np.exp(audit.numpy_logp(t_x.numpy(),w_np,b_np)),
            "numpy_using_torch_logits":np.exp(log_softmax(t_l)),
            "torch_exp_using_torch_logp":t_logp.exp().numpy(),
            "numpy_using_c_order_stats":np.exp(audit.numpy_logp(
                (value-c_mu)/np.where(c_sd > floor,c_sd,1),w_np,b_np)),
        }
        probability = tolerance(t_prob,n_prob)
        row, cls = probability["max_ratio_index"]
        top = np.sort(t_log[row])[-2:]
        v2_x = v2[role]
        v2_prob = F.log_softmax(v2_x @ torch.from_numpy(fixed_w).T+torch.from_numpy(fixed_b), dim=1).numpy()
        oracle_prob = audit.numpy_logp(exact[role],fixed_w,fixed_b)
        v2_fixed = audit.comparison(v2_prob,oracle_prob)
        roles[role] = {
            "raw_layout":layout(raw[role]), "numpy_selected_layout":layout(value),
            "torch_selected_layout":layout(pack["features"][name][role].to(torch.float64)[:,columns].numpy()),
            "x":errors(t_x.numpy(), n_x), "logits":errors(t_l,n_l), "logp":errors(t_log,n_log),
            "probability":{**errors(t_prob,n_prob), **probability},
            "probability_worst": {"row_ordinal":row, "class_id":cls+1, "torch_probability":float(t_prob[row,cls]),
                "numpy_probability":float(n_prob[row,cls]), "torch_class":int(t_log[row].argmax())+1,
                "numpy_class":int(n_log[row].argmax())+1, "torch_log_probability_margin":float(top[1]-top[0])},
            "v1_classification_disagreements":int(np.count_nonzero(t_log.argmax(1)!=n_log.argmax(1))),
            "hybrids":{key:{**errors(t_prob,alt),**tolerance(t_prob,alt)} for key,alt in hybrids.items()},
            "torch_x_to_decimal":errors(t_x.numpy(),exact[role]),
            "numpy_x_to_decimal":errors(n_x,exact[role]),
            "v2_x_to_decimal":{**errors(v2_x.numpy(),exact[role]),**tolerance(v2_x.numpy(),exact[role])},
            "v2_fixed_head_to_decimal_features":v2_fixed,
            "amplified_feature_columns":[columns[j] for j in np.flatnonzero(np.max(np.abs(t_x.numpy()-n_x),0)>1e-12)],
        }
        if role in high_precision_roles:
            dl, dlog, dp = reference.head(exact[role],w_np,b_np)
            tl, _, tp = reference.head(t_x.numpy(),w_np,b_np)
            roles[role]["decimal_head"] = {
                "torch_logits_to_decimal":errors(t_l,dl), "numpy_logits_to_decimal":errors(n_l,dl),
                "torch_logp_to_decimal":errors(t_log,dlog), "numpy_logp_to_decimal":errors(n_log,dlog),
                "torch_probability_to_decimal":{**errors(t_prob,dp),**tolerance(t_prob,dp)},
                "numpy_probability_to_decimal":{**errors(n_prob,dp),**tolerance(n_prob,dp)},
                "torch_logits_to_decimal_on_same_x":errors(t_l,tl),
                "torch_probability_to_decimal_on_same_x":errors(t_prob,tp),
            }
    return {"roles":roles, "train_statistics":{
        "torch_mean_vs_numpy":errors(mu_t,mu_n), "torch_sd_vs_numpy":errors(sd_t,sd_n),
        "torch_scale_vs_numpy":errors(scale_t,scale_n),
        "torch_mean_vs_decimal":errors(mu_t,exact_stats["mean"]), "numpy_mean_vs_decimal":errors(mu_n,exact_stats["mean"]),
        "torch_sd_vs_decimal":errors(sd_t,exact_stats["sd"]), "numpy_sd_vs_decimal":errors(sd_n,exact_stats["sd"]),
        "numpy_c_mean_vs_original":errors(c_mu,mu_n), "numpy_c_sd_vs_original":errors(c_sd,sd_n),
        "v2_mean_vs_decimal":errors(v2_meta["mean"].numpy(),exact_stats["mean"]),
        "v2_scale_vs_decimal":errors(v2_meta["scale"].numpy(),exact_stats["scale"]),
        "floor":floor, "columns":[{"source_column":col, "torch_mean":float(mt), "numpy_mean":float(mn),
            "torch_sd":float(st), "numpy_sd":float(sn), "decimal_sd":float(sd), "torch_scale":float(ct),
            "numpy_scale":float(cn), "torch_sd_minus_floor":float(st-floor), "numpy_sd_minus_floor":float(sn-floor),
            "torch_scaled":bool(st>floor), "numpy_scaled":bool(sn>floor), "v2_scaled":v2_flag}
            for col,mt,mn,st,sn,sd,ct,cn,v2_flag in zip(columns,mu_t,mu_n,sd_t,sd_n,exact_stats["sd"],
                scale_t,scale_n,v2_meta["scaled_columns"],strict=True)]}}


def load_v1(scope_path, execution_path, review_path):
    paths = {key:Path(path).resolve(strict=True) for key,path in
             zip(V1_SHA,(scope_path,execution_path,review_path),strict=True)}
    scope,execution,review = [reg.pinned_json(paths[key],V1_SHA[key]) for key in paths]
    reg.require(review["status"]=="incomplete_or_failed" and review["accepted_pairs"]==28
                and review["failed_initial_probes"]==38 and review["reviewed_initial_probes"]==216
                and review["scope_sha256"]==V1_SHA["scope"] and review["execution_sha256"]==V1_SHA["execution"]
                and review["protected_files_unchanged"] and not review["synthetic"]
                and not paths["review"].with_name("acceptance.json").exists(), "Original v1 failed chain differs")
    pins = {**review["protected_files"],**review["outputs"],**{str(path):V1_SHA[key] for key,path in paths.items()}}
    check_files(pins)
    cases = {row["case"]:reg.pinned_json(paths["review"].parent/(row["case"]+".json"),
        pins[str(paths["review"].parent/(row["case"]+".json"))]) for row in review["cases"]}
    return paths,scope,execution,review,pins,cases


def archived_case(root, name, execution_sha, pins, steps):
    folder = campaign.folder_at(root,name)
    case_path = folder/"execution.json"
    reg.require(pins[str(case_path)] == execution_sha,"Archived case execution pin differs")
    run = reg.pinned_json(case_path,execution_sha)
    pack_path, state_path, probability_path, report_path = (folder/"pack.pt",folder/f"full-{steps:04d}/state.pt",
                                                            folder/"selected-logp.pt",folder/f"full-{steps:04d}/report.json")
    check_files({str(path):pins[str(path)] for path in (pack_path,state_path,probability_path,report_path)})
    reg.require(all(run["files"][str(path)]==pins[str(path)] for path in
        (pack_path,state_path,probability_path,report_path)) and run["completed_resume_updates"]==0,
        "Archived pack/selected state/report differs")
    pack = torch.load(pack_path,map_location="cpu",weights_only=True)
    state = campaign.read_state(state_path,pins[str(state_path)])
    report = reg.pinned_json(report_path,pins[str(report_path)])
    probabilities = torch.load(probability_path,map_location="cpu",weights_only=True)
    return pack,state,report,probabilities


def diagnose_case(source, scene, plan, columns, archived, old_proof, precise):
    pack,state,report,probabilities = archived
    reg.require(report["plan"]==plan and pack["identity"]["query_labels_accessed"] is False, "Archived plan or role differs")
    audited = audit.audit_probes(pack,{"probe_plan":plan,"columns":columns},state,report,probabilities)
    reg.require(audited["numerical_failures"]==old_proof["numerical_failures"], "Could not reproduce original v1 failures")
    v1,_,old_meta = linear_probe.prepare(pack,plan)
    v2,_,new_meta = linear_probe.prepare(pack,{**plan,"policy":linear_probe.POLICY_V2})
    layers = {}
    for name in reg.LAYERS:
        cols = columns[name.split("_")[0]]
        reg.require(cols==old_meta[name]["columns"]==new_meta[name]["columns"],"Probe columns changed")
        layers[name] = compare_layer(pack,name,cols,plan,v1[name],v2[name],new_meta[name],
            state["layers"][name]["best"],probabilities[name],{role for layer,role in precise if layer==name})
        for role, endpoint in layers[name]["roles"].items():
            old = old_proof["layers"][name]["roles"][role]["cross_backend"]
            current = audited["layers"][name]["roles"][role]["cross_backend"]
            reg.require(old==current and endpoint["probability"]["passes"]==old["probabilities_close"],
                        "Detailed decomposition differs from original v1 numerical comparison")
    return {"source":source,"scene":scene,"layers":layers}


def fresh_output(out, paths, pins):
    out = external_directory(out)
    reg.require(out.parent==paths["review"].parent.parent and not out.exists()
        and all(out not in Path(file).parents for file in pins), "Fresh sibling diagnostic output required")
    out.mkdir(exist_ok=False)
    return out


def terminal_index(scope, pins):
    ledger = {}
    for history in (scope["history"],scope["history"]["pavia"]):
        item = history["sources"]["execution"]
        path = Path(item["path"])
        reg.require(pins[str(path)]==item["sha256"],"Archived terminal execution pin differs")
        rows = reg.pinned_json(path,item["sha256"])["cases"]
        ledger.update({row["case"]:(path.parent,row["execution_sha256"])
                       for row in rows if row["status"]=="executed"})
    return ledger


def worst_cases(cases):
    worst = {}
    for name,result in cases.items():
        for failure in result["initial"]["numerical_failures"]:
            candidate = (failure["maximum_absolute_probability_error"],name,failure["layer"],failure["role"])
            scene = result["role"]["target"]
            if scene not in worst or candidate > worst[scene]:
                worst[scene] = candidate
    return worst


def diagnose(scope_path, execution_path, review_path, out, code):
    started = time.monotonic()
    require_frozen_code(code)
    paths,scope,execution,review,pins,cases = load_v1(scope_path,execution_path,review_path)
    out = fresh_output(out,paths,pins)
    prior = terminal_index(scope,pins)
    worst = worst_cases(cases)
    counters, maxima, result_files = {}, {}, []
    for item in execution["cases"]:
        name,scene = item["case"],item["scene"]
        plan = scope["probe_plans"][scene]
        for source,(root,sha) in (("initial",(paths["execution"].parent,item["execution_sha256"])),
                                  ("terminal",prior[name])):
            precise = [tuple(value[2:]) for value in worst.values() if value[1]==name and source=="initial"]
            data = diagnose_case(source,scene,plan,scope["columns"],archived_case(root,name,sha,pins,plan["steps"]),
                                 cases[name][source],precise)
            tally = counters.setdefault(f"{source}/{scene}",Counter())
            peak = maxima.setdefault(f"{source}/{scene}",Counter())
            for layer in data["layers"].values():
                for endpoint in layer["roles"].values():
                    tally["endpoints"]+=1
                    tally["v1_probability_failures"]+=not endpoint["probability"]["passes"]
                    tally["v1_decision_disagreements"]+=endpoint["v1_classification_disagreements"]
                    tally["v2_standardization_failures"]+=not endpoint["v2_x_to_decimal"]["passes"]
                    fixed = endpoint["v2_fixed_head_to_decimal_features"]
                    tally["v2_fixed_head_failures"]+=not (fixed["probabilities_close"] and fixed["strict_decisions"])
                    for key in ("probability","x","logits","logp","v2_x_to_decimal"):
                        peak[key] = max(peak[key],endpoint[key]["max_abs"])
                    for key, swap in endpoint["hybrids"].items():
                        tally["hybrid_failures/"+key]+=not swap["passes"]
            filename = f"{name}-{source}.json"
            write_json(out/filename,data)
            result_files.append(filename)
    check_files(pins)
    require_frozen_code(code)
    observed = {name:count["v1_probability_failures"] for name,count in counters.items()}
    reg.require(sum(observed.values())==67 and len(prior)==len(cases)==54
                and sum(count["endpoints"] for count in counters.values())==864
                and all(not count["v1_decision_disagreements"] for count in counters.values()),
                "Full original 432+432 selected endpoint reproduction differs")
    passed = not any(count["v2_standardization_failures"] or count["v2_fixed_head_failures"]
                     for count in counters.values())
    path = out/"receipt.json"
    write_json(path,{"policy":POLICY,"status":"zero_update_checks_passed" if passed else "v2_zero_update_checks_failed",
        "commit":code["commit"],"source_hash":code["source_hash"],"original_v1_receipt_status":review["status"],
        "source_pins":V1_SHA,"protected_files_count":len(pins),"protected_files_unchanged":True,
        "worst_v1_initial_by_scene":worst,"endpoint_counters":counters,"maximum_errors":maxima,
        "outputs":campaign.pins_for(out,result_files),"wall_seconds":time.monotonic()-started,
        "new_head_updates":0,"audit_probe_updates":0,"backbone_updates":0,"bn_updates":0,
        "new_model_forwards":0,"raw_gt_deserialized":False,"query_labels_accessed":False,
        "v2_real_fitting_performed":False})
    return path
