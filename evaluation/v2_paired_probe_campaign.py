"""Execute and independently review new v2 heads on both authenticated backbone roles."""
import json
from pathlib import Path
import time

import torch

from evaluation import frozen_probe_audit as audit, frozen_probe_campaign as core, frozen_probe_registry as reg
from evaluation import frozen_probe_summary as csv_output, initial_final_probe_summary as summaries
from evaluation import linear_probe, probe_pair_audit_status as statuses, v2_paired_probe_registry as registry
from g4_support_context import check_files
from research_runtime import external_directory, hash_file, require_frozen_code, write_json


def output_location(out,evidence):
    out = external_directory(out)
    root = Path(evidence.sources["review"]["path"]).parent.parent
    reg.require(out.parent==root and not out.exists() and all(out not in Path(path).parents for path in evidence.pins),
                "Fresh sibling output outside all certified inputs required")
    return out


def stage_context(scope,case,stage):
    source = case["stages"][stage]
    pack = source["pack"]
    stage_case = {"name":case["name"],"capture_receipt_sha256":source["capture"]["sha256"]}
    stage_scope = {"probe_plan":scope["plans"][case["target"]],"paths":scope["paths"],"columns":scope["columns"]}
    contract = {"policy":registry.POLICY,"backbone_stage":stage,"model_state_sha256":source["model_state_sha256"],
                "source_pack_sha256":pack["sha256"],"source_pack_fingerprint":source["fingerprint"]}
    return stage_case,stage_scope,{pack["path"]:pack["sha256"]},contract


def allocation(scope_path,scope_sha,out,code):
    return {"policy":registry.POLICY,"scope":str(scope_path),"scope_sha256":scope_sha,
            "execution":str(out/"execution.json"),"commit":code["commit"],"source_hash":code["source_hash"]}


def owner_path(scope_path,scope):
    draft = scope["sources"]["draft"]["sha256"]
    reg.require(len(draft)==64 and all(char in "0123456789abcdef" for char in draft),"Pinned draft SHA256 required")
    path = scope_path.parent.parent/".wp06-v2-paired-allocations"/(draft+".json")
    reg.require(path.resolve()==path,"Execution allocation cannot traverse a symlink")
    return path


def total_updates(cases):
    return {stage:sum(case["stages"][stage]["verified_updates"] for case in cases) for stage in registry.ROLES}


def save_execution(path,state):
    state["updates_by_stage"] = total_updates(state["cases"])
    state["verified_updates"] = sum(state["updates_by_stage"].values())
    write_json(path,state)


def run(scope_path,scope_sha,out,code):
    started = time.monotonic()
    scope_path = Path(scope_path).resolve(strict=True)
    scope,evidence = registry.load(scope_path,scope_sha,code)
    out = output_location(out,evidence)
    owner = owner_path(scope_path,scope)
    reg.require(not owner.exists(),"This v2 scope already has an execution allocation")
    check_files(evidence.pins)
    registry.preflight(scope)
    require_frozen_code(code)
    preflight_seconds = time.monotonic()-started
    # A filesystem-exclusive claim also prevents two simultaneous processes from fitting the same scope.
    owner.parent.mkdir(exist_ok=True)
    with owner.open("x",encoding="utf-8") as stream:
        json.dump(allocation(scope_path,scope_sha,out,code),stream,indent=2)
        stream.write("\n")
    owner_sha = hash_file(owner)
    out.mkdir(exist_ok=False)
    path = out/"execution.json"
    rows = [{"case":case["name"],"scene":case["target"],"stages":{
                stage:{"status":"not_run","segments":[],"verified_updates":0} for stage in registry.ROLES}}
            for case in scope["cases"]]
    state = {"policy":registry.POLICY,"status":"running","commit":code["commit"],"scope_sha256":scope_sha,
        "owner_sha256":owner_sha,"synthetic":scope["synthetic"],"cases":rows,
        "preflight_seconds":preflight_seconds,"historical_head_reuse":0,"backbone_updates":0,
        "bn_updates":0,"raw_gt_deserialized":False,"query_labels_accessed":False}
    save_execution(path,state)
    for case,row in zip(scope["cases"],rows,strict=True):
        case_dir = core.folder_at(out,case["name"])
        case_dir.mkdir(exist_ok=False)
        for stage in registry.ROLES:
            stage_dir = core.folder_at(case_dir,stage)
            stage_case,stage_scope,local,contract = stage_context(scope,case,stage)
            recorded = row["stages"][stage]
            recorded.update(status="running")
            save_execution(path,state)

            def record(receipt):
                entry = {key:receipt[key] for key in ("path","start","until","probe_updates","wall_seconds","state_sha256")}
                recorded["segments"].append(entry)
                recorded["verified_updates"] += entry["probe_updates"]
                save_execution(path,state)

            def writer(_bound, *, local_only):
                return registry.load_pack(case,stage,scope["scene_classes"][case["target"]])

            try:
                sha = core.run_case(stage_dir,None,stage_case,local,stage_scope,scope_sha,record,
                                   writer=writer,contract=lambda _:contract)
                recorded.update(status="executed",execution_sha256=sha)
            except Exception as error:
                recorded.update(status="failed",error=repr(error),additional_unverified_updates_possible=True)
                state["status"] = "incomplete_or_failed"
            save_execution(path,state)
            if state["status"]=="incomplete_or_failed":
                break
        if state["status"]=="incomplete_or_failed":
            break
    else:
        state["status"] = "executed"
        reg.require(state["verified_updates"]==scope["planned_total_updates"],"Execution budget differs")
    check_files({**evidence.pins,str(scope_path):scope_sha,str(owner):owner_sha})
    require_frozen_code(code)
    state.update(protected_files_unchanged=True,wall_seconds=time.monotonic()-started)
    save_execution(path,state)
    return path


def checked_ledger(path,sha,scope_path,scope_sha,scope,code):
    state = reg.pinned_json(path,sha)
    owner = owner_path(scope_path,scope)
    owner_sha = state["owner_sha256"]
    reg.require(path.name=="execution.json" and reg.pinned_json(owner,owner_sha)==allocation(scope_path,scope_sha,path.parent,code)
        and state["status"] in ("executed","incomplete_or_failed") and state["policy"]==registry.POLICY
        and state["commit"]==code["commit"] and state["scope_sha256"]==scope_sha and state["synthetic"]==scope["synthetic"]
        and state["historical_head_reuse"]==state["backbone_updates"]==state["bn_updates"]==0
        and state["protected_files_unchanged"] is True
        and state["raw_gt_deserialized"] is state["query_labels_accessed"] is False,
        "Finalized v2 execution and allocation identity required")
    complete = True
    for case,row in zip(scope["cases"],state["cases"],strict=True):
        reg.require((row["case"],row["scene"])==(case["name"],case["target"]) and set(row["stages"])==set(registry.ROLES),
                    "Execution case order or roles differ")
        for stage,value in row["stages"].items():
            _,stage_scope,_,_ = stage_context(scope,case,stage)
            expected = [(kind,start,until,len(reg.LAYERS)*(until-start))
                        for kind,start,until,_,_ in core.schedule(stage_scope)]
            seen = [tuple(s[k] for k in ("path","start","until","probe_updates")) for s in value["segments"]]
            status = value["status"]
            reg.require(status in ("executed","failed","not_run")
                and seen==expected[:len(seen)]
                and value["verified_updates"]==sum(x[-1] for x in seen)
                and (status!="executed" or seen==expected and len(value["execution_sha256"])==64)
                and (status!="not_run" or not seen), "Execution interval, completed resume or role budget differs")
            complete = complete and status=="executed"
    updates = total_updates(state["cases"])
    reg.require((state["status"]=="executed")==complete
                and state["updates_by_stage"]==updates and state["verified_updates"]==sum(updates.values())
                and (not complete or state["verified_updates"]==scope["planned_total_updates"]),
                "Recorded v2 batch budget differs")
    return state,{str(owner):owner_sha,str(path):sha,str(scope_path):scope_sha}


def legacy_sensitivity(case,stage,selected,old_proof,audited):
    source = case["stages"][stage]["legacy_probabilities"]
    check_files({source["path"]:source["sha256"]})
    old_prob = torch.load(source["path"],map_location="cpu",weights_only=True)
    result = []
    for layer,record in audited["layers"].items():
        old = old_proof["layers"][layer]
        for role in ("train","val"):
            before,after = old_prob[layer][role],selected[layer][role]
            reg.require(before.shape==after.shape and before.dtype==after.dtype==torch.float64,
                        "Old sensitivity probabilities have wrong shape")
            changed = torch.where(before.argmax(1)!=after.argmax(1))[0].tolist()
            result.append({"layer":layer,"role":role,"v1_step":old["selected_step"],"v2_step":record["selected_step"],
                "changed_decisions":len(changed),"changed_row_ordinals":changed,
                "rows":len(before),"v1_oa":old["roles"][role]["selected"]["oa"],
                "v2_oa":record["roles"][role]["selected"]["oa"],
                "v1_nll":old["roles"][role]["selected"]["nll"],"v2_nll":record["roles"][role]["selected"]["nll"]})
    return result


def audit_stage(root,case,stage,recorded,scope,scope_sha,legacy):
    folder = core.folder_at(core.folder_at(root,case["name"]),stage)
    stage_case,stage_scope,local,contract = stage_context(scope,case,stage)
    pack,state,report,selected,counts,pins = core.checked_execution(folder,recorded["execution_sha256"],
        stage_scope,scope_sha,stage_case,local,contract=lambda _:contract)
    expected_counts = {key:value//54 for key,value in scope["planned_updates_per_stage"].items()}
    reg.require(linear_probe.fingerprint(pack)==contract["source_pack_fingerprint"]
        and report["policy"]==report["plan"]["policy"]==linear_probe.POLICY_V2
        and counts==expected_counts
        and sum(counts.values())==recorded["verified_updates"], "New v2 scientific state, input or role budget differs")
    for segment,(*_,name) in zip(recorded["segments"],core.schedule(stage_scope),strict=True):
        path = folder/name/"receipt.json"
        stored = reg.pinned_json(path,pins[str(path)])
        reg.require(all(stored[key]==value for key,value in segment.items()),"Durable segment ledger differs")
    scores = audit.audit_probes(pack,stage_scope,state,report,selected)
    trajectory = []
    for layer,value in state["layers"].items():
        best,first,last = value["best"],value["history"][0],value["history"][-1]
        trajectory.append({"layer":layer,"first":first,"last":last,"selected_step":best["step"],
                           "selected_nll":best["nll"],"parameters":scores["layers"][layer]["parameters"]})
    return {**scores,"case":case["name"],"role":case,"disposition":"new","backbone_stage":stage,
        "status":"failed_numerical_gate" if scores["numerical_failures"] else "accepted",
        "updates_by_path":counts,"verified_updates":recorded["verified_updates"],"audit_probe_updates":0,
        "exact_continuous_split_and_completed":True,"trajectory":trajectory,
        "v1_sensitivity":legacy_sensitivity(case,stage,selected,legacy,scores),
        "training_seconds":sum(item["wall_seconds"] for item in recorded["segments"])},{
        **pins,str(folder/"execution.json"):recorded["execution_sha256"]}


def trajectory_tables(out,results):
    trajectories,sensitivity = [],[]
    for pair in results:
        case = pair["role"]
        for stage in registry.ROLES:
            row = pair[stage]
            common = {"case":pair["case"],"scene":case["target"],"seed":case["seed"],"mode":case["mode"],
                      "method":case["method"],"backbone_stage":stage}
            for entry in row.get("trajectory",[]):
                trajectories.append({**common,**{key:entry[key] for key in ("layer","parameters","selected_step","selected_nll")},
                    **{f"{when}_{key}":value for when in ("first","last") for key,value in entry[when].items()}})
            sensitivity.extend({**common,**value,"changed_row_ordinals":" ".join(map(str,value["changed_row_ordinals"]))}
                               for value in row.get("v1_sensitivity",[]))
    csv_output.write_csv(out/"trajectories.csv",trajectories,("case","scene","backbone_stage","layer"))
    csv_output.write_csv(out/"v1-sensitivity.csv",sensitivity,("case","scene","backbone_stage","layer","role"))
    return ("trajectories.csv","v1-sensitivity.csv")


def review(scope_path,scope_sha,execution_path,execution_sha,out,code):
    started = time.monotonic()
    scope_path,execution_path = (Path(path).resolve(strict=True) for path in (scope_path,execution_path))
    scope,evidence = registry.load(scope_path,scope_sha,code)
    ledger,pins = checked_ledger(execution_path,execution_sha,scope_path,scope_sha,scope,code)
    out = output_location(out,evidence)
    out.mkdir(exist_ok=False)
    results,output_names = [],[]
    for case,row in zip(scope["cases"],ledger["cases"],strict=True):
        name = case["name"]
        pair = {"case":name,"role":case}
        for stage in registry.ROLES:
            recorded = row["stages"][stage]
            value = {"status":recorded["status"],"verified_updates":recorded["verified_updates"]}
            if recorded["status"]=="executed":
                try:
                    value,artifacts = audit_stage(execution_path.parent,case,stage,recorded,scope,scope_sha,
                                                  evidence.legacy[name][stage])
                    pins.update(artifacts)
                except Exception as error:
                    value.update(status="failed_review",error=repr(error))
            elif recorded["status"]=="failed":
                value.update(error=recorded["error"],additional_unverified_updates_possible=True)
            pair[stage] = value
        filename = name+".json"
        write_json(core.folder_at(out,filename),pair)
        output_names.append(filename)
        results.append(pair)
    reviewed = [pair[stage] for pair in results for stage in registry.ROLES if "layers" in pair[stage]]
    failures = [{"case":row["case"],"stage":row["backbone_stage"],**failure}
                for row in reviewed for failure in row["numerical_failures"]]
    failed_heads = len({(row["case"],row["stage"],row["layer"]) for row in failures})
    accepted_pairs = sum(all(pair[stage]["status"]=="accepted" for stage in registry.ROLES) for pair in results)
    heads = sum(len(row["layers"]) for row in reviewed)
    endpoint_count = sum(row["independent_selected_probability_roles"] for row in reviewed)
    complete = (accepted_pairs==54 and heads==432 and endpoint_count==864 and not failures
                and ledger["status"]=="executed" and ledger["verified_updates"]==scope["planned_total_updates"])
    output_names.extend(summaries.write_outputs(out,results,include_contrasts=False,scope_text=
        "New v2 heads on both fixed backbone stages; same-scene development evidence with reused validation."))
    directory = out/"paired-status-v2"
    output_names.extend(str(Path(directory.name)/name) for name in
                        statuses.write_outputs(directory,results,execution_sha,source_v2_execution=True))
    output_names.extend(trajectory_tables(out,results))
    all_pins = {**evidence.pins,**pins}
    check_files(all_pins)
    require_frozen_code(code)
    path = out/"receipt.json"
    result = {"policy":registry.POLICY,"status":"passed" if complete else "incomplete_or_failed",
        "commit":code["commit"],"source_hash":code["source_hash"],"synthetic":scope["synthetic"],
        "scope_sha256":scope_sha,"execution_sha256":execution_sha,"sources":scope["sources"],
        "verified_new_probe_updates":ledger["verified_updates"],"updates_by_stage":ledger["updates_by_stage"],
        "accepted_pairs":accepted_pairs,"reviewed_heads":heads,"accepted_heads":heads-failed_heads,
        "independent_selected_probability_endpoints":endpoint_count,"numerical_failures":failures,
        "cases":[{"case":pair["case"],"scene":pair["role"]["target"],"stages":{
            stage:pair[stage]["status"] for stage in registry.ROLES}} for pair in results],
        "v1_heads_reused":0,"completed_resume_updates":0,"audit_probe_updates":0,"backbone_updates":0,"bn_updates":0,
        "raw_gt_deserialized":False,"query_labels_accessed":False,"new_model_forwards":0,
        "nonselected_event_probability_replays":0,"protected_files":all_pins,"protected_files_unchanged":True,
        "outputs":core.pins_for(out,output_names),"wall_seconds":time.monotonic()-started}
    write_json(path,result)
    if complete:
        write_json(out/"acceptance.json",{"policy":registry.POLICY,"status":"passed","synthetic":scope["synthetic"],
            "commit":code["commit"],"scope_sha256":scope_sha,"execution_sha256":execution_sha,"review_sha256":hash_file(path),
            "accepted_pairs":54,"accepted_heads":432,"independent_selected_probability_endpoints":864,
            "verified_new_probe_updates":ledger["verified_updates"],"updates_by_stage":ledger["updates_by_stage"],
            "scope":"Same-scene development probe evidence with reused validation."})
    return path
