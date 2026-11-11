"""Fit initial probes once and independently audit them against authenticated terminal results."""
from functools import partial
from pathlib import Path
import time

from evaluation import allscene_probe_campaign as orchestration,allscene_probe_registry as scenes
from evaluation import frozen_probe_campaign as core,frozen_probe_audit as audit,frozen_probe_registry as reg
from evaluation import initial_final_probe_registry as registry,initial_final_probe_summary as summary,initial_probe_pack as pack
from g4_support_context import check_files
from research_runtime import hash_file,require_frozen_code,write_json


def run(scope,sha,out,code):
    return orchestration.run(scope,sha,out,code,authority=registry,
        run_case=partial(core.run_case,writer=pack.pack,contract=pack.contract))


def review_case(row,case,bound,local,scope,scope_sha,run_path):
    folder = core.folder_at(run_path.parent,case["name"])
    data,state,report,probabilities,counts,files = core.checked_execution(folder,row["execution_sha256"],
        scope,scope_sha,case,local,contract=pack.contract)
    reg.require(row["new_probe_updates"] == sum(counts.values()) and counts["completed"] == 0,
                "Pre-adaptation recorded update count differs")
    values = pack.independent(data,bound,scope,case,local)
    scores = audit.audit_probes(data,scope,state,report,probabilities)
    result = {**row,**scores,**values,"role":case,"updates_by_path":counts,
        "status":"failed_numerical_gate" if scores["numerical_failures"] else "accepted",
        "exact_continuous_split_and_completed":True,"audit_probe_updates":0,
        "raw_gt_deserialized":False,"query_labels_accessed":False}
    return result,{str(folder/"execution.json"):row["execution_sha256"],**files}


def review(scope_path,scope_sha,run_path,run_sha,out,code):
    started = time.monotonic()
    scope_path,run_path = (Path(path).resolve(strict=True) for path in (scope_path,run_path))
    scope,launch,cases,inputs,reused,elapsed = registry.load(scope_path,scope_sha,code)
    ledger,artifacts = orchestration.validate_run(run_path,run_sha,scope_path,scope_sha,scope,cases,reused,code)
    out = registry.output(out,launch,scope,scope_path.parent,run_path.parent)
    out.mkdir(exist_ok=False)
    results = []
    for row,(case,bound,local) in zip(ledger["cases"],cases,strict=True):
        name = case["name"]
        if row["status"] == "executed":
            try:
                value,files = review_case(row,case,bound,local,scenes.case_scope(scope,case),scope_sha,run_path)
                artifacts = scenes.merge_pins(artifacts,files)
            except Exception as error:
                value = {**row,"role":case,"status":"failed_review","error":repr(error)}
        else:
            value = {**row,"role":case}
        terminal = reused[name]
        result = {"case":name,"role":case,"initial":value,"terminal":terminal}
        write_json(core.folder_at(out,name+".json"),result)
        results.append(result)
    tables = summary.write_outputs(out,results)
    reviewed = [row for row in results if "layers" in row["initial"]]
    accepted = sum(row["initial"]["status"]=="accepted" for row in results)
    failed_probes = sum(len({fail["layer"] for fail in row["initial"]["numerical_failures"]}) for row in reviewed)
    complete = (accepted == len(reviewed) == len(results) == len(reused) == 54 and failed_probes == 0
        and ledger["status"] == "executed" and ledger["total_probe_updates"] == scope["expected_probe_updates"]
        and ledger["new_probe_updates"] == scope["planned_new_probe_updates"] == scope["reused_probe_updates"])
    pins = scenes.merge_pins(inputs,artifacts)
    check_files(pins)
    require_frozen_code(code)
    receipt = {"policy":registry.POLICY,"status":"passed" if complete else "incomplete_or_failed",
        "commit":code["commit"],"scope_sha256":scope_sha,"execution_sha256":run_sha,"synthetic":scope["synthetic"],
        "accepted_pairs":accepted,"reviewed_initial_probes":4*len(reviewed),
        "accepted_initial_probes":4*len(reviewed)-failed_probes,"historical_terminal_probes":4*len(reused),
        "paired_probe_endpoints":4*(len(reviewed)+len(reused)),"failed_initial_probes":failed_probes,
        "new_probe_updates":ledger["new_probe_updates"],"historical_probe_updates":ledger["historical_probe_updates"],
        "total_probe_updates":ledger["total_probe_updates"],"initial_capture":scope["initial_capture"],"history":scope["history"],
        "cases":[{"case":row["case"],"scene":row["role"]["target"],"initial_status":row["initial"]["status"],
                  "terminal_status":row["terminal"]["status"]} for row in results],
        "audit_probe_updates":0,"completed_resume_updates":0,"backbone_updates":0,"bn_updates":0,
        "raw_gt_deserialized":False,"query_labels_accessed":False,
        "input_authentication_seconds":elapsed,"wall_seconds":time.monotonic()-started,
        "protected_files":pins,"protected_files_unchanged":True,
        "outputs":core.pins_for(out,(*tables,*(row["case"]+".json" for row in results)))}
    path = out/"receipt.json"
    write_json(path,receipt)
    if complete:
        write_json(out/"acceptance.json",{"status":"passed","policy":registry.POLICY,"pairs":54,
            "initial_probes":216,"terminal_probes":216,"probe_endpoints":432,
            "scope_sha256":scope_sha,"independent_receipt_sha256":hash_file(path),
            "new_probe_updates":ledger["new_probe_updates"],"historical_probe_updates":ledger["historical_probe_updates"],
            "synthetic":scope["synthetic"]})
    return path
