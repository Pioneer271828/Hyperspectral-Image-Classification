"""Authenticate the accepted all-scene terminal campaign and nested PaviaU reuse."""
from pathlib import Path

from evaluation import allscene_probe_campaign as ledger, allscene_probe_registry as scenes
from evaluation import allscene_probe_summary as summary, frozen_probe_campaign as paths, frozen_probe_registry as reg
from g4_support_context import check_files
from model.research_config import digest


COMMIT = "c5f61ccd6c12f3aab9d04c94c172aa372a5dd451"
SOURCE = "a00b2c1bd79b7e185aecd9c8223d8c10ec1c611b5ea7cb07ec2594499c5ab8d7"
POLICY = "wp06_allscene_terminal_acceptance_reuse_v1"
SHA = {"scope": "c9134b2c8db670d48170efe5e16e5cf2f36230f632e5b3a23c38cff20cda9484",
       "execution": "423ae72725ce3b1dda9eebb27ffd27aaf6f56252667d64f4d13462f5a2c957f9",
       "review": "536f284b597af97fedd6d8b24aa13f7dc7d93168f9680b7fbc857a90cd995613",
       "acceptance": "c25a0678c3c8295eeeaec24a4e4e13c009d3f5e69182c196ca2abf63ffa8bc45"}


def authenticate(scope_path, execution_path, review_path, *, synthetic=False):
    files = {key: Path(value).resolve(strict=True) for key, value in
             zip(("scope", "execution", "review"), (scope_path, execution_path, review_path), strict=True)}
    files["acceptance"] = files["review"].with_name("acceptance.json")
    reg.require([path.name for path in files.values()] == ["scope.json","execution.json","receipt.json","acceptance.json"],
                "Exact terminal campaign files required")
    scope, execution, review, acceptance = [reg.pinned_json(path,SHA[key]) for key,path in files.items()]
    steps = scope["paths"]["full"][0]
    updates = 54*len(reg.LAYERS)*steps*2
    reg.require((scope["commit"],scope["source_hash"]) == (COMMIT,SOURCE)
        and scope["policy"] == execution["policy"] == review["policy"] == acceptance["policy"] == scenes.POLICY
        and scope["synthetic"] is review["synthetic"] is acceptance["synthetic"] is synthetic
        and not scope["missing_capture_scenes"] and scope["scene_classes"] == scenes.CLASSES
        and scope["layers"] == list(reg.LAYERS) and (synthetic or scope["paths"] == reg.PATHS)
        and scope["columns"] == {stage:reg.probe_columns(64,64,width) for stage,width in (("stage1",128),("stage2",64))}
        and scope["probe_plans"] == {scene:{**reg.PLAN,"classes":classes,"steps":steps} for scene,classes in scenes.CLASSES.items()}
        and scope["validation"] == "every_step_first_minimum_val_nll"
        and scope["expected_probe_updates"] == execution["total_probe_updates"] == review["total_probe_updates"]
            == acceptance["total_probe_updates"] == updates
        and execution["status"] == "executed" and review["status"] == acceptance["status"] == "passed"
        and review["commit"] == COMMIT and review["strict_decisions"] is True
        and review["scope_sha256"] == acceptance["scope_sha256"] == SHA["scope"]
        and review["execution_sha256"] == SHA["execution"] and acceptance["independent_receipt_sha256"] == SHA["review"]
        and review["accepted_cases"] == acceptance["cases"] == 54
        and review["accepted_probes"] == review["reviewed_probes"] == acceptance["probes"] == 216
        and review["failed_probes"] == review["audit_probe_updates"] == review["completed_resume_updates"]
            == review["backbone_updates"] == review["bn_updates"] == 0
        and review["query_labels_accessed"] is review["raw_gt_deserialized"] is False
        and review["protected_files_unchanged"] is True, "Accepted terminal identity, fixed protocol or evidence coverage differs")
    check_files({**review["protected_files"], **review["outputs"]})
    pavia = [scope["history"]["sources"][key]["path"] for key in ("scope","execution","review")]
    launch, cases, pins, nested_identity, nested, _ = scenes.authenticate(scope["freeze"],scope["freeze_sha256"],
        scope["analysis"],scope["analysis_sha256"],scope["capture_index"],scope["capture_index_sha256"],pavia,
        (),synthetic=synthetic)
    reg.require(scope["cases"] == [case for case,_,_ in cases] and scope["history"] == nested_identity
        and scope["protected_file_count"] == len(pins) and scope["protected_files_sha256"] == digest(pins),
        "Terminal registered matrix, protected files or nested history differs")
    _, execution_pins = ledger.validate_run(files["execution"],SHA["execution"],files["scope"],SHA["scope"],
        scope,cases,nested,scope)
    covered = {**pins, **execution_pins, str(files["scope"]):SHA["scope"]}
    proofs, bounds = {}, {}
    for (case,bound,local), executed, reviewed in zip(cases,execution["cases"],review["cases"],strict=True):
        name = case["name"]
        result_path = paths.folder_at(files["review"].parent,name+".json")
        result = reg.pinned_json(result_path,review["outputs"][str(result_path)])
        reg.require(reviewed["case"] == result["case"] == name and result["role"] == case
            and reviewed["status"] == result["status"] == "accepted" and not result["numerical_failures"]
            and result["exact_continuous_split_and_completed"] is True
            and result["updates_by_path"] == {"full":len(reg.LAYERS)*steps,"split":len(reg.LAYERS)*steps,"completed":0}
            and set(result["layers"]) == set(reg.LAYERS), "Terminal independent verdict differs")
        if executed["status"] == "executed":
            folder = paths.folder_at(files["execution"].parent,name)
            _, _, _, _, counts, artifacts = paths.checked_execution(folder,executed["execution_sha256"],
                scenes.case_scope(scope,case),SHA["scope"],case,local)
            reg.require(counts == result["updates_by_path"], "Terminal path accounting differs")
            covered.update({str(folder/"execution.json"):executed["execution_sha256"], **artifacts})
        else:
            reg.require(nested[name]["layers"] == result["layers"], "Nested PaviaU endpoint proof differs")
        proofs[name] = {**result, "probe_updates":2*len(reg.LAYERS)*steps}
        bounds[name] = bound
    reg.require(covered == review["protected_files"]
        and set(review["outputs"]) == {str(files["review"].parent/name) for name in
            (*summary.TABLES, *(name+".json" for name in proofs))}, "Full terminal artifact coverage differs")
    identity = {"policy":POLICY,"commit":COMMIT,"source_hash":SOURCE,"probe_updates":updates,"pavia":nested_identity,
                "sources":{key:{"path":str(path),"sha256":SHA[key]} for key,path in files.items()}}
    return scope,launch,proofs,bounds,identity,{**covered, **review["outputs"], **{str(p):SHA[k] for k,p in files.items()}}
