"""Metadata-first admission of externally authorized completed dynamic route donors."""
import json
import platform
from pathlib import Path

import numpy as np
import scipy
import torch

from model.research_config import canonical_json, digest
from research_runtime import external_directory, hash_file
from transfer import registered_development as dev
from transfer.graph_route_context import route_state_hash
from transfer.graph_route_training import GraphRouteRun


CATALOG = "wp06_registered_route_candidate_catalog_v1"
ORIGIN = "wp06_development_route_provenance_v1"


def require(ok,message):
    if not ok:
        raise ValueError(message)


def read(pin):
    return json.loads(dev._read(pin))


def runtime_identity():
    return {"python":platform.python_version(),"torch":torch.__version__,"torch_git":torch.version.git_version,
            "numpy":np.__version__,"scipy":scipy.__version__,"machine":platform.machine(),
            "torch_threads":torch.get_num_threads(),"torch_default_dtype":str(torch.get_default_dtype()),
            "deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision":torch.get_float32_matmul_precision(),"device":"cpu"}


def producer_compatibility(producer,consumer):
    require(not producer["dirty"] and digest(producer["files"])==producer["source_hash"],
            "Producer source manifest differs or was dirty")
    require(not consumer["dirty"] and digest(consumer["files"])==consumer["source_hash"],
            "Consumer source manifest differs or was dirty")
    if producer["commit"]==consumer["commit"]:
        require(producer["source_hash"]==consumer["source_hash"],"Same commit has a different source fingerprint")
        return "exact_source"
    # code_snapshot covers every Python module and repository config. Compare
    # both directions so a newly introduced module cannot change imports.
    original = {name:sha for name,sha in producer["files"].items() if not name.startswith("tests/")}
    current = {name:sha for name,sha in consumer["files"].items() if not name.startswith("tests/")}
    require(original and original==current,"Historical route runtime source is not byte-compatible")
    return "producer_runtime_files_identical"


def catalog(pin):
    value = read(pin)
    require(set(value)=={"policy","candidates"} and value["policy"]==CATALOG
        and isinstance(value["candidates"],list) and value["candidates"]
        and len({row["name"] for row in value["candidates"]})==len(value["candidates"])
        and all(set(row)=={"name","registration","attempt","provenance","source_issue"} and row["name"]
                for row in value["candidates"]), "Exact pinned candidate catalog required")
    return value["candidates"]


def file_pins(development,raw):
    path = Path(development["path"]).resolve(strict=True)
    dev._released_record(raw["record"])
    require(raw["version"]==1 and raw["excluded"]=="all_non_development_observations_and_original_GT",
            "Only original development-only observation transport is allowed")
    roles = [tile for tile in raw["record"]["roles"] if tile["role"] in ("devtrain","devval")]
    require(len(roles)==len(raw["blocks"]) and {tile["role"] for tile in roles}=={"devtrain","devval"},
            "Missing entire development role or context")
    files = [development]
    for tile,row in zip(roles,raw["blocks"],strict=True):
        name = f'tile-{tile["id"]}.npy'
        target = (path.parent/name).resolve(strict=True)
        require(set(row)=={"id","role","name","sha256"} and
            (row["id"],row["role"],row["name"])==(tile["id"],tile["role"],name)
            and target.parent==path.parent, "Missing or reordered training/validation block")
        files.append({"path":str(target),"sha256":row["sha256"]})
    return roles,files


def check_pins(pins):
    for pin in pins:
        path = Path(pin["path"]).resolve(strict=True)
        external_directory(path)
        require(hash_file(path)==pin["sha256"],f"Original input hash differs: {path}")


def metadata(candidate):
    registration_pin,attempt_pin = candidate["registration"],candidate["attempt"]
    require(registration_pin is not None and attempt_pin is not None,
            "Original registered GraphRouteRun and execution status are both required")
    registration,attempt = read(registration_pin),read(attempt_pin)
    require(set(registration)=={"version","expected_commit","source","metadata","development","plan","bank","bank_hash",
                "lr","torch_seed","contract_hash","spec"} and registration["version"]==1,
            "Original registered GraphRouteRun schema required")
    source_metadata = read(registration["metadata"])
    raw = read(registration["development"])
    roles,development = file_pins(registration["development"],raw)
    spec,record = registration["spec"],raw["record"]
    budget = registration["plan"]["updates"]
    acceptance = attempt.get("acceptance") or {}
    summary = {"scene":raw["dataset"],"model_seed":registration["torch_seed"],
        "exposure_seed":registration["plan"].get("seed"),"support_seed":record["train_support"]["seed"],
        "validation_support_seed":record["validation_support"]["seed"],"classes":len(record["classes"]),
        "k_train":len(record["train_support"]["labels"])//len(record["classes"]),
        "space_protocol":record["kind"],"parent_plan_hash":record["parent_plan_hash"],
        "validation_hash":record["validation_hash"],"training_labels":record["training_labels"],
        "validation_labels":record["validation_labels"],"source_issue":candidate["source_issue"],
        "source":registration["source"],"source_metadata":registration["metadata"],
        "source_model_config_sha256":digest(source_metadata),"source_state":spec["source_state"],
        "fixed_training_spec_sha256":digest({k:v for k,v in spec.items() if k!="graph_route"}),
        "lr":registration["lr"],
        "route":spec["graph_route"]["kind"],"bank":registration["bank"],"budget":budget,
        "completed":attempt.get("completed_updates"),"status":attempt.get("status"),
        "selected_step":acceptance.get("selected_step"),"selected_state":acceptance.get("selected_state_hash"),
        "producer_commit":attempt.get("code",{}).get("commit"),"producer_source_hash":attempt.get("code",{}).get("source_hash"),
        "devtrain_tiles":[t["id"] for t in roles if t["role"]=="devtrain"],
        "devval_tiles":[t["id"] for t in roles if t["role"]=="devval"],"development":registration["development"],
        "contract_hash":registration["contract_hash"],"exposure_plan_hash":registration["plan"]["plan_hash"],
        "selected_validation_history":"fused_head_only"}
    return summary,registration,attempt,development


def authorize(candidate,code,*,allow_synthetic=False):
    summary,registration,attempt,development = metadata(candidate)
    provenance_pin = candidate["provenance"]
    require(provenance_pin is not None,"Independent data provenance or producer runtime evidence is missing")
    origin = read(provenance_pin)
    require(origin.get("policy")==ORIGIN and origin.get("status")=="accepted"
        and type(origin.get("synthetic")) is bool and (allow_synthetic or not origin["synthetic"]),
        "Candidate has synthetic or unapproved development provenance")
    controls = ("scene","model_seed","support_seed","validation_support_seed","k_train","space_protocol",
                "parent_plan_hash","validation_hash","source_issue","producer_commit","producer_source_hash",
                "source","source_metadata","development","contract_hash")
    require(all(origin.get(key)==summary[key] for key in controls) and type(origin["source_issue"]) is int
        and origin["source_issue"]>0 and origin["registration"]==candidate["registration"]
        and origin["attempt"]==candidate["attempt"] and origin["runtime"]==runtime_identity()
        and origin["raw_gt_accessed"] is origin["query_accessed"] is False,
        "Declared scene, seeds, release, producer runtime or original custody differs")
    require(registration["bank"] is registration["bank_hash"] is None and summary["route"]=="dynamic",
            "A completed dynamic donor is required; fixed bank is ineligible")
    acceptance = attempt.get("acceptance") or {}
    budget = summary["budget"]
    require(attempt.get("status")=="completed" and attempt.get("registration")==candidate["registration"]
        and attempt.get("contract_hash")==registration["contract_hash"]
        and attempt.get("query")==acceptance.get("query")==0
        and type(budget) is int and budget>0
        and attempt.get("total_budget")==attempt.get("completed_updates")==attempt.get("target_updates")
            ==acceptance.get("updates")==budget
        and acceptance.get("status")=="completed_execution_evidence" and acceptance.get("optimizer_steps_verified") is True
        and type(summary["selected_step"]) is int and 1<=summary["selected_step"]<=budget
        and attempt.get("checkpoint") is not None
        and registration["expected_commit"]==summary["producer_commit"], "Donor has no completed original dynamic contract")
    compatibility = producer_compatibility(attempt["code"],code)
    pins = [candidate["registration"],candidate["attempt"],provenance_pin,registration["source"],
            registration["metadata"],attempt["checkpoint"],*development]
    check_pins(pins)
    return {**summary,"synthetic":origin["synthetic"],"compatibility":compatibility,"runtime":origin["runtime"],
            "checkpoint":attempt["checkpoint"]},registration,attempt,pins


def restore(candidate,code,*,allow_synthetic=False):
    summary,reg,status,pins = authorize(candidate,code,allow_synthetic=allow_synthetic)
    _,reference,files = dev._load_registration(candidate["registration"])
    require(all(file in pins for file in files) and reference.bank is None,"Registered source/contexts changed")
    checkpoint = status["checkpoint"]
    run = GraphRouteRun.load(reference.source,reference.scene,reference.record,checkpoint["path"],plan=reference.plan,
        expected_sha256=checkpoint["sha256"],expected_contract_hash=reference.contract_hash)
    require(canonical_json(run.finish())==canonical_json(status["acceptance"])
        and route_state_hash(run.source)==summary["source_state"], "Selected execution or original source state differs")
    check_pins(pins)
    return summary,run,reg,pins
