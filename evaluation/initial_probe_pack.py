"""Authenticate pre-adaptation archives under a separate producer and pack policy."""
from pathlib import Path

from evaluation import authenticated_probe_pack as packs, frozen_probe_audit as audit, frozen_probe_registry as reg
from evaluation import frozen_representation_capture as original
from g4_support_context import read
from model.research_config import digest
from research_runtime import hash_file


CAPTURE_POLICY = "e0_frozen_full54_pre_adaptation_capture_v1"
POLICY = "authenticated_full54_pre_adaptation_train_val_probe_pack_v1"
PRODUCER = "77242952ded4fdbaf0a566a59fa5cd52a9ab6c51"
SOURCE = "865407faf7b10c18a3a44de47fdf4ab51012f47479870dd8ba47a029ca03b524"
OPTIONS = {"maximum_regions":1024,"seed":64,"variance_floor":1e-24}


def zero_updates(row):
    return (row["optimizer_updates"] == row["bn_updates"] == 0
        and row["raw_labels_deserialized"] is row["new_query_release"] is False)


def spectrum(bound, terminal):
    rows = {}
    for stage in ("stage1","stage2"):
        initial, selected = [read(pack[0].parent/(stage+"-summary.json")) for pack in (bound,terminal)]
        keys = ("maximum_regions","seed","sampled_region_indices","sampled_region_masses")
        matched = all(initial[k] == selected[k] for k in keys)
        for branch in ("cnn","graph"):
            before, after = (x[branch]["effective_dimension"] for x in (initial,selected))
            rows[stage+"_"+branch] = {"regions":initial["sampled_regions"] if matched else None,
                "initial":before,"terminal":after,"sample_alignment":"matched" if matched else "different",
                "terminal_minus_initial":after-before if matched and before is not None and after is not None else None}
    return rows


def authenticate(path, sha, scope, terminals, *, synthetic):
    path = Path(path).resolve(strict=True)
    batch = reg.pinned_json(path,sha)
    producer = batch["producer"]
    reg.require(path.name == "receipt.json" and batch["policy"] == CAPTURE_POLICY
        and batch["status"] in ("passed","incomplete_or_failed") and batch["initialization_only"] is False
        and batch["base_commit"] == packs.CAPTURE_COMMIT and batch["base_source_hash"] == packs.CAPTURE_SOURCE
        and producer["commit"] == PRODUCER and producer["source_hash"] == SOURCE and producer["dirty"] is False
        and digest(producer["files"]) == SOURCE and batch["options"] == OPTIONS and zero_updates(batch)
        and batch["protected_files_unchanged"] is True, "Unapproved pre-adaptation capture producer or boundary")
    freeze_sha, analysis_sha = scope["freeze_sha256"], scope["analysis_sha256"]
    reg.require((batch["freeze_sha256"],batch["analysis_sha256"]) == (freeze_sha,analysis_sha),
                "Pre-adaptation and terminal frozen boundary differs")
    frozen, _, launch, boundary = original._authenticate_boundary(scope["freeze"],freeze_sha,scope["analysis"],analysis_sha,producer)
    reg.require(batch["protected_files"] == boundary and batch["boundary_sha256"] == digest(boundary)
        and [row["case"] for row in batch["cases"]] == [row["name"] for row in scope["cases"]]
        and len(batch["cases"]) == 54 and batch["model_forwards"] == sum(r["status"]=="captured" for r in batch["cases"])
        and (batch["status"] == "passed") == (batch["model_forwards"]==54), "Incomplete pre-adaptation roster or boundary")
    common = {(row["target"],row["seed"]):row for row in launch["common_initials"]}
    reg.require(len(common) == len(launch["common_initials"]) == 9, "Nine original shared initializations required")
    all_pins = {**boundary,str(path):sha}
    cases = []
    for captured, selected in zip(batch["cases"],scope["cases"],strict=True):
        name = selected["name"]
        case, record, _, _, local = original._authenticate_case(frozen,launch,boundary,name)
        role = {"name":name,"target":selected["target"],"seed":selected["seed"],"method":selected["method"],
            "mode":selected["mode"],"case_sha256":selected["case_sha256"],"family_hash":selected["family_hash"],
            "initial_state_sha256":case["initial_state_sha256"],"initial_step":0,
            "terminal_model_sha256":selected["primary_sha256"],"terminal_step":selected["selected_step"],
            "phase_hashes":selected["phase_hashes"],"released_rows_sha256":selected["released_rows_sha256"],
            "observation_Q_sha256":record["prepared_fields"]["Q"]}
        if captured["status"] != "captured":
            reg.require(captured["status"] == "failed", "Unfinished pre-adaptation case must remain pending")
            cases.append(({**role,"disposition":"missing_capture","capture_error":captured["error"]},None,{}))
            continue
        receipt_path = path.parent/name/"receipt.json"
        reg.require(receipt_path == receipt_path.resolve(strict=True)
            and captured["receipt"] == str(receipt_path), "Pre-adaptation receipt escapes its assigned role")
        receipt = reg.pinned_json(receipt_path,captured["sha256"])
        proof = common[(case["target"],case["seed"])]
        reg.require(receipt["policy"] == CAPTURE_POLICY and receipt["status"] == "passed"
            and receipt["stage_role"] == "pre_adaptation" and receipt["case"] == name
            and receipt["producer_commit"] == PRODUCER and receipt["producer_source_hash"] == SOURCE
            and receipt["case_sha256"] == selected["case_sha256"] and receipt["selected_step"] == 0
            and receipt["model_forwards"] == captured["model_forwards"] == 1 and zero_updates(receipt)
            and receipt["initialization"] == case["initialization"] and receipt["source_spec"] == case["source_spec"]
            and receipt["common_fresh"] == proof and receipt["graph_binding"] == case["graph_binding"]
            and receipt["initial_state_sha256"] == case["initial_state_sha256"]
                == receipt["evidence"]["state_sha256"] == case["initialization"]["initial_state_sha256"]
            and case["initialization"]["head_seed"] == case["seed"]+1000
            and case["target_context"]["query_labels_accessed"] is False
            and receipt["evidence"]["registered_state_flags_rng_inputs_unchanged"] is True
            and receipt["evidence"]["stage_calls"] == dict.fromkeys(original.STAGES,1)
            and receipt["options"] == OPTIONS and set(receipt["outputs"]) == packs.OUTPUTS
            and (receipt["freeze_sha256"],receipt["analysis_sha256"],receipt["boundary_sha256"])
                == (freeze_sha,analysis_sha,batch["boundary_sha256"]), "Pre-adaptation role or state proof differs")
        bound = packs.bind_releases(receipt_path,captured["sha256"],receipt,role,case,record,boundary,local_pins=local)
        terminal = terminals[name]
        reg.require(fingerprint_rows(bound) == fingerprint_rows(terminal)
            == (role["released_rows_sha256"],role["phase_hashes"],role["observation_Q_sha256"])
            and bound[3]["family_hash"] == terminal[3]["family_hash"] == role["family_hash"]
            and bound[1]["evidence"]["observation"] == terminal[1]["evidence"]["observation"],
            "Initial/terminal role labels, rows or original Q differ")
        family_path = str(Path(case["inputs"]["family"]).resolve(strict=True))
        local = {**local, **packs.capture_files(bound),family_path:boundary[family_path]}
        all_pins.update(local)
        cases.append(({**role,"disposition":"new","capture_receipt":str(receipt_path),
            "capture_receipt_sha256":captured["sha256"],"regional_effective_dimensions":{} if synthetic else spectrum(bound,terminal)},
            bound,local))
    identity = {"policy":CAPTURE_POLICY,"path":str(path),"sha256":sha,"commit":PRODUCER,"source_hash":SOURCE,
                "model_forwards":batch["model_forwards"]}
    return launch,cases,all_pins,identity


def fingerprint_rows(bound):
    _, _, record, _, phases, overlay, _, _ = bound
    return reg.fingerprint(overlay),{key:digest(value) for key,value in phases.items()},record["prepared_fields"]["Q"]


def contract(case):
    return {"policy":POLICY,"stage_role":"pre_adaptation","initial_state_sha256":case["initial_state_sha256"],
            "initial_step":0}


def identity(bound):
    path,receipt,_,family,phases,_,case,_ = bound
    return {**contract(case),"capture_receipt_sha256":hash_file(path),"case":case["name"],
        "freeze_sha256":receipt["freeze_sha256"],"analysis_sha256":receipt["analysis_sha256"],
        "family_hash":family["family_hash"],"phase_hashes":{key:digest(value) for key,value in phases.items()},
        "query_labels_accessed":False,"label_authorization":"accepted_case_family_K10_V5_phases"}


def pack(bound, *, local_only=False):
    return packs.pack_features(bound,identity(bound),local_only=local_only)


def independent(pack, bound, scope, case, local):
    reg.require(pack.get("identity") == identity(bound)
        and pack["identity"]["capture_receipt_sha256"] == case["capture_receipt_sha256"],
        "Authorized pre-adaptation pack identity differs")
    return audit.audit_features(pack,bound,scope,local)
