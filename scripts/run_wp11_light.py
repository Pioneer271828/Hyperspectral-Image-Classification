"""Register and execute one isolated CPU light-head training segment."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.wp11_confirmation import validate_full_history
from inference.trained_light import export_trained_light, load_trained_light
from model.research_config import digest
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from transfer.g3_training import state_hash
from transfer.light_head_training import LightHeadRun
from transfer.registered_development import _load_source, _read, load_development


POLICY = "dy-registered-isolated-light-head-v1"


def load_inputs(spec):
    if (set(spec) != {"development", "full_weights", "full_metadata", "full_history", "history_evidence", "training"}
            or set(spec["training"]) != {"updates", "seed", "lr", "class_balance"}):
        raise ValueError("Only isolated development, full model, audited history and explicit training options are allowed")
    scene, record, pins = load_development(spec["development"])
    full = _load_source(spec["full_weights"], spec["full_metadata"])
    history = json.loads(_read(spec["full_history"]))
    if history["history_evidence_sha256"] != spec["history_evidence"]["sha256"]:
        raise ValueError("Full history evidence differs")
    _read(spec["history_evidence"])
    validate_full_history(full, scene, record, history)
    pins.extend(spec[k] for k in ("full_weights", "full_metadata", "full_history", "history_evidence"))
    return full, scene, record, pins


def verify_inputs(pins, code, out):
    require_frozen_code(code)
    if out.exists() or any(out == path or out in path.parents for path in (Path(p["path"]).resolve() for p in pins)):
        raise ValueError("New external output outside input packages required")
    if any(hash_file(p["path"]) != p["sha256"] for p in pins):
        raise ValueError("Pinned input changed")


def plan(specification, out, code):
    spec = json.loads(_read(specification))
    full, scene, record, pins = load_inputs(spec)
    run = LightHeadRun(full, scene, record, **spec["training"])
    verify_inputs([specification, *pins], code, out)
    registration = dict(policy=POLICY, commit=code["commit"], source_hash=code["source_hash"],
                        inputs=spec, contract_hash=run.contract_hash, optimizer_updates=0, query=0)
    out.mkdir()
    write_json(out / "registration.json", registration)
    return registration


def execute(registration, out, code, *, to_step, resume=None):
    value = json.loads(_read(registration))
    if (set(value) != {"policy", "commit", "source_hash", "inputs", "contract_hash", "optimizer_updates", "query"}
            or value["policy"] != POLICY or value["commit"] != code["commit"]
            or value["source_hash"] != code["source_hash"] or (value["optimizer_updates"], value["query"]) != (0, 0)):
        raise ValueError("Registration must bind this exact frozen code and zero-step plan")
    spec = value["inputs"]
    full, scene, record, pins = load_inputs(spec)
    pins.append(registration)
    if resume is None:
        run = LightHeadRun(full, scene, record, **spec["training"])
    else:
        external_directory(resume["path"])
        run = LightHeadRun.load(full, scene, record, resume["path"], expected_sha256=resume["sha256"],
                                expected_contract_hash=value["contract_hash"])
        pins.append(resume)
    if (run.contract_hash != value["contract_hash"]
            or digest({key: run.spec[key] for key in spec["training"]}) != digest(spec["training"])
            or type(to_step) is not int or not run.completed <= to_step <= run.spec["updates"]):
        raise ValueError("Registered light-head contract or absolute step differs")
    verify_inputs(pins, code, out)
    out.mkdir()
    first = run.completed
    started = perf_counter()
    try:
        history = run.advance(to_step)
        checkpoint = out / "training.pt"
        checkpoint_sha = run.save(checkpoint)
        artifact = out / "light.pt"
        artifact_sha = None
        selected = None
        if to_step == run.spec["updates"]:
            artifact_sha = export_trained_light(full, scene, record, checkpoint, artifact,
                                               expected_training_sha256=checkpoint_sha, expected_contract_hash=run.contract_hash)
            selected = load_trained_light(full, artifact, expected_sha256=artifact_sha,
                                          expected_training_sha256=checkpoint_sha,
                                          expected_contract_hash=run.contract_hash).specification
        require_frozen_code(code)
        if any(hash_file(p["path"]) != p["sha256"] for p in pins):
            raise ValueError("Pinned input changed during the segment")
    except Exception as error:
        write_json(out / "receipt.json", dict(status="failed", registration_sha256=registration["sha256"],
                   completed_before=first, attempted_through=to_step, completed_observed=run.completed,
                   query=0, error_type=type(error).__name__, error=str(error)))
        raise
    receipt = dict(status="completed" if artifact_sha else "partial", registration_sha256=registration["sha256"],
                   contract_hash=run.contract_hash, completed=to_step, optimizer_updates=to_step-first, query=0,
                   source_state=state_hash(full), head_state=state_hash(run.head), history_hash=digest(history),
                   selected_step=run.selected_step, selected=selected, training_sha256=checkpoint_sha,
                   light_sha256=artifact_sha, wall_seconds=perf_counter()-started)
    write_json(out / "receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "run"))
    for key in ("expected-commit", "registration", "registration-sha256", "out"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--to-step", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--resume-sha256")
    args = parser.parse_args()
    code = code_snapshot()
    if sys.platform != "linux" or not __debug__ or code["dirty"] or code["commit"] != args.expected_commit:
        raise ValueError("Clean expected Linux light-head checkout required")
    torch.set_num_threads(1)
    pin = dict(path=args.registration, sha256=args.registration_sha256)
    out = external_directory(args.out)
    if args.operation == "plan":
        if any(v is not None for v in (args.to_step, args.resume, args.resume_sha256)):
            parser.error("planning cannot run or resume training")
        plan(pin, out, code)
    else:
        if args.to_step is None or bool(args.resume) != bool(args.resume_sha256):
            parser.error("run requires --to-step and, when resuming, both checkpoint path and hash")
        resume = dict(path=args.resume, sha256=args.resume_sha256) if args.resume else None
        execute(pin, out, code, to_step=args.to_step, resume=resume)


if __name__ == "__main__":
    main()
