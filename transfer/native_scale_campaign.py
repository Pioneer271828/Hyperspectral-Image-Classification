"""Authenticate completed native training, freeze all endpoints, and score held-out conditions."""

from copy import deepcopy
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
from time import perf_counter

import torch

from joint_training import _run_lock
from model.research_config import canonical_json, digest
from research_runtime import tensor_hash
from transfer import native_scale_training as training, registered_development as transport
from transfer.development_bank import _atomic_new
from transfer.development_blocks import array_hash, read_pin
from transfer.native_scale_fixture import Fixture, INTERVENTIONS, _CONFIRM, condition_for, confirmation_catalog
from transfer.native_scale_observation import forward, head_hashes
from transfer.native_scale_supervision import metrics
from transfer.spatial_multiscale import ARMS
from transfer.xuzhou_multiscale_audit import _exact


PHASES = ("full","prefix","resume","completed")
CLASSES = (9,22)


def case_names():
    return tuple([f"{c}-{s}-{arm}" for c in CLASSES for s in training.SEEDS for arm in ARMS]
                 + [f"{c}-64-joint-hard" for c in CLASSES])


def read(pin,root):
    return json.loads(read_pin(pin,root))


def packet(pin,root):
    return torch.load(io.BytesIO(read_pin(pin,root)),map_location="cpu",weights_only=True)


def register(root,out,producer):
    out.mkdir()
    cases = {}
    for name in case_names():
        classes,seed,arm,*hard = name.split("-")
        spec = training.contract(int(classes),int(seed),arm,producer,mode="high_purity" if hard else "mixture")
        cases[name] = transport._json_new(out/f"{name}.json",spec)
    return transport._json_new(out/"matrix.json",dict(producer=producer,cases=cases,
        main_budget=9216,hard_budget=96,real_updates=0,confirmation="after_complete_freeze_only"))


def matrix(pin,root,producer):
    reg = read(pin,root)
    if reg["producer"]!=producer or set(reg["cases"])!=set(case_names()):
        raise ValueError("Incomplete or unrelated native matrix")
    return reg


def stage(root,matrix_pin,case,phase,producer):
    reg = matrix(matrix_pin,root,producer)
    if case not in reg["cases"] or phase not in PHASES:
        raise ValueError("Unregistered native phase")
    spec_pin = reg["cases"][case]
    spec = read(spec_pin,root)
    folder = Path(matrix_pin["path"]).parent/case
    folder.mkdir(exist_ok=True)
    with _run_lock(folder):
        previous = {"resume":"prefix","completed":"resume"}.get(phase)
        prior = None
        if previous:
            receipt = read(transport._pin(folder/f"{previous}.json"),root)
            if (receipt["case"],receipt["phase"],receipt["registration"])!=(case,previous,spec_pin):
                raise ValueError("Native predecessor is unrelated")
            prior = packet(receipt["checkpoint"],root)
        out = folder/f"{phase}.pt"
        result = training.execute(spec,phase,out,producer,prior)
        return transport._json_new(folder/f"{phase}.json",dict(**result,case=case,
                                    registration=spec_pin,checkpoint=transport._pin(out)))


def _same_scores(replayed,event):
    for row in replayed["rows"]:
        for score in row["scores"].values():
            for key in ("projected_confusion","coarse_hard_confusion","coarse_hard_count","coarse_hard_correct"):
                score.pop(key)
    return _exact(replayed,event)


def audit(root,matrix_pin,out,producer):
    reg = matrix(matrix_pin,root,producer)
    result = {}
    exposed = {}
    for case,spec_pin in reg["cases"].items():
        spec = read(spec_pin,root)
        folder = Path(matrix_pin["path"]).parent/case
        rows = [read(transport._pin(folder/f"{phase}.json"),root) for phase in PHASES]
        if any((r["case"],r["phase"],r["registration"])!=(case,phase,spec_pin)
               for r,phase in zip(rows,PHASES,strict=True)):
            raise ValueError("Stage identity changed")
        full,prefix,resumed,completed = (packet(row["checkpoint"],root) for row in rows)
        updates = [spec["steps"],spec["prefix"],spec["steps"]-spec["prefix"],0]
        if ([row["updates"] for row in rows]!=updates or not _exact(full,resumed) or not _exact(full,completed)
                or not _exact(full["history"][:spec["prefix"]],prefix["history"])
                or not _exact(full["validation"][:len(prefix["validation"])],prefix["validation"])):
            raise ValueError(f"Native continuation differs for {case}")
        schedule = tuple((r["tile"],r["observation"],r["labels"],r["supervised_area"]) for r in full["history"])
        key = (spec["classes"],spec["seed"],spec["supervision"])
        if schedule != exposed.setdefault(key,schedule):
            raise ValueError("Paired native exposures differ")
        first = min(full["validation"],key=lambda row:row["fused_nll"])
        if full["selected"]["step"]!=first["step"]:
            raise ValueError("Native first-min selection differs")
        endpoints = {}
        for name,state,event in (("selected",full["selected"]["state"],first),
                                 ("last",full["model"],full["validation"][-1])):
            replay = training.Run(spec,producer)
            replay.model.load_state_dict(state,strict=True)
            tested = replay.evaluate(event["step"],detailed=True)
            if not _same_scores(deepcopy(tested),event):
                raise ValueError("Independently reconstructed endpoint differs")
            endpoints[name] = tested
        result[case] = dict(exact=True, stages=rows, history=full["history"],
                            validation=full["validation"],endpoints=endpoints,
                            clocks=[row["step"] for row in full["validation"]])
    return transport._json_new(out,dict(producer=producer,matrix=matrix_pin,cases=result,main_updates=9216,
                                      hard_updates=96,real_updates=0,confirmation_reads=0))


def freeze(root,matrix_pin,audit_pin,out,producer):
    reg = matrix(matrix_pin,root,producer)
    evidence = read(audit_pin,root)
    if (evidence["producer"]!=producer or evidence["matrix"]!=matrix_pin or set(evidence["cases"])!=set(reg["cases"])
            or (evidence["main_updates"],evidence["hard_updates"],evidence["confirmation_reads"])!=(9216,96,0)):
        raise ValueError("Native freeze requires an accepted complete audit")
    cases = {}
    for name in reg["cases"]:
        r = evidence["cases"][name]
        if not r["exact"] or len(r["endpoints"])!=2:
            raise ValueError("Missing accepted native endpoint")
        full = r["stages"][0]
        spec = read(reg["cases"][name],root)
        if full["updates"]!=spec["steps"] or r["stages"][-1]["updates"]!=0:
            raise ValueError("Native freeze encountered incomplete training")
        ckpt = packet(full["checkpoint"],root)
        if ckpt["registration"]!=digest(spec):
            raise ValueError("Frozen model does not match its registered case")
        cases[name] = dict(registration=reg["cases"][name],checkpoint=full["checkpoint"],
                           selected=ckpt["selected"]["step"],last=spec["steps"],
                           selected_state=_state_digest(ckpt["selected"]["state"]),
                           last_state=_state_digest(ckpt["model"]))
    return transport._json_new(out,dict(producer=producer,matrix=matrix_pin,audit=audit_pin,cases=cases,
        catalog=[list(key) for key in confirmation_catalog()],interventions=list(INTERVENTIONS),
        effect=dict(training.EFFECT),real_updates=0,confirmation_reads=0))


def _state_digest(state):
    return digest({key:tensor_hash(value) for key,value in state.items()})


def _confirmation_json(path, value):
    _atomic_new(path, (canonical_json(value)+"\n").encode())
    return transport._pin(path)


@torch.no_grad()
def confirm(root, freeze_pin, out, producer):
    root = Path(root).resolve(strict=True)
    out = Path(out).resolve()
    if out==root or not out.is_relative_to(root):
        raise ValueError("Confirmation output must be inside the authorized runs root")
    if out.exists():
        raise ValueError("Confirmation output is single-use")
    freeze_pin = dict(freeze_pin)
    reg = read(freeze_pin,root)
    pinned = matrix(reg["matrix"],root,producer)
    evidence = read(reg["audit"],root)
    if (reg["producer"]!=producer or set(reg["cases"])!=set(pinned["cases"])
            or evidence["matrix"]!=reg["matrix"] or evidence["producer"]!=producer
            or set(evidence["cases"])!=set(pinned["cases"]) or reg["effect"]!=training.EFFECT
            or reg["confirmation_reads"]!=0 or reg["catalog"]!=[list(x) for x in confirmation_catalog()]
            or reg["interventions"]!=list(INTERVENTIONS)):
        raise ValueError("Confirmation requires a matching completed freeze and untouched catalog")
    # Authenticate every endpoint before creating or reading the held-out stream.
    checkpoints = {}
    for name, item in reg["cases"].items():
        accepted = evidence["cases"][name]
        if (not accepted["exact"] or item["registration"]!=pinned["cases"][name]
                or item["checkpoint"]!=accepted["stages"][0]["checkpoint"]):
            raise ValueError("Unaccepted native confirmation model")
        spec = read(item["registration"],root)
        checkpoint = packet(item["checkpoint"],root)
        if (checkpoint["registration"]!=digest(spec) or checkpoint["cursor"]!=spec["steps"]
                or item["selected"]!=checkpoint["selected"]["step"] or item["last"]!=checkpoint["cursor"]
                or item["selected_state"]!=_state_digest(checkpoint["selected"]["state"])
                or item["last_state"]!=_state_digest(checkpoint["model"])):
            raise ValueError("Wrong model, checkpoint or frozen endpoint")
        checkpoints[name] = spec, checkpoint
    folder = Path(reg["matrix"]["path"]).parent
    completed = folder/"confirmation-complete.json"
    claim_path = folder/"confirmation-claim.json"
    with _run_lock(folder):
        if completed.exists():
            raise ValueError("Frozen matrix confirmation was already consumed")
        if claim_path.exists():
            raise ValueError("Frozen matrix confirmation was already claimed; inspect its existing artifacts")
        out.mkdir()
        # Make the output directory itself durable before binding it in the claim.
        parent = os.open(out.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        # Publish and sync the exclusive claim before touching the held-out
        # stream. An exception or interruption must never release this claim.
        claim = _confirmation_json(claim_path,dict(
            kind="wp07-native-confirmation-claim-v1",matrix=reg["matrix"],freeze=freeze_pin,
            producer=producer,output=str(out)))
        started = perf_counter()
        # Materialize each held-out field/profile/view once for the entire campaign.
        observations = {}
        for classes in CLASSES:
            fixture = Fixture(classes,"confirmation",_confirmation=_CONFIRM)
            observations[classes] = [(tile,factor,band,profile,fixture.materialize(tile,factor,band,profile,producer))
                for tile in fixture.tiles for factor,band,profile in confirmation_catalog()]
        materialization_seconds = perf_counter()-started
        cases = {}
        graph_seconds = forward_seconds = 0.
        for name,(spec,checkpoint) in checkpoints.items():
            run = training.Run(spec,producer)
            endpoints = {}
            for endpoint,state in (("selected",checkpoint["selected"]["state"]),("last",checkpoint["model"])):
                run.model.load_state_dict(state,strict=True)
                run.model.eval()
                rows = []
                for tile,factor,band,profile,(actual,labels,targets) in observations[spec["classes"]]:
                    predictions = {}
                    for intervention in INTERVENTIONS:
                        condition = condition_for(actual,intervention)
                        timed = {}
                        output,graph = forward(run.model,actual,producer,"confirmation",condition=condition,timed=timed)
                        graph_seconds += sum(timed["graph_seconds"])
                        forward_seconds += timed["forward_seconds"]
                        predictions[intervention] = dict(condition=asdict(condition),heads=head_hashes(output,actual),
                            graph=graph,scores=metrics(output,actual,labels,targets,detailed=True))
                    first = predictions["matched"]["graph"]
                    if any(value["graph"]["q"]!=first["q"] or value["graph"]["q_shape"]!=first["q_shape"]
                           for value in predictions.values()):
                        raise ValueError("Physical intervention changed the actual observation/Q")
                    if spec["arm"] not in ("metadata","joint") and any(
                        value["heads"]!=predictions["matched"]["heads"] for value in predictions.values()
                    ):
                        raise ValueError("Conditions affected a condition-independent arm")
                    rows.append(dict(tile=tile,factor=factor,band=band,profile=profile,
                        observation=actual.receipt,labels=array_hash(labels),supervision=targets.summary(),
                        predictions=predictions))
                endpoints[endpoint] = rows
            cases[name] = endpoints
        result = _confirmation_json(out/"confirmation.json",dict(producer=producer,freeze=freeze_pin,claim=claim,cases=cases,
            materializations=sum(map(len,observations.values())),forward_calls=len(checkpoints)*2*16*len(INTERVENTIONS),
            timing=dict(total_seconds=perf_counter()-started,materialization_seconds=materialization_seconds,
                        forward_seconds=forward_seconds,graph_seconds=graph_seconds),real_updates=0))
        _confirmation_json(completed,dict(producer=producer,matrix=reg["matrix"],freeze=freeze_pin,claim=claim,
                                          confirmation=result))
        return result
