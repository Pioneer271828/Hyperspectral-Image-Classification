"""Read-only explanation of the original v1 synthetic long-coverage failure."""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from evaluation.frozen_training_stages import HEADS, VIEWS, file_pin, json_pin, require
from evaluation.native_spectral_control import (closest, column_positions, label_scores, mean_scores,
                                              native_signatures, normalize, support_references, verify_loader)
from model.research_config import digest
from scripts.run_frozen_band_region import _write
from transfer import development_bank as custody
from transfer.g3_training import state_hash
from transfer.native_coverage_queue import source_seed
from transfer.native_long_campaign import MatchedProvider, STEPS
from transfer.native_long_run import packet
from transfer.spatial_coverage_protocol import bank_view

V1_COMMIT = "0df6cf7239484b45d463e4f5d47f519fd8a3aa57"
V1_FREEZE = "d50f8ed7a21461036a80028e8149fc8f78bfab8e238db252af6b3301dad0531c"
V1_REVIEW = "1203d5d8d5ab298c0bac81b1275cbf62d13b33614969b440f8c06ad3dd9c171e"
FEATURES = ("raw","loader","coordinates","vca","pre_pool","pooled","cnn","graph")


def codebook(scene, bands, classes):
    valid = column_positions(bands.valid,"full")
    outside = column_positions(bands.valid,"central_gap")
    central = [band for band in valid if band not in outside]
    rng = np.random.default_rng([source_seed(scene),7919])
    centers = rng.normal(0,1.9,size=(classes,len(bands.valid))).astype(np.float32)
    centers[1,outside] = centers[0,outside]
    centers[0,central], centers[1,central] = -2.5,2.5
    centers[2:4,outside] = np.asarray([-1,1])[:,None]*np.linspace(.8,2.6,len(outside))
    centers[2:4,central] = np.asarray([-2,2])[:,None]
    return centers


def verify_fit_statistics(arrays, support, statistics):
    low = np.min(np.concatenate([arrays[key].reshape(-1,arrays[key].shape[-1]) for key in support]),axis=0)
    high = np.max(np.concatenate([arrays[key].reshape(-1,arrays[key].shape[-1]) for key in support]),axis=0)
    require(np.array_equal(low,np.asarray(statistics["minimum"])) and np.array_equal(high,np.asarray(statistics["maximum"])),
            "Sealed fit statistics did not originate from support alone")
    return low,high


def old_observations(provider):
    bank = provider.bank
    bands = bank.release.bands
    roles = provider.roles
    rows = {role:[bank.role_row("devtrain" if role == "support" else "devval",i) for i in ids]
            for role,ids in roles.items()}
    directory = Path(bank.pin["root"])
    members = bank.members
    arrays = {row["id"]:custody._decode(directory,members[row["id"]]["member"],members[row["id"]]["content"],
                   row,len(bands.band_ids)) for blocks in rows.values() for row in blocks}
    low,high = verify_fit_statistics(arrays,roles["support"],bank.release.statistics)
    centers = codebook(provider.scene,bands,provider.classes)
    selected = {}
    maxima = {}
    for role, blocks in rows.items():
        for band in VIEWS:
            view = []
            inverse_error, amplitude = [], []
            positions = column_positions(bands.valid,band)
            for row in blocks:
                observation, labels, targets = bank_view(bank,row,provider.producer,band)
                truth, index, raw, got, error = verify_loader(row,bank.release.shape,arrays[row["id"]],low,high,
                                                               positions,observation,labels,targets,bands)
                raw,got = (z.reshape(-1,z.shape[-1])[index] for z in (raw,got))
                indexed_centers = centers[:,positions]
                signal = indexed_centers[truth-1,:]
                amplitude.append(float(np.max(np.abs(raw-signal))))
                inverse_error.append(error)
                coords = observation.view.observation.coordinates.reshape(-1,2)[index]
                view.append(dict(observation=observation,index=index,truth=truth,raw=raw,loader=got,coordinates=coords))
            require(max(amplitude) < .25, "Archived class spectral signatures disagree with released coordinates")
            selected[(role,band)] = view
            maxima[f"{role}/{band}"] = dict(signature_error=max(amplitude),inverse_error=max(inverse_error),
                    below_zero=sum(int((r["loader"] < 0).sum()) for r in view),
                    above_one=sum(int((r["loader"] > 1).sum()) for r in view),
                    normalized_min=min(float(r["loader"].min()) for r in view),
                    normalized_max=max(float(r["loader"].max()) for r in view),
                    point_counts=np.bincount(np.concatenate([r["truth"] for r in view]),
                                             minlength=provider.classes+1)[1:].tolist())
    baseline = {}
    for role in ("support","calibration","evaluation"):
        for band in VIEWS:
            positions = column_positions(bands.valid,band)
            data,train = selected[(role,band)],selected[("support",band)]
            stack = lambda group,key: np.concatenate([row[key] for row in group])
            y,y0 = stack(data,"truth"),stack(train,"truth")
            physical = centers[:,positions]
            norm, _ = normalize(physical,low[positions],high[positions])
            results = {f"{key}_known_mean":label_scores(y,closest(stack(data,key),means),provider.classes)
                      for key,means in (("raw",physical),("loader",norm))}
            for key in ("raw","loader","coordinates"):
                for k,pred in support_references(stack(train,key),y0,stack(data,key),provider.classes).items():
                    results[f"{key}_{k}"] = label_scores(y,pred,provider.classes)
            baseline[f"{role}/{band}"] = results
    return selected, dict(ranges=maxima,reference_scores=baseline,
                          normalization_sources=len(roles["support"]), identical_central_pair=True)


def signatures(model, provider, staged):
    result = {}
    for (role,band),rows in staged.items():
        out = []
        for row in rows:
            actual = row["observation"]
            feats, heads, _ = native_signatures(model,actual,provider.producer,actual.receipt["role"])
            out.append((row,feats,heads))
        result[(role,band)] = out
    return result


def stage_scores(model, provider, staged, *, y_shift=False):
    result = {}
    capture = signatures(model,provider,staged)
    for (role,band), group in capture.items():
        fit = capture[("support",band)]
        y = np.concatenate([row["truth"] for row,_,_ in group])
        yfit = np.concatenate([row["truth"] for row,_,_ in fit])
        stack = lambda items,key: np.concatenate([feats[key][row["index"]] for row,feats,_ in items])
        heads = {head:mean_scores(y,np.concatenate([heads[head][row["index"]] for row,_,heads in group]),
                                   provider.classes) for head in HEADS}
        probes = {}
        for key in FEATURES[3:]:
            for method,points in support_references(stack(fit,key),yfit,stack(group,key),provider.classes).items():
                probes[f"{key}_{method}"] = label_scores(y,points,provider.classes)
        cell = dict(heads=heads, probes=probes, region_counts=[feats["regions"] for _,feats,_ in group],
                    mean_pool_variance_ratio=float(np.mean([feats["pool_variance_ratio"] for _,feats,_ in group])))
        if y_shift and role != "support" and band == "full":
            first_y = min(float(row["coordinates"][:,0].min()) for row,_,_ in fit)
            y0 = min(float(row["coordinates"][:,0].min()) for row,_,_ in group)
            shifted = []
            for row,_,_ in group:
                actual = row["observation"]
                _, values, _ = native_signatures(model,actual,provider.producer,actual.receipt["role"],
                                                  flat_coordinates=y0-first_y)
                shifted.append({key:v[row["index"]] for key,v in values.items()})
            cell["support_aligned_y"] = {key:mean_scores(y,np.concatenate([row[key] for row in shifted]),
                                                         provider.classes) for key in HEADS}
        result[f"{role}/{band}"] = cell
    return result


def run(root, out, producer, freeze_pin, audit_pin):
    require(freeze_pin["sha256"] == V1_FREEZE and audit_pin["sha256"] == V1_REVIEW,
            "Expected the fixed v1 long-coverage freeze and independently accepted readout")
    before = perf_counter()
    archive, review = json_pin(root,freeze_pin),json_pin(root,audit_pin)
    queue = json_pin(root,archive["campaign"])
    require(queue["producer"] == V1_COMMIT and queue["scope"] == review["scope"] == "synthetic_only"
            and review["freeze"] == freeze_pin and review["optimizer_updates"] == archive["updates"] == 18000
            and review["query_accessed"] is review["raw_gt_loaded"] is False,
            "V1 synthetic/archive identity differs")
    cached, verified, cases = {},{},{}
    for name, source in archive["cases"].items():
        provider = MatchedProvider(root,{"commit":V1_COMMIT},queue,name)
        if provider.scene not in cached:
            cached[provider.scene], verified[provider.scene] = old_observations(provider)
        fit = json_pin(root,source)
        spec = json_pin(root,fit["spec"])
        last = packet(root,fit["checkpoint"])["training"]
        stage = last["selected"]["step"]
        full = last["validation"]
        require(last["registration"] == fit["registration"] == digest(spec)
                and len(last["history"]) == STEPS and all(p["band"] == "full" for p in spec["validation"])
                and all(spec["training"][index] == {k:event[k] for k in spec["training"][index]}
                        for index,event in enumerate(last["history"]))
                and min(full,key=lambda x:x["fused_nll"])["step"] == stage == review["cases"][name]["selected"],
                "V1 exposure or full-band first-min selection differs")
        initial_pin = next(pin for pin in fit["checkpoints"] if Path(pin["path"]).name == "clock-0000.pt")
        initial = packet(root,initial_pin)["training"]
        model = provider.model_factory(provider.baseline(provider.seed),"off",spec["module_seed"]).cpu().eval()
        states = (("initial",0,initial["model"]),("selected",stage,last["selected"]["state"]),("last",STEPS,last["model"]))
        diagnostics = {}
        for label,step,state in states:
            model.load_state_dict(state,strict=True)
            finger = state_hash(model)
            scores = stage_scores(model,provider,cached[provider.scene],y_shift=label in ("selected","last"))
            require(state_hash(model) == finger, "Retrospective capture modified a frozen state")
            diagnostics[label] = dict(step=step, fingerprint=finger, bands=scores)
        cases[name] = dict(scene=provider.scene, archived_device=spec["numeric"]["device"], selected_step=stage,
                           native_full_replay=review["cases"][name]["full_validation_bitwise"],
                           exposure_counts={band:sum(p["band"]==band for p in spec["training"]) for band in VIEWS},
                           stages=diagnostics)
    require(len(cases) == 12 and len(verified) == 2, "Exactly twelve previously frozen synthetic cases required")
    return _write(out,dict(status="passed",scope="retrospective_v1",producer=producer["commit"],
                  freeze=freeze_pin,audit=audit_pin,cases=cases,spectral_inputs=verified,
                  wall_seconds=perf_counter()-before,replayed_on="cpu_original_input",new_optimizer_updates=0,
                  query_accessed=False,raw_gt_loaded=False))
