"""Read-only native grids over the original development banks."""

from time import perf_counter

import numpy as np
import torch

from transfer import development_bank as bank, development_bank_custody as custody
from transfer.development_block_probe import VIEWS
from transfer.g3_training import state_hash
from transfer.native_scale_observation import FACTORS, forward, from_bank, head_hashes
from transfer.native_scale_supervision import sparse_targets
from transfer.spatial_multiscale import ARMS, initialize, inventory
from scripts.run_spatial_multiscale import BANK_PRODUCER, NICK, REAL, read


@torch.no_grad()
def scan(root,scene,producer):
    prior = dict(path=str(root/f"wp02-bank-review-fix01/{NICK[scene]}-replay06.json"),sha256=REAL[scene])
    historical = read(prior,root)
    if historical["scene"]!=scene or historical["producer"]!=BANK_PRODUCER or historical["real_updates"]:
        raise ValueError("Original development bank receipt differs")
    intake = custody.PROVIDERS[scene](root)
    opened = bank.Bank(historical["bank"],root,intake.release)
    release = opened.release
    original = intake.baseline
    state = state_hash(original)
    if state!=release.source.get("baseline_state",release.source["state"]):
        raise ValueError("Native scan source model differs")
    arms = ("off",) if scene=="paviaU" else ARMS
    models = {arm:initialize(original,arm,7321) for arm in arms}
    initial = inventory(original,models)
    representatives = set(historical["representatives"])
    older = {(x["context"]["tile"],x["view"]):x for x in historical["rows"]}
    timings = dict(materialize_seconds={},forward_seconds={},graph_seconds={})
    rows = []
    legacy = 0
    old_rows = 0
    start = perf_counter()
    for row in opened.rows.values():
        role, tile = row["role"], row["id"]
        for band,spec in VIEWS.items():
            if tile not in representatives and band!="full":
                continue
            for factor in FACTORS:
                tick = perf_counter()
                actual, authorized = from_bank(opened,role,tile,spec,factor,producer)
                targets = sparse_targets(actual,authorized)
                timings["materialize_seconds"].setdefault(factor,[]).append(perf_counter()-tick)
                observed, graphs = {}, {}
                for arm in (arms if tile in representatives else ("off",)):
                    timed = {}
                    out,graphs[arm] = forward(models[arm],actual,producer,role,timed=timed)
                    observed[arm] = head_hashes(out,actual)
                    timings["forward_seconds"].setdefault(factor,[]).append(timed["forward_seconds"])
                    timings["graph_seconds"].setdefault(factor,[]).append(sum(timed["graph_seconds"]))
                if len({graph["q"] for graph in graphs.values()})!=1:
                    raise ValueError("Real multiscale arms did not share the same initial assignment")
                if factor==1 and (tile,band) in older:
                    old = older[(tile,band)]
                    comparable = set(observed)&set(old["probabilities"])
                    if any(observed[arm]!=old["probabilities"][arm] for arm in comparable):
                        raise ValueError("Native factor 1 changed historical probabilities")
                    if any(graphs["off"][k]!=v for k,v in old["graph"]["off"]["built"][0].items()):
                        raise ValueError("Native factor 1 changed historical off graph")
                    legacy += 3*len(comparable)
                    old_rows += 1
                if "metadata" in observed and (observed["metadata"]!=observed["mean"]
                                               or observed["joint"]!=observed["content"]):
                    raise ValueError("Unknown real GSD altered metadata fallback")
                rows.append(dict(observation=actual.receipt,band=band,tile=tile,role=role,heads=observed,
                                 graphs=graphs,sparse=targets.summary()))
    opened.verify_authority(historical["bank"])
    if state!=state_hash(original) or inventory(original,models)!=initial or custody.protect(intake)<1:
        raise ValueError("Real native scan changed a source, model or earlier artifact")
    timings = {kind:{factor:dict(calls=len(x),median=float(np.median(x)),sum=sum(x))
                     for factor,x in by_factor.items()} for kind,by_factor in timings.items()}
    return dict(scene=scene,producer=producer,prior=prior,bank=opened.pin,source_state=state,models=initial,
                representatives=sorted(representatives),blocks=len(opened.rows),rows=rows,
                elapsed_seconds=perf_counter()-start, timings=timings, historical_rows=old_rows,
                historical_heads=legacy,real_updates=0,raw_gt=0,query=0,new_labels=0)
