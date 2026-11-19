"""Blind spectral/context oracles over the released synthetic observations."""

from pathlib import Path

import numpy as np

from evaluation.frozen_training_stages import require
from evaluation.native_spectral_control import (closest, column_positions, label_scores, normalize,
                                              support_references, verify_loader)
from evaluation.wp06_observation_audit import verify_fit_statistics
from transfer import development_bank as custody
from transfer.native_positive_fixture import SCIENTIFIC, CLASSES, TRAIN_VIEWS, centers
from transfer.spatial_coverage_protocol import bank_view


def observations(bank, roles, producer, target_role):
    release, axis = bank.release,bank.release.bands
    directory = Path(bank.pin["root"])
    table = bank.members
    groups = {"support":[bank.role_row("devtrain",i) for i in roles["support"]],
              target_role:[bank.role_row("devval",i) for i in roles[target_role]]}
    arrays = {row["id"]:custody._decode(directory,table[row["id"]]["member"],table[row["id"]]["content"],row,
                     len(axis.band_ids)) for rows in groups.values() for row in rows}
    low,high = verify_fit_statistics(arrays,roles["support"],release.statistics)
    observed,proof = {},{}
    for role,rows in groups.items():
        for view in TRAIN_VIEWS:
            indices = column_positions(axis.valid,view)
            low_cols,high_cols = low[indices],high[indices]
            samples,err = [],[]
            for row in rows:
                actual,labels,targets = bank_view(bank,row,producer,view)
                truth,cells,raw,got,inverse = verify_loader(row,release.shape,arrays[row["id"]],low,high,indices,
                                                           actual,labels,targets,axis)
                ys,xs = np.divmod(cells,raw.shape[1])
                require(((0 < ys)&(ys < raw.shape[0]-1)&(0 < xs)&(xs < raw.shape[1]-1)).all(),
                        "Released position left the center of its registered carrier")
                ring = np.stack([raw[ys+dy,xs+dx] for dy in (-1,0,1) for dx in (-1,0,1) if dy or dx],axis=0).mean(0)
                err.append(inverse)
                samples.append(dict(observation=actual,cells=cells,truth=truth,
                                    ids=[list(point) for point in targets.points],raw=raw[ys,xs],loaded=got[ys,xs],
                                    ring=ring,coordinates=actual.view.observation.coordinates[ys,xs]))
            observed[(role,view)] = samples
            proof[f"{role}/{view}"] = dict(inverse_error=max(err),columns=indices.tolist(),
                                            normalized_min=min(float(s["loaded"].min()) for s in samples),
                                            normalized_max=max(float(s["loaded"].max()) for s in samples))
    return observed,proof,low,high


def oracle(bank,roles,producer,target_role,profile):
    rows,proof,low,high = observations(bank,roles,producer,target_role)
    book = centers()
    outputs = {}
    for view in TRAIN_VIEWS:
        positions = column_positions(bank.release.bands.valid,view)
        known = book[:,positions]
        loader_means,_ = normalize(known,low[positions],high[positions])
        train, target = rows[("support",view)],rows[(target_role,view)]
        stack = lambda batch,name: np.concatenate([b[name] for b in batch])
        y0, y = stack(train,"truth"),stack(target,"truth")
        scores = {key+"_known":label_scores(y,closest(stack(target,key),means),CLASSES)
                  for key,means in (("raw",known),("loaded",loader_means),("ring",known))}
        for key in ("raw","loaded","ring","coordinates"):
            for method,prediction in support_references(stack(train,key),y0,stack(target,key),CLASSES).items():
                scores[f"{key}_{method}"] = label_scores(y,prediction,CLASSES)
        outputs[view] = scores
    full = outputs["full"]
    gates = {name: full[key]["aa"] >= SCIENTIFIC["oracle_positive_aa"] if project
             else full[key]["aa"] <= SCIENTIFIC["negative_upper_aa"]
             for name,key,project in (("local","loaded_known",profile["local"]),
                                       ("context","ring_known",profile["context"]))}
    return rows,dict(metrics=outputs,loader=proof,gates=gates,fit_roles=list(roles["support"]))
