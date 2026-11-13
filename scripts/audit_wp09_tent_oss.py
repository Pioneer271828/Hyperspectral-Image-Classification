"""Independently verify the original and candidate frame tensors and scope."""

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from research_runtime import hash_file, tensor_hash, write_json
from transfer.online_tta_run import committed_history, load_checkpoint, new_session, predict, restore, isolated_rng
from transfer.online_tta_stream import StreamRegistration, read_pin


HEADS = ("fused", "graph", "cnn")
TOLERANCE = dict(atol=1e-6, rtol=1e-5, equal_nan=False)


def main():
    p=ArgumentParser(description=__doc__)
    for key in ("candidate","original","out"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    root=a.candidate
    scope=json.loads((root/"scope.json").read_text())
    outcome=json.loads((root/"receipt.json").read_text())
    comparisons=[]
    for name,pin in scope["registrations"].items():
        historical=StreamRegistration(scope["references"][name])
        reg=StreamRegistration(pin)
        binding=json.loads((root/(name+"-full")/"binding.json").read_text())
        reg.adaptation_provider=binding["adaptation_provider"]
        refs,_=committed_history(historical,a.original/(name+"-full"),complete=True)
        evs,_=committed_history(reg,root/(name+"-full"),complete=True)
        resumed,_=committed_history(reg,root/(name+"-split"),complete=True)
        assert resumed==evs
        source=reg.source_model()
        for i,(before,after) in enumerate(zip(refs,evs,strict=True),1):
            old_dir=a.original/(name+"-full")/f"frame-{i:06d}"
            new_dir=root/(name+"-full")/f"frame-{i:06d}"
            old_state=load_checkpoint(old_dir/"checkpoint.pt",before["checkpoint"])
            new_state=load_checkpoint(new_dir/"checkpoint.pt",after["checkpoint"])
            assert (old_state["cursor"],old_state["frame"],before["optimizer"],before["rng"]) == (
                    new_state["cursor"],new_state["frame"],after["optimizer"],after["rng"])
            with isolated_rng(reg.settings["seed"]):
                session=new_session(reg,source)
                if not after["reset"]:
                    previous=evs[i-2]
                    restore(session,load_checkpoint(root/(name+"-full")/f"frame-{i-1:06d}"/"checkpoint.pt",previous["checkpoint"]))
                frame=reg.frames[i-1]
                observed=predict(session,reg.observation(frame["observation"]),len(reg.value["class_ids"]))
            heads={}
            with np.load(old_dir/"prediction.npz",allow_pickle=False) as old, np.load(new_dir/"prediction.npz",allow_pickle=False) as new:
                for head in HEADS:
                    left,right=old[head],new[head]
                    assert left.shape==right.shape and np.isfinite(left).all() and np.isfinite(right).all()
                    np.testing.assert_array_equal(right,observed[head])
                    assert tensor_hash(torch.from_numpy(right))==after["predictions"]["heads"][head]
                    disagree=int(np.count_nonzero(~np.isclose(right.astype(float),left.astype(float),**TOLERANCE)))
                    assert disagree==0 and np.array_equal(left.argmax(1),right.argmax(1))
                    heads[head]=dict(elements=left.size,exceeded=disagree,max_abs=float(np.max(np.abs(right-left))))
            assert before["changed"]==after["changed"]
            comparisons.append(dict(case=name,frame=i,heads=heads,changed=len(after["changed"]),counts=after["counts"]))
    actual=Counter()
    for item in comparisons:
        actual.update(item["counts"])
    assert len(comparisons)==144 and outcome["actual"]=={k:2*v for k,v in actual.items()}
    write_json(a.out,dict(status="passed",source_model_reloads=len(scope["registrations"]),comparisons=comparisons,
        independent_forwards=len(comparisons),optimizer_updates=0,statistics_updates=0,query=0,raw_gt=0))
if __name__=="__main__":
    main()
