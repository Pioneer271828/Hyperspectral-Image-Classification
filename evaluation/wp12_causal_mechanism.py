"""Recount structural access, functional information and class outcomes separately."""

import numpy as np

from evaluation.frozen_training_stages import HEADS, require
from transfer.wp_causal_graph_design import FOCUS, REGIMES

TOL = 1e-6


def _margin(scores, label, other):
    scores=np.asarray(scores,dtype=np.float64)
    return np.log(np.clip(scores[:,label-1],1e-12,1.))-np.log(np.clip(scores[:,other-1],1e-12,1.))


def _head(original, candidate, sources, label, other):
    first,last=(np.asarray(row,dtype=np.float64) for row in (original,candidate))
    old,new=(arr.argmax(axis=1)+1 for arr in (first,last))
    before,after=(label==arr for arr in (old,new))
    result=dict(n=len(old),baseline_correct=int(before.sum()),candidate_correct=int(after.sum()),
        newly_correct=int((~before&after).sum()),new_errors=int((before&~after).sum()),
        margin_before=_margin(first,label,other).tolist(),margin_after=_margin(last,label,other).tolist(),
        before_predictions=old.tolist(),after_predictions=new.tolist())
    variants={}
    rescues=~before&after
    for name,alt in sources.items():
        alt=np.asarray(alt,dtype=np.float64)
        delta=_margin(last,label,other)-_margin(alt,label,other)
        prediction=alt.argmax(axis=1)+1
        changed=np.max(np.abs(last-alt),axis=1)>TOL
        variants[name]=dict(changed=int(changed.sum()),margin_delta=delta.tolist(),
                            correct=int((prediction==label).sum()),
                            rescue_lost=int((rescues&(prediction!=label)).sum()),
                            max_probability_difference=float(np.max(np.abs(last-alt),initial=0.)))
    result["fixed_weight_controls"]=variants
    if variants:
        require(variants["unrelated_swap"]["changed"]==0,
                "Controlled unrelated source reached the held target probabilities")
    return result


def summarize(first, second, band, tiles, scene):
    focus=FOCUS[scene]
    blocks={}
    for tile in tiles:
        a,b=(arm[(band,tile)]["accuracy"] for arm in (first,second))
        require(a is not None and b is not None and a["order"]==b["order"] and a["truth"]==b["truth"],
                "Blind causal readout lost the fixed released-point ordering")
        path_a,path_b=(row["signal"] for row in (a,b))
        regime=path_a["case"]
        require(regime==path_b["case"] and regime in REGIMES,"Different held causal regime")
        records={}
        for label,other in (focus,focus[::-1]):
            idx=np.flatnonzero(np.asarray(a["truth"])==label)
            counts=[row["classes"][str(label)] for row in (path_a,path_b)]
            n,old,new=len(idx),counts[0]["reachable"],counts[1]["reachable"]
            require(all(row["points"]==n and row["reachable"] in (0,n) for row in counts),
                    "Controlled hard partition must identify whole-source reachability within each target patch")
            heads={}
            for head in HEADS:
                variants={k:np.asarray(v["heads"][head])[idx] for k,v in b.get("interventions",{}).items()}
                heads[head]=_head(np.asarray(a["heads"][head])[idx],np.asarray(b["heads"][head])[idx],
                                  variants,label,other)
                if variants and (not path_b["cue_present"] or new==0):
                    require(heads[head]["fixed_weight_controls"]["source_swap"]["changed"]==0,
                            "Source bypassed a deleted/unreachable controlled graph")
            records[str(label)]=dict(n=n,baseline_reachable=old,candidate_reachable=new,
                first_reachable=n if old==0 and new==n else 0,heads=heads,competitor=other)
        blocks[str(tile)]=dict(regime=regime,signals_present=path_b["cue_present"],focus=records,
                              ordered_point_roster=a["order"])
    return dict(unit="released_point_on_original_spatial_block",blocks=blocks,
        direction="receiver_sender",control="same_weight_source_swap_unrelated_swap_and_matching_reverse_switch",
        repeated_use="reused_block_point_rosters_across_seeds_views_and_interventions_are_not_independent")
