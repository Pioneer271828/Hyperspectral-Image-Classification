"""Preregistered directed switches and two-layer geometry for synthetic mechanisms."""

from collections import Counter

import numpy as np

from evaluation.frozen_training_stages import require
from model.research_config import digest
from transfer.controlled_regional_support import invariants, match_support

VERSION = "released_causal_region_witness_v1"
SIZE = 32
POLICIES = ("causal_original","causal_matched","causal_placebo","causal_evolving")
REGIMES = ("new_path","existing_path","local","deleted","sham")
FOCUS = {"xuzhou":(5,6),"whu-honghu":(15,21)}
TARGETS = (0,1)
DECOYS = (2,3)
SOURCES = (4,5)
UNRELATED = (6,7)
HUB = 8
POSITIONS = ((16,4),(16,9),(9,16),(23,16),(3,16),(29,16),(16,28),(16,23),(16,16))
SELF = {(i,i) for i in range(len(POSITIONS))}
BASE = SELF | {(0,2),(1,3),(6,4),(7,5)} | {(HUB,i) for i in range(HUB)}
WITNESSES = {
    "matched": (((0,2),(6,4),(1,3),(7,5)),((0,4),(6,2),(1,5),(7,3))),
    "placebo": (((0,2),(1,3)),((0,3),(1,2))),
}


def lattice(layout):
    require(layout in ("a","b","h"),"Registered disjoint synthetic layout required")
    if layout=="b":
        return tuple((x,SIZE-1-y) for y,x in POSITIONS)
    return POSITIONS


def memberships(layout):
    membership = np.full((SIZE,SIZE),HUB,dtype=np.int64)
    for i,(y,x) in enumerate(lattice(layout)[:-1]):
        membership[y-2:y+3,x-2:x+3] = i
    return membership


def components(pairs, *, undirected):
    neighbors = [set() for _ in POSITIONS]
    for receiver,sender in pairs:
        neighbors[sender].add(receiver)
        if undirected:
            neighbors[receiver].add(sender)
    reached = []
    for i in range(len(POSITIONS)):
        known = {i}
        remaining = [i]
        while remaining:
            new = neighbors[remaining.pop()]-known
            remaining.extend(new)
            known.update(new)
        reached.append(known)
    return sorted({tuple(sorted(j for j in destinations if i in reached[j])) for i,destinations in enumerate(reached)})


def _switch(base, witness, points):
    remove,add = map(set,witness)
    require(remove<=base and not add&base and len(remove)==len(add) and len(remove)>0
            and not remove&SELF,"Invalid frozen directed switch witness")
    chosen = base-remove|add
    require(invariants(chosen,points)==invariants(base,points)
            and all(components(chosen,undirected=undirected)==components(base,undirected=undirected)
                    for undirected in (False,True)),
            "Switch changed degrees, distances or component partition")
    return chosen


def support(layout, regime):
    points = lattice(layout)
    alt = {k:_switch(BASE,v,points) for k,v in WITNESSES.items()}
    first = alt["matched"] if regime=="existing_path" else BASE
    return {k:sorted(v) for k,v in dict(first=first,original=BASE,**alt).items()}


def reachable(first,second,target,source):
    intermediate = {sender for receiver,sender in second if receiver==target}
    return any(receiver in intermediate and sender==source for receiver,sender in first)


def emitters(regime):
    if regime=="local":
        return TARGETS
    if regime in ("new_path","existing_path"):
        return SOURCES
    return ()


def certify(layout):
    points = lattice(layout)
    paths={}
    for regime in REGIMES:
        graph=support(layout,regime)
        paths[regime]={phase:[reachable(graph["first"],edges,t,s)
                             for t,s in zip(TARGETS,emitters(regime))]
                       for phase,edges in graph.items() if phase!="first"}
    require(paths["new_path"]=={"original":[False,False],"matched":[True,True],"placebo":[False,False]}
            and all(paths[k][phase]==[True,True] for k in ("existing_path","local")
                    for phase in ("original","matched","placebo")),
            "Witness does not distinguish the intended two-layer source paths")
    return dict(algorithm="fixed_directed_switch_witness",status="MATCHED",points=[list(p) for p in points],
                witness={k:[[list(p) for p in side] for side in w] for k,w in WITNESSES.items()},
                counts=invariants(BASE,points),strong=[list(x) for x in components(BASE,undirected=False)],
                weak=[list(x) for x in components(BASE,undirected=True)],paths=paths)


def search_controls(layout):
    points=lattice(layout)
    original=sorted(BASE)
    allowed=sorted(BASE|{p for w in WITNESSES.values() for p in w[1]})
    common=(sorted(SELF|{(HUB,i) for i in range(HUB)}),points,[0]*len(points),list(range(len(points))))
    exact=match_support(original,original,*common,seed=0)
    limited=match_support(original,allowed,*common,seed=1,limit=1)
    require(exact["status"]=="INFEASIBLE" and limited["status"]=="NOT_FOUND_WITHIN_BUDGET",
            "Preflight controls failed to distinguish no solution and exhausted search")
    return dict(exhaustive=exact,limited=limited)


def contract(bank,scene,roles,layout,tiles,cue_bands):
    value=dict(schema=VERSION,bank=bank,scene=scene,roles=digest(roles),layout=layout,tiles=tiles,
               cue_bands=cue_bands,witness=certify(layout))
    return value


def verify(value, *, bank, scene, roles=None, bands=None):
    require(set(value)=={"schema","bank","scene","roles","layout","tiles","cue_bands","witness"}
            and value["schema"]==VERSION and value["bank"]==bank and value["scene"]==scene and scene in FOCUS
            and value["witness"]==certify(value["layout"]),
            "Causal graph design or authenticated switch witness changed")
    tiles=value["tiles"]
    require(tiles and all(k.isdecimal() and isinstance(v,dict) and set(v)=={"role","regime"}
            and v["role"] in ("devtrain","devval") and v["regime"] in REGIMES for k,v in tiles.items()),
            "Causal graph tile/role roster differs")
    require(len(value["cue_bands"])==12 and value["cue_bands"]==sorted(set(value["cue_bands"])),
            "The preregistered emitter columns changed")
    if bands is not None:
        from transfer.spatial_coverage_protocol import retained_positions

        kept=retained_positions(bands,"central_gap")
        omitted=[x for i,x in enumerate(bands.band_ids) if i not in kept]
        require(value["cue_bands"]==omitted,"Central view must actually delete every emitter column")
    if roles is not None:
        expected={str(t):"devtrain" if role=="support" else "devval" for role,items in roles.items() for t in items}
        require(value["roles"]==digest(roles) and {k:v["role"] for k,v in tiles.items()}==expected
                and Counter(tiles[str(t)]["regime"] for t in roles["evaluation"])=={k:2 for k in REGIMES},
                "Causal preflight must cover every registered spatial block and both held repeats")
    return value


def phase(policy, role, step, budget):
    if policy=="causal_evolving":
        return ("original" if step<=budget//3 else "placebo" if step<=2*budget//3 else "matched"
                ) if role=="devtrain" else "matched"
    return policy.removeprefix("causal_")
