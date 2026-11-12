"""Summarize strictly checked signal results without exporting observations or coordinates."""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

if __package__:
    from . import runtime as rt
else:
    import runtime as rt


def describe(values):
    values = sorted(values)
    def quantile(p):
        point = (len(values)-1)*p
        low = math.floor(point)
        return values[low]+(values[math.ceil(point)]-values[low])*(point-low)
    return {"minimum":values[0],"q05":quantile(.05),"median":quantile(.5),"q95":quantile(.95),
            "maximum":values[-1],"mean":math.fsum(values)/len(values),"zero_count":sum(x==0 for x in values)}


def profile(rows):
    output = {"entries":len(rows)}
    fields = ("gradient","absolute_gradient","finite_difference","absolute_difference",
              "plus_probability_delta","minus_probability_delta","distance_to_center_over_diagonal",
              "distance_to_other_target_over_diagonal","mean_to_both_targets_over_diagonal")
    for index,field in enumerate(fields):
        output.update({field+"_"+k:v for k,v in describe([x[index] for x in rows]).items()})
    return output


def write_csv(path, rows):
    with path.open("x",encoding="utf-8",newline="") as file:
        writer = csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rt.pin(path)


def table(groups, fields):
    return [{**dict(zip(fields,key,strict=True)),**profile(values)} for key,values in sorted(groups.items())]


def summarize(scope_pin, out):
    scope = rt.read(scope_pin)
    root = rt.checked(scope_pin).parent
    prod_pin = rt.pin(root/"production"/"receipt.json")
    audit_pin = rt.pin(root/"reference"/"receipt.json")
    prod, audit = rt.read(prod_pin), rt.read(audit_pin)
    rt.require(prod["scope"]==audit["scope"]==scope_pin and audit["production"]==prod_pin
        and prod["status"]==audit["status"]=="passed" and prod["requests"]==scope["requests"]==audit["checks"]["requests"]
        and prod["state_before_equals_after"] and audit["state_before_equals_after"],"Incomplete strict comparison")
    full,transitions,short = defaultdict(list),defaultdict(list),defaultdict(list)
    graph_rates = defaultdict(Counter)
    geometry = []
    gradient_epsilon_differences = 0
    for block,tile_pin,proof_pin in zip(scope["rows"],prod["tiles"],audit["tiles"],strict=True):
        tile,proof = rt.read(tile_pin),rt.read(proof_pin)
        rt.require(tile["tile_id"]==proof["tile_id"]==block["tile_id"] and proof["production"]==tile_pin
            and tile["role"]==proof["role"]==block["role"] and
            [row["request"] for row in tile["rows"]]==block["requests"],"Detached or incomplete tile evidence")
        role = block["role"]
        anchors = {a["id"]:a for a in block["anchors"]}
        y0,y1,x0,x1 = block["box"]
        geometry.append({"tile_id":block["tile_id"],"role":role,"height":y1-y0,"width":x1-x0,
                         "corner_to_center_over_diagonal":block["anchors"][1]["distance_to_center"],
                         "each_source_to_other_target_over_diagonal":block["anchors"][1]["distance_to_other_target"],
                         "each_source_mean_to_both_targets_over_diagonal":block["anchors"][1]["mean_to_both_targets"]})
        previous = {}
        for packet in tile["rows"]:
            output_class,epsilon = packet["request"]["class_id"],packet["request"]["epsilon"]
            result = packet["result"]
            reference = result["baseline_support"]
            for row in result["rows"]:
                global_id,band = row["source"]
                identity = output_class,global_id,band
                if epsilon==rt.EPSILONS[0]:
                    previous[identity] = row["gradient"]
                else:
                    gradient_epsilon_differences+=sum(a!=b for a,b in zip(previous[identity],row["gradient"],strict=True))
                anchor = anchors[global_id]
                source = anchor["name"]
                flags = []
                for sign,support in enumerate(row["supports"]):
                    state = (row["q_changed"][sign],support["a0_hash"]!=reference["a0_hash"],
                             row["graph_values_changed"][sign],row["topology_changed"][sign])
                    flags.append(state)
                    if output_class==1:
                        rate = graph_rates[(role,epsilon,source,band,("plus","minus")[sign])]
                        rate.update({"perturbations":1,**dict(zip(("q_changed","initial_graph_values_changed",
                            "routed_graph_values_changed","routed_topology_changed"),map(int,state),strict=True))})
                any_flags = tuple(int(a or b) for a,b in zip(*flags,strict=True))
                for i,head in enumerate(rt.HEADS):
                    grad = row["gradient"][i]
                    data = (grad,abs(grad),row["finite_difference"][i],row["absolute_difference"][i],
                            row["score_deltas"][0][i],row["score_deltas"][1][i],anchor["distance_to_center"],
                            anchor["distance_to_other_target"],anchor["mean_to_both_targets"])
                    key = (role,head,output_class,epsilon,source,band)
                    full[key].append(data)
                    transitions[(*key,*any_flags)].append(data)
                    short[(role,head,epsilon)].append(data)
    rt.require(len(geometry)==78 and len(full)==648 and sum(map(len,full.values()))==audit["checks"]["gradient_entries"]
        and sum(v["perturbations"] for v in graph_rates.values())==1872,"Incorrect complete summary dimensions")
    rate_fields = ("role","epsilon","source_location","band_index","direction")
    rate_rows = []
    totals = Counter()
    for key,counts in sorted(graph_rates.items()):
        totals.update(counts)
        rate_rows.append({**dict(zip(rate_fields,key,strict=True)),**counts,
            **{k+"_rate":v/counts["perturbations"] for k,v in counts.items() if k!="perturbations"}})
    out = Path(out).resolve()
    rt.require(out.parent==root and out.name=="summary","Output must be a fresh summary sibling")
    out.mkdir(exist_ok=False)
    labels = ("role","head","output_class","epsilon","source_location","band_index")
    compact = table(short,("role","head","epsilon"))
    outputs = [
        write_csv(out/"strata.csv",table(full,labels)),
        write_csv(out/"strata-by-transition.csv",table(transitions,(*labels,"q_changed_either_direction",
            "initial_graph_changed_either_direction","routed_graph_changed_either_direction","topology_changed_either_direction"))),
        write_csv(out/"perturbation-rates.csv",rate_rows),
        write_csv(out/"geometry-distance.csv",geometry),
        write_csv(out/"role-head-epsilon.csv",compact),
    ]
    return rt.write(out/"receipt.json",{"status":"passed","scope":scope_pin,"production":prod_pin,"independent_reference":audit_pin,
        "checks":audit["checks"],"outputs":outputs,"gradient_entries":audit["checks"]["gradient_entries"],
        "gradient_entries_different_between_epsilons":gradient_epsilon_differences,
        "full_strata":len(full),"transition_strata":len(transitions),
        "unique_perturbation_rates":dict(totals),"compact":compact,"source_index_note":"physical band indices are zero-based",
        "coordinates_exported":False,"labels_exported":False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope",nargs=2,required=True,metavar=("PATH","SHA256"))
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(rt.cli_pin(args.scope),args.out)),flush=True)
