"""Verify role-bound native pathways, historical readouts and device controls in WP12."""

from collections import defaultdict
from statistics import fmean
import subprocess

from evaluation.frozen_training_stages import HEADS, file_pin, json_pin, require
from evaluation.wp12_block_readouts import block_metrics
from evaluation.wp12_linear_attribution import FOCUS
from evaluation.wp12_native_pathways import COMPONENTS, GROUPS, SITES
from evaluation.wp12_native_pathways_real import CASES, SCHEMA
from scripts.run_frozen_band_region import _write

REVIEW = "wp12_native_pathway_review_v1"
BRANCH_CASES = ("xz-initial","xz-selected","hh-short-shared","hh-long-scratch")


def authenticate(root,location,code):
    link = file_pin(location)
    value = json_pin(root,link)
    subprocess.run(["git","merge-base","--is-ancestor",value["producer"],code["commit"]],check=True)
    return link,value


def real_case(root,place,case,code):
    link,marker = authenticate(root,place/case/"complete.json",code)
    plan = json_pin(root,marker["plan"])
    require(plan["schema"]==marker["schema"]==SCHEMA and plan["case"]==case
            and marker["producer"]==plan["producer"] and set(marker["cases"])==set(plan["roster"])
            and marker["optimizer_updates"]==0 and marker["query_accessed"] is marker["raw_gt_loaded"] is False,
            "Incomplete or unauthorized native role report")
    if case.startswith("hh-long"):
        require(plan["basis"]["archived_path_cursor"]==60 and plan["basis"]["approved_combined_updates"]==120,
                "Historical two-path optimizer budget differs")
    blocks = {}
    for key,receipt in marker["cases"].items():
        block = json_pin(root,receipt)
        require(block["plan"]==marker["plan"] and block["key"]==key and block["model"]==plan["model"]
                and block["role"] in ("support","calibration_development"),"Foreign durable block")
        measured = block["measurement"]
        expected = set(FOCUS[plan["scene"]])|set(plan["controls"])
        points = measured["rows"]
        counts = block["class_counts"]
        require(measured["probability_parity"]<=3e-6 and measured["rebuilt_scope"] is None
                and set(measured["feature_receipts"])==set(SITES) and
                [r["ordinal"] for r in points]==list(range(1,len(points)+1))
                and len(points)==sum(counts[c-1] for c in expected),"Native endpoint or released point count differs")
        for point in points:
            require(point["y"] in expected and point["fused_gradient_reconstruction"]<3e-6
                    and set(point["groups"])==set(GROUPS),"Foreign fixed-head or pixel-path grouping")
            for name,summary in point["groups"].items():
                values = summary["heads"]
                if summary["n"]:
                    require(set(values)==set(COMPONENTS) and all(v["l1"]>=0 and
                            0<=v["zero_sources"]<=summary["n"] for v in values.values()),
                            "Malformed native input sensitivity")
                    if name=="far_or_unreachable":
                        require(values["graph"]["l1"]==values["fused_graph"]["l1"]==0,
                                "Disconnected/two-stage unreachable source conveyed graph gradients")
                else:
                    require(values is None,"Unavailable spatial sources were filled")
        blocks[key] = block
    return link,plan,blocks


def role_summary(plan,blocks):
    roles = defaultdict(list)
    for datum in blocks.values():
        roles[datum["role"],datum["view"]].append(datum)
    output = {}
    for (role,view),items in sorted(roles.items()):
        scores = {}
        for cls in sorted(set(FOCUS[plan["scene"]])|set(plan["controls"])):
            tuples = [(block["block"],point) for block in items for point in block["measurement"]["rows"]
                       if point["y"]==cls]
            labels = [row["y"] for _,row in tuples]
            if not tuples:
                scores[str(cls)] = dict(n=0,heads=None)
                continue
            heads = {head:dict(mean_fixed_margin=fmean(row["heads"][head]["margin"] for _,row in tuples),
                blocks=block_metrics(labels,[row["heads"][head]["predicted"] for _,row in tuples],
                                     [tile for tile,_ in tuples],len(items[0]["class_counts"]))) for head in HEADS}
            sources = {}
            for name in GROUPS:
                valid = [row["groups"][name] for _,row in tuples if row["groups"][name]["n"]]
                sums = {head:dict(mean_l1=fmean(group["heads"][head]["l1"] for group in valid),
                    exactly_zero_directions=sum(group["heads"][head]["signed_direction"]==0 for group in valid))
                        for head in COMPONENTS} if valid else None
                sources[name] = dict(available_target_points=len(valid),sources=sum(x["n"] for x in valid),heads=sums)
            scores[str(cls)] = dict(n=len(tuples),heads=heads,sources=sources)
        output[f"{role}/{view}"] = dict(class_profiles=scores,blocks=len(items))
    return dict(views=output,blocks=len(blocks),point_visits=sum(len(x["measurement"]["rows"]) for x in blocks.values()),
        summed_measurement_seconds=sum(x["measurement"]["seconds"] for x in blocks.values()),
        summed_materialize_seconds=sum(x["materialize_seconds"] for x in blocks.values()),
        summed_native_capture_seconds=sum(x["measurement"]["native_capture_seconds"] for x in blocks.values()),
        basis=plan["basis"],model=plan["model"],independent_accuracy=False)


def projected(root,place,case,plan,blocks,code):
    link,body = authenticate(root,place/(case+".json"),code)
    require(body["schema"]=="wp12_native_fused_branch_projection_v1" and body["case"]==case
            and body["model"]==plan["model"] and body["basis"]==plan["basis"]
            and body["optimizer_updates"]==0 and not body["raw_gt_loaded"] and not body["query_accessed"],
            "Separate projected margins crossed frozen endpoint identity")
    rosters = defaultdict(dict)
    for key,item in body["measurement"].items():
        focus = [p for p in blocks[key]["measurement"]["rows"] if p["y"] in FOCUS[plan["scene"]]]
        rows = item["rows"]
        require(len(rows)==len(focus) and all(r["truth"]==q["y"] and
            r["predictions"]=={h:q["heads"][h]["predicted"] for h in HEADS} for r,q in zip(rows,focus,strict=True)),
            "Projection diagnostic changed frozen point order or predicted classes")
        for r in rows:
            require(abs(r["bias"]+r["cnn_dot"]+r["graph_dot"]-r["margin"])<2.5e-5,
                    "Fused native branch reconstruction differs")
        rosters[item["role"],item["band"]][item["block"]] = rows
    summary = {}
    for (role,band),records in rosters.items():
        result = {}
        for cls in FOCUS[plan["scene"]]:
            points = [p for rows in records.values() for p in rows if p["truth"]==cls]
            require(points and len({p["rival"] for p in points})==1,"Projection opponent changed")
            result[str(cls)] = dict(n=len(points),rival=points[0]["rival"],
                      mean={name:fmean(p[name] for p in points) for name in ("margin","bias","cnn_dot","graph_dot")})
        summary[f"{role}/{band}"] = result
    contrasts = {}
    if case.startswith("xz-"):
        full = rosters["val","full"]
        for view in ("central_gap","matched_dispersed"):
            alt = rosters["val",view]
            require(full.keys()==alt.keys(),"Retained comparison lost a geometry block")
            by_class = defaultdict(list)
            for tile,left in full.items():
                for a,b in zip(left,alt[tile],strict=True):
                    require(all(a[k]==b[k] for k in ("ordinal","truth","rival","bias")),
                            "Within-model view change altered the fixed rival or bias")
                    by_class[a["truth"]].append((a,b))
            contrasts[view] = {str(cls):dict(n=len(rows),mean_shift={name:fmean(b[name]-a[name] for a,b in rows)
                           for name in ("margin","bias","cnn_dot","graph_dot")}) for cls,rows in by_class.items()}
    return dict(pin=link,same_role_view_deltas=contrasts,role_view_margins=summary,seconds=body["seconds"])


def devices(cpu,cuda):
    require(cpu["schema"]==cuda["schema"]=="wp12_native_pathway_controls_v1"
            and cpu["predeclared"]==cuda["predeclared"] and cpu["numeric"]["physical_uuid"] is None
            and cuda["numeric"]["physical_uuid"] and cpu["optimizer_updates"]==cuda["optimizer_updates"]==0,
            "Foreign device or directed fixture")
    error = {k:0. for k in ("derivative","finite","margins")}
    reads = 0
    for seed,control in cpu["seeds"].items():
        for arm,records in control.items():
            for a,b in zip(records,cuda["seeds"][seed][arm],strict=True):
                for left,right in (zip(a,b,strict=True) if arm=="support_width" else [(a,b)]):
                    for kind in error:
                        for name in left[kind]:
                            reads += 1
                            error[kind] = max(error[kind],abs(left[kind][name]-right[kind][name]))
    require(error["derivative"]<3e-6 and error["margins"]<3e-6 and error["finite"]<.0002,
            "Physical GPU failed controlled native arithmetic tolerance")
    return dict(gpu=cuda["numeric"]["physical_uuid"],scalar_comparisons=reads,maximum_absolute_difference=error,
                cpu_seconds=cpu["seconds"],gpu_seconds=cuda["seconds"])


def real_device(root,gpu_path,base):
    plan,blocks = base
    gpu_plan = json_pin(root,file_pin(gpu_path/"plan.json"))
    other = [p for p in gpu_path.glob("train-*.json")]
    require(len(other)==2 and all(plan[x]==gpu_plan[x] for x in ("case","scene","model","basis","roster")),
            "GPU sampled a different real checkpoint or role")
    maximum = 0.
    values = 0
    for file in other:
        cuda = json_pin(root,file_pin(file))
        cpu = blocks[cuda["key"]]
        for a,b in zip(cpu["measurement"]["rows"],cuda["measurement"]["rows"],strict=True):
            require(a["y"]==b["y"] and a["rival"]==b["rival"] and a["ordinal"]==b["ordinal"],
                    "GPU changed bound development points")
            for h in HEADS:
                require(a["heads"][h]["predicted"]==b["heads"][h]["predicted"],"Real native CPU/GPU changed decision")
                values += 1
                maximum = max(maximum,abs(a["heads"][h]["margin"]-b["heads"][h]["margin"]))
    return dict(physical_uuid=gpu_plan["device"]["physical_uuid"],released_blocks=len(other),
                matched_three_head_points=values,maximum_margin_difference=maximum)


def review(root,out,code,*,real_root,branch_root,cpu_path,gpu_path,gpu_real,external_path,readout_path):
    source = {}
    real = {}
    stages = {}
    for case in CASES:
        link,plan,blocks = real_case(root,real_root,case,code)
        source[case] = link
        stages[case] = plan,blocks
        real[case] = role_summary(plan,blocks)
    branches = {case:projected(root,branch_root,case,*stages[case],code) for case in BRANCH_CASES}
    cp,cpu = authenticate(root,cpu_path,code)
    gp,gpu = authenticate(root,gpu_path,code)
    ep,external = authenticate(root,external_path,code)
    rp,readouts = authenticate(root,readout_path,code)
    require(external["schema"]=="wp12_native_three_roles_test_v1" and external["cost_annotations_hidden"] is True
            and set(external["spatial"])=={"boundary","observed_interior","narrow"}
            and len(readouts["selected_readouts"])==12 and readouts["new_probe_updates"]==external["optimizer_updates"]==0,
            "Released roles, spatial proof or prior signed readouts are missing")
    require(all(row["n"]>0 and not row["qualification"]["fused"]["protection_evaluable"]
                for row in external["spatial"].values()),"Insufficient trusted geometry was promoted")
    out.parent.mkdir(parents=True,exist_ok=True)
    return _write(out,dict(schema=REVIEW,producer=code["commit"],native=real,receipts=source,projected_branches=branches,
            controls=dict(cpu=cp,gpu=gp,verification=devices(cpu,gpu),
                          real_sample=real_device(root,gpu_real,stages["xz-selected"])),
            equal_capacity=dict(pin=rp,readouts=readouts["selected_readouts"],
                                fixed_vs_own_view=readouts["fixed_and_per_view"],source=readouts["accepted_sources"]),
            external=dict(pin=ep,spatial={k:dict(n=v["n"],qualification=v["qualification"])
                                     for k,v in external["spatial"].items()},axes=["support","calibration","evaluation"]),
            authorized_real_interventions=0,new_real_optimizer_updates=0,raw_gt_loaded=False,query_accessed=False,
            wp04_executed=False,independent_real_qualification=None))
