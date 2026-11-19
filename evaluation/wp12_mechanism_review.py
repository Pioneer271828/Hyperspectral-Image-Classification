"""Independent WP12 reduction for deidentified development and synthetic mechanism receipts."""

from collections import Counter, defaultdict
from itertools import combinations, product
import csv
import math

import numpy as np

from evaluation import wp12_mechanism_factorial as factorial, wp12_mechanism_real as real
from evaluation.classification_metrics import classification_metrics
from evaluation.frozen_training_stages import HEADS, file_pin, json_pin, pinned, probabilities, require
from evaluation.wp12_locked_support import CSV as OLD_POINTS, historical_points, transitions
from model.research_config import digest
from transfer import native_mechanism_campaign as edges, native_mechanism_edges as edge_shape

SCHEMA = "wp05_wp06_wp12_mechanism_review_v1"


def scope(row, schema, mode, *, updates):
    require(row["schema"]==schema and row["scope"]==mode and row["raw_gt_loaded"] is row["query_accessed"] is False
            and row.get("real_updates",0)==0 and row.get("optimizer_updates",updates)==updates
            and row.get("independent_real_accuracy") is None and row.get("independent_accuracy") is None,
            "Independent accuracy requires its approved publication, not a development or synthetic receipt")


def fit_counts(truth, a, b, classes, *, full=None):
    result=transitions(truth,a,b,classes)
    if full is not None:
        for label,row in result.items():
            y=int(label)
            row["full_correct"]=sum(t==y==p for t,p in zip(truth,full,strict=True))
            row["already_wrong_in_full"]=row["n"]-row["full_correct"]
            row["new_errors_from_full"]=sum(t==y==p and q!=y for t,p,q in zip(truth,full,a,strict=True))
            row["rescued_full_to_target_losses"]=sum(t==y==p==r and q!=y
                                        for t,p,q,r in zip(truth,full,a,b,strict=True))
    return result


def real_reduction(root, folders, gpu_dir, out):
    snapshots,digests,gpu_summary={},{},{}
    for directory in folders:
        for path in sorted((directory/"cpu").glob("*.json")):
            signed=file_pin(path)
            row=json_pin(root,signed)
            scope(row,real.SCHEMA,"released_development_offline_only",updates=0)
            require(row["case"] not in snapshots and row["full_donor_available_to_training_or_deployment"] is False
                    and row["device"]=="cpu" and row["roles"]==["devval"]
                    and row["registration"]==file_pin(directory/"registration.json")["sha256"],
                    "Old donor or unapproved training/deployment role entered the diagnostic")
            snapshots[row["case"]]=row
            digests[row["case"]]=signed["sha256"]
    require(set(snapshots)==set(real.CASES),"Release lacks frozen development cases")
    original=historical_points(root)
    archive=pinned(root,dict(path=str(root/OLD_POINTS[0]),sha256=OLD_POINTS[1]))
    archived_margins=defaultdict(list)
    with archive.open(newline="",encoding="utf-8") as old:
        for row in csv.DictReader(old):
            if row["margin"] and row["readout"].endswith("_ridge"):
                archived_margins[row["case"],row["view"],row["readout"],row["truth"]].append(
                    (int(row["fixed_rival"]),float(row["margin"])))
    results={}
    fields=("scene","case","step","point","truth","target","donor","site","control","head",
            "predicted","highest_other","fixed_rival","margin","highest_margin")
    with (out/"development-points.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for name,case in snapshots.items():
            flat=defaultdict(lambda: defaultdict(list))
            true=[]
            blocked=Counter()
            for group in case["groups"]:
                truth=group["truth"]
                require(group["offset"]==len(true),"Point ordinals are not consecutive")
                true.extend(truth)
                choices={f"{band}<-{band}/original/base":rows for band,rows in group["base"].items()}
                for key,row in group["variants"].items():
                    if row["status"]=="incompatible_region_assignment":
                        require(key.endswith("/pooled") and "scores" not in row, "Incompatible region semantics were silently exchanged")
                        blocked[key]+=1
                        continue
                    require(row["status"]=="paired", "Unknown one-site exchange")
                    choices[key]=row["scores"]
                for band,sample in group["edge"].items():
                    for arm,row in sample["scores"].items():
                        choices[f"{band}<-{band}/fixed_edges/{arm}"]=row
                for key,heads in choices.items():
                    names,site,trial=key.split("/")
                    target,donor=names.split("<-")
                    require(set(heads)==set(HEADS),"Frozen change lacks matched original heads")
                    for head,rows in heads.items():
                        before=group["base"][target][head]
                        original_full=group["base"]["full"][head]
                        if target==donor=="full" and trial=="aligned":
                            require(rows==before,"Identity replay changed predictions or margins")
                        require(len(rows)==len(truth),"Frozen donor changed label denominator")
                        values=flat[key][head]
                        values.extend((label,baseline["predicted"],altered["predicted"],full["predicted"],altered["margin"])
                                      for label,baseline,altered,full in zip(truth,before,rows,original_full,strict=True))
                        for j,(label,row) in enumerate(zip(truth,rows,strict=True),group["offset"]+1):
                            writer.writerow(dict(scene=case["scene"],case=name,step=case["step"],point=j,truth=label,
                                           target=target,donor=donor,site=site,control=trial,head=head,**row))
            metrics={}
            for key,heads in flat.items():
                metrics[key]={}
                for head,packed in heads.items():
                    truth,a,b,full,margins=zip(*packed,strict=True)
                    cells=fit_counts(truth,a,b,case["classes"],full=full)
                    for c,r in cells.items():
                        eligible=[m for y,m in zip(truth,margins,strict=True) if y==int(c) and m is not None]
                        r["mean_fixed_margin"]=float(np.mean(eligible)) if eligible else None
                    metrics[key][head]=dict(points=len(truth),all_classes=cells)
            baseline={}
            for band in real.VIEWS:
                before,after=(flat[f"{slot}<-{slot}/original/base"] for slot in ("full",band))
                baseline[band]={}
                for head in HEADS:
                    a,b=([row[1] for row in rows[head]] for rows in (before,after))
                    if (name,band,"native_"+head) in original:
                        require([(y,p) for _,y,p in original[name,band,"native_"+head]]==list(zip(true,b,strict=True)),
                                "Archived full-class published decisions are inconsistent")
                    baseline[band][head]=fit_counts(true,b,b,case["classes"],full=a)
                fused,graph,cnn=([row[1] for row in after[head]] for head in HEADS)
                baseline[band]["head_disagreement"]={
                    str(label):dict(total=sum(y==label for y in true),different=sum(y==label and len({a,b,c})>1
                                              for y,a,b,c in zip(true,fused,graph,cnn,strict=True)))
                    for label in range(1,case["classes"]+1)}
            stages={"vca":"vca","pre_pool":"pre_pool","pooled":"pooled","cnn":"cnn","graph":"graph","fused":"fused"}
            historical={}
            for site in stages:
                if (name,"full",site+"_ridge") not in original:
                    continue
                historical[site]={}
                for band in real.VIEWS:
                    rows=original[name,band,site+"_ridge"]
                    y=[label for _,label,_ in rows]
                    old=[p for _,_,p in original[name,"full",site+"_ridge"]]
                    history=[p for _,_,p in rows]
                    require(y==true,"Different historical support readout denominator")
                    report=transitions(y,old,history,case["classes"])
                    for label,value in report.items():
                        scores=archived_margins[name,band,site+"_ridge",label]
                        value["historical_fixed_rival"]=scores[0][0] if scores else None
                        value["historical_mean_margin"]=float(np.mean([row[1] for row in scores])) if scores else None
                    historical[site][band]=report
            results[name]=dict(scene=case["scene"],step=case["step"],total=len(true),major_rivals=case["locked_rivals"],
                focus=case["focus"],authority=case["authority"],checkpoint=case["checkpoint"],source=digests[name],
                baseline=baseline,interventions=metrics,pooled_incompatible=dict(blocked),
                historical_view_refit_ridge_only=historical,
                edges=[dict(block=b["block"],band=band,shape=r["shape"],removed=r["removed"],
                            match_status=r["match_status"],distance_mode=r["mode"])
                       for b in case["groups"] for band,r in b["edge"].items()])
    if gpu_dir is not None:
        for path in sorted(gpu_dir.glob("*.json")):
            row=json_pin(root,file_pin(path))
            scope(row,real.SCHEMA,"released_development_offline_only",updates=0)
            cpu=snapshots[row["case"]]
            require(row["state"]==cpu["state"] and row["checkpoint"]==cpu["checkpoint"],"CPU/GPU development states differ")
            n,changed,largest=0,0,0.
            for a,b in zip(cpu["groups"],row["groups"],strict=True):
                require(a["truth"]==b["truth"],"CUDA development labels differ")
                available_a={key:item["scores"] for key,item in a["variants"].items() if item["status"]=="paired"}
                available_b={key:item["scores"] for key,item in b["variants"].items() if item["status"]=="paired"}
                require(available_a.keys()==available_b.keys(),"CUDA feasible exchange sites differ")
                for u,v in (*zip(a["base"].values(),b["base"].values(),strict=True),
                            *((available_a[k],available_b[k]) for k in available_a)):
                    for head in HEADS:
                        for first,second in zip(u[head],v[head],strict=True):
                            n+=1
                            changed+=first["predicted"]!=second["predicted"]
                            largest=max(largest,abs(first["highest_margin"]-second["highest_margin"]))
            gpu_summary[row["case"]]=dict(compared=n,changed=changed,max_margin_difference=largest,
                                           admission=row["admission"],cuda_edge_status=[g["edge"]!= {} for g in row["groups"]].count(True))
    return dict(cases=results,device=gpu_summary,historical_csv=OLD_POINTS[1],
                csv_sha256=file_pin(out/"development-points.csv")["sha256"],
                causal_scope="single-intermediate-sufficiency_only_not_a_prediction_model")


def predictions(row, key="head"):
    truth=row["truth"]
    return {h:(probabilities(row[key][h],len(truth),9).argmax(1)+1).tolist() for h in HEADS}


def classification(truth,p):
    metric=classification_metrics(truth,p,list(range(1,10)))
    return dict(aa=metric["aa"],oa=metric["oa"],confusion=metric["confusion_matrix"],per_class=metric["per_class"])


def factorial_contrast(cells, outcome):
    # Positive effects follow second-minus-first, with difference-of-differences for interactions.
    axes=("placement","metadata","grid")
    results={}
    for size in (1,2,3):
        for focus in combinations(range(3),size):
            retained=tuple(i for i in range(3) if i not in focus)
            delta=0.
            for point in product((0,1),repeat=3):
                sign=math.prod(1 if point[i] else -1 for i in focus)
                delta+=sign*outcome(cells[point])
            results["*".join(axes[i] for i in focus)]=delta/(2**len(retained))
    return results


def factorial_reduction(root, directory):
    output={}
    manifest=json_pin(root,file_pin(directory/"factorial-registration.json"))
    require(digest(manifest)==digest(factorial.registration(root)),"Frozen factorial registration differs")
    histories={}
    for path in sorted(directory.glob("cpu-s*.json")):
        receipt=json_pin(root,file_pin(path))
        scope(receipt,factorial.SCHEMA,"independent_synthetic_confirmation",updates=0)
        require(receipt["registration"]==digest(manifest) and receipt["manifest"]==digest(manifest)
                and receipt["approximate_real_physical_axis_promoted"] is False,
                "A factorial row modifies the physical metadata policy")
        histories[receipt["case"]]=receipt
        groups={}
        for item in receipt["rows"]:
            by_key={}
            for cell in item["cells"]:
                r=cell["result"]
                k=(factorial.VIEWS.index(r["band"]),factorial.METADATA.index(r["metadata"]),
                    factorial.FACTORS.index(r["factor"]))
                require(k not in by_key and len(r["bands"])==13,"Factorial fixed retained column cardinality differs")
                by_key[k]=r
            require(len(by_key)==8 and len({r["point_set"] for r in by_key.values()})==1
                    and all(r["truth"]==item["old_joint_stress"]["truth"] for r in by_key.values()),
                    "Factorial role or original-pixel denominator differs")
            roc={}
            metrics={}
            base=by_key[0,0,0]
            base_dec=predictions(base)
            for loc in product((0,1),repeat=3):
                r=by_key[loc]
                if loc[1]==1:
                    factorial.verify_input_pair(by_key[loc[0],0,loc[2]],r)
                pred=predictions(r)
                metrics[loc]={head:classification(r["truth"],values) for head,values in pred.items()}
                roc["".join(map(str,loc))]=dict(view=[r["band"],r["metadata"],r["factor"]],grid=r["grid"],
                              metrics=metrics[loc],against_reference={h:transitions(r["truth"],base_dec[h],v,9)
                                                                    for h,v in pred.items()})
            effects={h:dict(aa=factorial_contrast(metrics,lambda row:row[h]["aa"]),
                            rare_recall=factorial_contrast(metrics,lambda row:row[h]["per_class"][-1]["recall"]))
                     for h in HEADS}
            old=item["old_joint_stress"]
            groups[item["mode"]]=dict(cells=roc,effects=effects,
                             old_joint_stress={h:classification(old["truth"],p) for h,p in predictions(old).items()})
        output[receipt["case"]]=dict(scopes=groups,source_checkpoint=receipt["checkpoint"],
                                    sha256=file_pin(path)["sha256"])
    require(len(output)==24, "Incomplete frozen production factorial")
    parity={}
    for path in sorted(directory.glob("cuda-s*.json")):
        gpu=json_pin(root,file_pin(path))
        scope(gpu,factorial.SCHEMA,"independent_synthetic_confirmation",updates=0)
        cpu=histories[gpu["case"]]
        require(cpu["state"]==gpu["state"] and cpu["checkpoint"]==gpu["checkpoint"],"Factorial device state differs")
        changes,total,max_error=0,0,0.
        for a,b in zip(cpu["rows"],gpu["rows"],strict=True):
            require(a["mode"]==b["mode"] and a["tile"]==b["tile"],"Device factorial roster differs")
            for x,y in zip(a["cells"],b["cells"],strict=True):
                u,v=x["result"],y["result"]
                require(all(u[k]==v[k] for k in ("band","metadata","factor","spectral","point_set","grid")),
                        "CUDA factorial did not use the same observation")
                for h in HEADS:
                    z,w=map(np.asarray,(u["head"][h],v["head"][h]))
                    changes+=int((z.argmax(1)!=w.argmax(1)).sum())
                    total+=len(z)
                    max_error=max(max_error,float(np.abs(z-w).max()))
        parity[gpu["case"]]=dict(compared=total,changed=changes,max_abs_error=max_error,device=gpu["device"])
    coverage={}
    for mode in ("cpu","cuda"):
        path=directory/(mode+"-coverage.json")
        if path.exists():
            item=json_pin(root,file_pin(path))
            scope(item,factorial.SCHEMA,"coverage_checkpoint_and_fallback_engineering_only",updates=0)
            require(set(item["modes"])==set(factorial.COVERAGE) and all(v["checkpoint_restored"] for v in item["modes"].values()),
                    "Coverage mode lost checkpoint or fallback compatibility")
            coverage[mode]=item
    require("cpu" in coverage,"Missing all-mode engineering proof")
    return dict(cases=output,device=parity,coverage=coverage,
                factor_levels=dict(placement=factorial.VIEWS,metadata=factorial.METADATA,grid=factorial.FACTORS),
                fixed_band_count=13,old_joint_stress_retained_count=12,
                synthetic_nominal_is_not_verified_real=True)


def edge_reduction(root, directory, gpu_dir, program):
    output={}
    plan,link=edges.load(root,program,json_pin(root,file_pin(program/"registration.json"))["producer"])
    raw={}
    for path in sorted(directory.glob("*.json")):
        receipt=json_pin(root,file_pin(path))
        scope(receipt,edges.SCHEMA,"independent_synthetic_confirmation",updates=0)
        require(receipt["registration"]==link["sha256"] and receipt["case"] in plan["cases"]
                and set(receipt["arms"])==set(edge_shape.ARMS),"Unregistered edge treatment")
        raw[receipt["case"]]=receipt
        baseline=receipt["arms"]["fixed_none"]
        baseline_rows={(row["block"],row["band"]):row for row in baseline["rows"]}
        results={}
        for arm,trained in receipt["arms"].items():
            require(trained["budget"]==edge_shape.STEPS and all(trained[k]==baseline[k] for k in
                    ("initial","training","validation","manifest","train_device")),"Edge budgets, inputs or devices differ")
            items={}
            for key in ("rows","frozen_deletions","region_replacements"):
                rows=[]
                for row in trained[key]:
                    base=baseline_rows[row["block"],row["band"]]
                    require(all(row[k]==base[k] for k in ("truth","target_digest","spectral","ids","geometry_hash","grid"))
                            and row["boundary"]["width"]==2 and row["boundary"]["boundary_points"]==len(row["truth"])
                            and row["graph"]["q_shape"][1]==9,"Independent exact controls cross a different field or point set")
                    expected_edges=17 if row["arms"]=="fixed_none" else 13
                    require(row["graph"]["a0_nnz"]==expected_edges and row["regions"]==base["regions"]
                            and len(row["regions"]["squared_distance_multiset"])==4
                            and row["regions"]["masks_distinct_from_high"],"Matched graph removed a different count or distance population")
                    p,q=predictions(row,"heads"),predictions(base,"heads")
                    per_head={}
                    for name in HEADS:
                        result=classification(row["truth"],p[name])
                        delta=fit_counts(row["truth"],q[name],p[name],9)
                        active={k:v for k,v in delta.items() if v["n"]}
                        require(set(active)=={"1","2","3"},"Averaged scores omit a planted class")
                        passed=all((cell["losses"]==0 if int(label)==edge_shape.RARE else
                                    (cell["gains"]-cell["losses"])/cell["n"]>=edges.GATES["ordinary_recall_delta"])
                                   for label,cell in active.items())
                        aa_delta=result["aa"]-classification(row["truth"],q[name])["aa"]
                        per_head[name]=dict(metrics=result,against_fixed=delta,thresholds_pass=passed and aa_delta>=edges.GATES["aggregate_delta"])
                    values=[r["total_seconds"] for r in row["times"]]
                    require(len(values)==2 and all(math.isfinite(v) and v>0 for v in values),
                            "Matched total runtime witness is invalid")
                    rows.append(dict(block=row["block"],band=row["band"],intervention=row["arms"],
                                     geometry=row["boundary"]["orientation"],heads=per_head,
                                     region_action=[row.get(k) for k in ("operation","target","donor")],
                                     total_seconds=values))
                items[key]=rows
            results[arm]=dict(selected=trained["selected"],recovery_exact=trained["recovery"],source=trained["checkpoint"],
                               training_device=trained["train_device"],comparison=items)
        task=plan["cases"][receipt["case"]]["task"]
        adequacy={}
        for arm,trained in receipt["arms"].items():
            by_view={}
            for band in edge_shape.VIEWS:
                points=[row for row in trained["rows"] if row["band"]==band]
                truth=[label for row in points for label in row["truth"]]
                choice=[val for row in points for val in predictions(row,"heads")["fused"]]
                result=classification(truth,choice)
                flags={}
                if (task=="local" or band=="full" or (task=="remote" and arm=="fixed_none")):
                    flags["planted_positive"]=result["aa"]>=edges.GATES["positive_min_aa"]
                if task=="erased" and band==edge_shape.BAND:
                    flags["erased_null"]=result["aa"]<=edges.GATES["erased_gap_max_aa"]
                by_view[band]=dict(aa=result["aa"],ordinary_and_rare=result["per_class"],checks=flags)
            adequacy[arm]=by_view
        output[receipt["case"]]=dict(arms=results,receipt=file_pin(path)["sha256"],adequacy=adequacy)
    require(set(output)==set(edges.GROUPS),"Edge recovery or independent confirmation group incomplete")
    require(output[edges.REFERENCE]["arms"]["fixed_none"]["recovery_exact"],"Registered recovery proof absent")
    parity={}
    if gpu_dir is not None:
        for path in sorted(gpu_dir.glob("*.json")):
            right=json_pin(root,file_pin(path))
            scope(right,edges.SCHEMA,"independent_synthetic_confirmation",updates=0)
            left=raw[right["case"]]
            matched,different,max_abs=0,0,0.
            times={}
            for key in edge_shape.ARMS:
                a,b=left["arms"][key],right["arms"][key]
                require(a["state"]==b["state"] and a["checkpoint"]==b["checkpoint"],"Edge device checkpoint differs")
                for group in ("rows","frozen_deletions","region_replacements"):
                    for u,v in zip(a[group],b[group],strict=True):
                        require(all(u[k]==v[k] for k in ("truth","ids","spectral","geometry_hash","target_digest"))
                                and u["graph"]["q_shape"]==v["graph"]["q_shape"],"Edge device matched geometry differs")
                        for h in HEADS:
                            z,w=map(np.asarray,(u["heads"][h],v["heads"][h]))
                            matched+=len(z)
                            different+=int((z.argmax(1)!=w.argmax(1)).sum())
                            max_abs=max(max_abs,float(np.abs(z-w).max()))
                samples=[i for row in b["rows"] for i in row["times"]]
                values=[s["total_seconds"] for s in samples]
                times[key]=dict(p50=float(np.quantile(values,.5)),p95=float(np.quantile(values,.95)),
                                peak=max(s["peak_cuda_bytes"] for s in samples))
            parity[right["case"]]=dict(compared=matched,changed=different,max_abs_error=max_abs,
                                          costs=times,device=right["device"])
    return dict(cases=output,device=parity,historical=plan["historical"],effective_updates=36*edge_shape.STEPS,
                extra_reference_updates=edge_shape.STEPS,registered_gates=edges.GATES,
                confirmation_role="four_independent_narrow_boundary_blocks",
                labels_confirmed_only=True,exact_support_equalities=["count","in_degree","out_degree","squared_distance_multiset"])
