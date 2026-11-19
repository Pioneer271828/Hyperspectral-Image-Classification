"""Standalone point CSV/paired gate, checkpoint and device consumption."""

from collections import defaultdict
import csv
import math

import numpy as np

from evaluation.classification_metrics import classification_metrics
from evaluation.frozen_training_stages import HEADS, file_pin, json_pin, probabilities, require
from evaluation.wp12_locked_support import transitions
from evaluation.wp12_paired_head_capture import RAW, TRACE, METADATA, REPEATS
from transfer import native_paired_head as head, native_paired_head_campaign as campaign, native_paired_head_fixture as fixture

SCHEMA="wp12_paired_head_and_ordinal_review_v1"


def checked_scope(row):
    require(row["schema"]==RAW and row["scope"]=="independent_synthetic_confirmation"
            and row["real_updates"]==0 and row["independent_real_accuracy"] is None
            and row["query_accessed"] is row["raw_gt_loaded"] is False,
            "Development evidence cannot supply independent paired accuracy")


def signed_cases(root,folder,plan,link,consumer,case_names):
    result={}
    for path in sorted(folder.glob("*.json")):
        index=file_pin(path)
        row=json_pin(root,index)
        checked_scope(row)
        name=row["case"]
        require(name==path.stem and name in case_names and row["registration"]==link["sha256"]
                and row["consumer"]==consumer and row["producer"]==plan["producer"]
                and set(row["arms"])==set(plan["cases"][name]["policies"]), "Foreign paired publication")
        result[name]=(row,index["sha256"])
    require(set(case_names)==set(result),"Missing role or case in independent paired publication")
    return result


def validate_pair(group, plan, name):
    outline=plan["cases"][name]
    kind=outline["kind"]
    listed=[group["arms"][key] for key in outline["policies"]]
    reference=listed[0]
    classes=fixture.SPECS[kind][1]
    expected=list(range(1,classes+1))
    for arm in listed:
        require(all(arm[key]==reference[key] for key in ("initial","source","trainable","dimension","class_axis",
            "training_order","labels_seen","role","updates","train_device"))
            and arm["dimension"]==64 and arm["trainable"]==65*classes and arm["class_axis"]==expected
            and arm["updates"]==outline["budget"]
            and arm["selected"]==min(arm["calibration"],key=lambda row:row["nll"])["step"],
            "Equal-head work, source, original exposure or first-min selection differs")
        require(set(arm["layers"]["last"]["changed"])=={f'heads.{fixture.SPECS[kind][0]}.weight',
                        f'heads.{fixture.SPECS[kind][0]}.bias'}, "Frozen graph or CNN altered during head training")
        expected_sources=(kind,) if kind!="fallback" else ("fallback",*fixture.FINAL)
        expected_count=len(expected_sources)*len(fixture.VIEWS)*3*(len(METADATA) if kind=="fallback" else 1)
        require(len(arm["rows"])==expected_count,"Held axes, metadata control or original band comparison missing")
    if kind=="xuzhou":
        a,b=listed
        require(a["exposure"]=={"full/declared":outline["budget"]}
                and set(b["exposure"])=={f"{band}/declared" for band in fixture.VIEWS}
                and sum(b["exposure"].values())==outline["budget"],"Fixed three-view support cycle differs")
    else:
        require(all(a["exposure"]==reference["exposure"] for a in listed),"An arm changed the registered observation exposure")
    if name==campaign.REFERENCE[kind]:
        require(reference["recovery_exact"],"Paired resume/complete checkpoint proof is absent")


def matched_points(before,after):
    require(all(before[key]==after[key] for key in ("source","tile","band","metadata","layout","mechanism","truth",
              "points","spectral","geometry","fit","shape","band_ids")),"Paired head changed geometry, retained columns or truth")


def predicted(row,classes,readout):
    return (probabilities(row["heads"][readout],len(row["truth"]),classes).argmax(1)+1).tolist()


def pair_scores(truth,a,b,classes,scene,*,head_kind,guard_population):
    original=classification_metrics(truth,a,range(1,classes+1))
    changed=classification_metrics(truth,b,range(1,classes+1))
    cells=transitions(truth,a,b,classes)
    target=campaign.GATES["targets"][scene]
    selected=guard_population
    sufficient=all(cells[str(cls)]["n"]>=selected for cls in range(1,classes+1))
    delta=changed["aa"]-original["aa"]
    protects={}
    for key,value in cells.items():
        gain=value["gains"]-value["losses"]
        if value["n"]:
            recall= gain/value["n"]
            passed= recall>=campaign.GATES["ordinary_delta_recall"]
            if scene=="whu-honghu" and int(key) in target:
                passed=value["losses"]<=campaign.GATES["honghu_rare_added_errors"]
        else:
            recall,passed=None,False
        protects[key]=dict(recall_delta=recall,old=value["before"],new=value["after"],n=value["n"],
                           corrected=value["gains"],introduced=value["losses"],passed=passed)
    eligible=sufficient and delta>=campaign.GATES["aggregate_delta"] and all(row["passed"] for row in protects.values())
    targeted=(all(protects[str(y)]["corrected"]>=protects[str(y)]["introduced"] for y in target)
              and sum(protects[str(y)]["corrected"]-protects[str(y)]["introduced"] for y in target)
                  >=campaign.GATES["target_correct_gain_min"])
    return dict(head=head_kind,total=len(truth),original_aa=original["aa"],candidate_aa=changed["aa"],delta_aa=delta,
                original_oa=original["oa"],candidate_oa=changed["oa"],original_matrix=original["confusion_matrix"],
                new_matrix=changed["confusion_matrix"],transitions=cells,ordinary_and_rare=protects,
                adequate=sufficient,protected=eligible,target_gain=targeted,
                applied_population_threshold=guard_population,accepted=eligible and targeted)


def validate_option(kind,arm,base,actual):
    matched_points(base,actual)
    if kind!="fallback" or actual["metadata"]!="missing":
        require(base["heads"]["cnn"]==actual["heads"]["cnn"]
                and base["heads"]["graph"]==actual["heads"]["graph"] and base["graph"]==actual["graph"],
                "Frozen diagnostic heads or native graph changed under head-only comparison")
    if kind=="fallback" and actual["metadata"]=="missing" and arm!="legacy":
        require(actual["mode"]=="index" and actual["wavelength_quality"]=="index_only"
                and actual["order"]!=base["order"],"Ordinal fallback acquired a physical calibration")
    require(len(actual["times"])==REPEATS and all(math.isfinite(row["total"]) and
            row["total"]>=max(row["prep"],row["forward"]) for row in actual["times"]),
            "Native prediction cost is incomplete")


def primary(kind, pairs):
    if kind=="honghu":
        return [row for row in pairs if row[0]["band"]=="full"]
    return [row for row in pairs if row[0]["band"]=="central_gap" and row[0]["mechanism"]=="retained"
            and (kind!="fallback" or row[0]["metadata"]=="missing")]


def summarize_cases(cpu, plan, out):
    values={}
    fields=("scene","case","arm","source","block","layout","mechanism","band","metadata","head",
            "point_ordinal","truth","baseline","candidate")
    with (out/"paired-points.csv").open("w",encoding="utf-8",newline="") as stream:
        csv_out=csv.DictWriter(stream,fieldnames=fields)
        csv_out.writeheader()
        for name,(group,index) in cpu.items():
            validate_pair(group,plan,name)
            kind=plan["cases"][name]["kind"]
            scene,classes,*_=fixture.SPECS[kind]
            control,*arms=plan["cases"][name]["policies"]
            base=group["arms"][control]
            outcome={}
            for arm in arms:
                other=group["arms"][arm]
                records=defaultdict(list)
                pairs=list(zip(base["rows"],other["rows"],strict=True))
                for a,b in pairs:
                    validate_option(kind,arm,a,b)
                    key="|".join(a[k] for k in ("source","layout","mechanism","band","metadata"))
                    records[key].append((a,b))
                    for h in HEADS:
                        orig,new=(predicted(row,classes,h) for row in (a,b))
                        csv_out.writerows(dict(scene=scene,case=name,arm=arm,source=a["source"],block=a["tile"],
                          layout=a["layout"],mechanism=a["mechanism"],band=a["band"],metadata=a["metadata"],head=h,
                          point_ordinal=i,truth=y,baseline=u,candidate=v)
                          for i,(y,u,v) in enumerate(zip(a["truth"],orig,new,strict=True),1))

                def reduce(subset):
                    result={}
                    for h in HEADS:
                        ys,olds,news=[],[],[]
                        for a,b in subset:
                            ys.extend(a["truth"])
                            olds.extend(predicted(a,classes,h))
                            news.extend(predicted(b,classes,h))
                        result[h]=pair_scores(ys,olds,news,classes,scene,head_kind=h,
                                              guard_population=campaign.GATES["minimum_supported_confirmation_per_class"])
                    return result

                points=reduce(primary(kind,pairs))
                groups={key:reduce(rows) for key,rows in records.items()}
                negatives=[]
                for a,b in pairs:
                    if a["mechanism"]=="erased" and a["band"]=="central_gap":
                        label=b["truth"]
                        value=classification_metrics(label,predicted(b,classes,"fused"),range(1,classes+1))["aa"]
                        negatives.append(dict(source=a["source"],metadata=a["metadata"],aa=value,
                              passes=value<=campaign.GATES["synthetic_erased_gap_max_aa"]))
                latency=[r["total"] for row in other["rows"] for r in row["times"]]
                control_latency=[r["total"] for row in base["rows"] for r in row["times"]]
                outcome[arm]=dict(primary=points,by_source_layout_and_view=groups,erased_negative=negatives,
                    cpu_cost=dict(p50=float(np.quantile(latency,.5)),p95=float(np.quantile(latency,.95)),
                                  control_p50=float(np.quantile(control_latency,.5)),
                                  control_p95=float(np.quantile(control_latency,.95))),
                    accepted=points["fused"]["accepted"] and all(r["passes"] for r in negatives))
            values[name]=dict(scene=scene,kind=kind,receipt=index,arms=outcome,
                registered={name:{k:v for k,v in arm.items() if k!="rows"} for name,arm in group["arms"].items()})
    return values


def compare_devices(cpu,gpu):
    output={}
    for name,(row,sha) in gpu.items():
        sample=cpu[name][0]
        stats={}
        for key,arm in row["arms"].items():
            left=sample["arms"][key]
            require(arm["state"]==left["state"] and arm["checkpoint"]==left["checkpoint"],"Different selected device states")
            compared,changed,error=0,0,0.
            for a,b in zip(left["rows"],arm["rows"],strict=True):
                matched_points(a,b)
                require(a["graph"]["q_shape"]==b["graph"]["q_shape"] and a["order"]==b["order"],
                        "Native graph or metadata identity differs by device")
                for head in HEADS:
                    p,q=map(np.asarray,(a["heads"][head],b["heads"][head]))
                    compared+=len(p)
                    changed+=int((p.argmax(1)!=q.argmax(1)).sum())
                    error=max(error,float(np.abs(p-q).max()))
            timing=[time for review in arm["rows"] for time in review["times"]]
            totals=[time["total"] for time in timing]
            stats[key]=dict(compared=compared,changed=changed,max_abs=error,
                            p50=float(np.quantile(totals,.5)),p95=float(np.quantile(totals,.95)),
                            peak=max(time["peak_cuda_bytes"] for time in timing))
        output[name]=dict(receipt=sha,device=row["device"],arms=stats)
    return output


def review(root, program, cpu_dir, gpu_dir, out, producer, consumer, intake_pin, trace_pin):
    plan,link=campaign.load(root,program,producer)
    cpu=signed_cases(root,cpu_dir,plan,link,consumer,set(plan["cases"]))
    cuda=signed_cases(root,gpu_dir,plan,link,consumer,set(campaign.REFERENCE.values()))
    result=summarize_cases(cpu,plan,out)
    custody=json_pin(root,intake_pin)
    trace=json_pin(root,trace_pin)
    require(trace["schema"]==TRACE and trace["declared_backward_compatible"]
            and custody["scope"]=="read_only_inventory" and custody["real_updates"]==0,
            "Trusted read-only or retained source trace proof is missing")
    adoption={}
    for kind,arms in head.ARMS.items():
        for arm in arms[1:]:
            rows=[x["arms"][arm] for x in result.values() if x["kind"]==kind]
            require(len(rows)==6,"A paired model/support draw is missing")
            adoption[f"{kind}/{arm}"]=dict(seeds=6,targets_and_protection_passed=sum(
               row["primary"]["fused"]["accepted"] for row in rows), including_negative_controls=sum(row["accepted"] for row in rows),
               independent_synthetic_acceptance=all(row["accepted"] for row in rows),
               deploy_real=False)
    return dict(schema=SCHEMA,producer=producer,consumer=consumer,registration=link["sha256"],cases=result,
        source_ancestry=plan["multisource_ancestry"],support_draw_sizes=plan["support_draw_sizes"],
        k1_engineering=plan["role_k1_feasibility"],gates=plan["gates"],adoption=adoption,
        device=compare_devices(cpu,cuda),real_custody=custody,trace=trace,synthetic_updates=3936,reference_updates=304,
        point_csv_sha256=file_pin(out/"paired-points.csv")["sha256"],
        independent_real_accuracy=None,query_accessed=False,raw_gt_loaded=False,real_updates=0,wp04_executed=False)
