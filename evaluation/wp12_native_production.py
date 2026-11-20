"""Freeze and independently reduce same-workload production confirmation receipts."""

from itertools import product
import math
from pathlib import Path
import resource
from time import perf_counter

import numpy as np
import torch

from evaluation.classification_metrics import classification_metrics
from evaluation.frozen_training_stages import HEADS, file_pin, json_pin, probabilities, require
from model.research_config import digest
from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import _write
from transfer import native_production_campaign as campaign, native_production_fixture as fixture
from transfer import native_production_model as candidate
from transfer.g3_training import state_hash
from transfer.native_long_run import packet
from transfer.spatial_coverage_protocol import retained_positions
from transfer.xuzhou_multiscale_audit import _exact

SCHEMA = "wp12_native_production_independent_v1"
RAW = "wp12_native_production_frozen_rows_v1"
CAPTURED_REPEATS = 3


def endpoints(root,out,plan,plan_pin,producer,name):
    receipt = json_pin(root,file_pin(out/"cases"/name/"receipt.json"))
    archived = plan["design"][name]
    require(receipt["producer"] == producer and receipt["plan"] == plan_pin["sha256"]
            and receipt["case"] == name and receipt["updates"] == archived["budget"]
            and receipt["query_accessed"] is receipt["raw_gt_loaded"] is False and receipt["real_updates"] == 0,
            "Unknown or incomplete production result")
    spec = json_pin(root,receipt["spec"])
    end = packet(root,receipt["full"])
    best = min(end["validation"],key=lambda row:row["fused_nll"])
    require(end["registration"] == digest(spec) and end["cursor"] == len(end["history"]) == spec["steps"]
            and end["selected"]["step"] == best["step"] == receipt["selected"]
            and end["selected"]["fused_nll"] == best["fused_nll"]
            and all(item["supervised_area"] > 0 for item in end["history"]),
            "Selected production state or effective training budget differs")
    p = campaign.provider(root,out,plan,name,{"commit":producer})
    p.validate_history(spec,end)
    require(not set(p.roles["evaluation"]) & {row["tile"] for row in spec["validation"]}
            and not set(p.roles["evaluation"]) & {row["tile"] for row in spec["training"]},
            "Confirmation was exposed through optimization or selection")
    if name == campaign.REFERENCE:
        require(receipt["checkpoint_equal"] and all(_exact(end,packet(root,receipt["phases"][phase]["pin"]))
                    for phase in ("resume","completed")), "Exact native recovery evidence differs")
    return p, receipt, spec, end


def selected_model(p, receipt, end, device):
    state = end["selected"]["state"]
    rebuilt = p.model_factory(p.baseline(p.seed),p.config.arm,p.seed+7321).eval()
    rebuilt.load_state_dict(state,strict=True)
    exported = receipt["selected_export"]
    if exported is not None:
        path = Path(exported["weights"]["path"])
        require(file_pin(path) == exported["weights"] and file_pin(path.with_suffix(".pt.json")) == exported["metadata"],
                "Deployment state files differ from the receipt")
        loaded, config, axis = candidate.open_production(path,device)
        require(config == p.config and axis == p.axis and state_hash(loaded) == state_hash(rebuilt) == exported["state"],
                "Selected native training and deployment weights differ")
        return loaded
    return rebuilt.to(device)


def workload(plan, labels, roles):
    for tile in roles["evaluation"]:
        block = labels[str(tile)]
        trials = plan["matched_confirmation"] if block["domain"]=="in" else plan["shifted_confirmation"]
        for factor,band in trials:
            yield tile, factor, band, block


@torch.inference_mode()
def point_capture(p, model, tile, factor, band, checkpoint, execution):
    execution.sync()
    device = execution.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    actual, _, targets = p.confirmation(checkpoint,tile,factor,band)
    prepared = perf_counter()-started
    result, graph, timing = p.registered_forward(model,actual,"devval",execution)
    projected = [actual.lattice.project(x).detach().cpu().numpy() for x in result[:3]]
    execution.sync()
    total = perf_counter()-started
    if p.config.arm == "pixel":
        graph["q_nonempty"] = 0
    else:
        (indices, weights), _ = execution.graphs
        graph["q_nonempty"] = int(np.unique(indices[1][weights > 0]).size)
    width = actual.lattice.scene_shape[1]
    y0,_,x0,_ = actual.lattice.box
    # The projected array is in original pixel order even for a partial edge.
    ix = [(key//width-y0)*(actual.lattice.box[3]-x0)+key%width-x0 for key,_ in targets.points]
    rows = {head:p[ix].tolist() for head,p in zip(HEADS,projected,strict=True)}
    sample = dict(total_seconds=total,prepare_seconds=prepared,forward_seconds=timing["forward_seconds"],
                  graph_seconds=sum(timing["graph_seconds"]),
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                  process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    evidence = dict(truth=[label for _,label in targets.points], heads=rows, graph=graph,
                    retained_ids=list(actual.view.bands.band_ids), retained_sha256=actual.receipt["spectral"],
                    released=targets.summary(), grid=list(actual.lattice.shape), original_pixels=len(projected[0]))
    return evidence,sample


def verify_scope(data):
    require(data["schema"] == RAW and data["scope"] == "independent_synthetic_confirmation"
            and data["independent_real_accuracy"] is None and data["real_updates"] == 0
            and data["query_accessed"] is data["raw_gt_loaded"] is False,
            "Only independent synthetic confirmation can be reduced here")


def write_capture(root,out,report,code,producer,execution,only=None):
    plan, plan_pin = campaign.plan_for(root,out,{"commit":producer})
    report.mkdir(exist_ok=True, parents=True)
    require(only is None or only in fixture.SEEDS, "Unknown representative confirmation seed")
    proof = {}
    execution.capture_values = True
    for name, cell in plan["design"].items():
        p, receipt, spec, end = endpoints(root,out,plan,plan_pin,producer,name)
        if cell["input"].startswith("source"):
            proof[name] = dict(updates=end["cursor"], state=receipt["selected_export"]["state"],
                               parent=spec["initialization"]["source"], layout=plan["inputs"][cell["input"]]["kind"],
                               bank_sha256=p.bank.pin["sha256"], detector_bands=len(p.bank.release.bands.band_ids),
                               validation=[dict(step=r["step"],nll=r["fused_nll"]) for r in end["validation"]])
            continue
        if only is not None and cell["seed"] != only:
            continue
        destination = report/f"{name}.json"
        if destination.exists():
            existing = json_pin(root,file_pin(destination))
            verify_scope(existing)
            require(existing["device"] == execution.numeric and existing["producer"] == producer
                    and existing["consumer"] == code["commit"] and existing["case"] == name,
                    "Existing independent per-device receipt differs")
            continue
        model = selected_model(p,receipt,end,execution.device)
        before = state_hash(model)
        rows = []
        labels = plan["inputs"][cell["input"]]["blocks"]
        for tile,factor,band,block in workload(plan,labels,p.roles):
            baseline,_ = point_capture(p,model,tile,factor,band,receipt["full"],execution)
            times = []
            for _ in range(CAPTURED_REPEATS):
                measured,timing = point_capture(p,model,tile,factor,band,receipt["full"],execution)
                require(measured == baseline, "Repeated frozen cost call changed classification or graph")
                times.append(timing)
            rows.append(dict(tile=tile,factor=factor,band=band,mode=block["mode"],domain=block["domain"],
                             **baseline, repeats=times))
        require(before == state_hash(model), "Frozen deployment mutated selected weights")
        require_frozen_code(code)
        train_time = {phase: row["timing"] for phase,row in receipt["phases"].items()}
        _write(destination,dict(schema=RAW,scope="independent_synthetic_confirmation",case=name,producer=producer,
              consumer=code["commit"],plan=plan_pin["sha256"],checkpoint=receipt["full"]["sha256"],
              device=execution.numeric, initial=spec["initialization"],selected=receipt["selected"], model=before,
              parameters=spec["parameters"],config=cell["config"],rows=rows,effective_updates=end["cursor"],
              training_device=spec["numeric"], train_time=train_time, recovery_exact=receipt["checkpoint_equal"],
              optimization_and_selection_inputs=digest(dict(training=spec["training"],validation=spec["validation"],
                                                          inputs=spec["inputs"])),
              independent_real_accuracy=None,query_accessed=False,raw_gt_loaded=False,real_updates=0))
        print(f"frozen {name} {execution.device}",flush=True)
    source = report/"source-lineage.json"
    require(proof["source-a"]["parent"] is None and proof["source-b"]["parent"] == proof["source-a"]["state"],
            "Sequential two-source shared state is broken")
    if not source.exists():
        _write(source,proof)


def validate_row(p, row, task, checkpoint):
    tile,factor,band,block = task
    require((row["tile"],row["factor"],row["band"],row["domain"],row["mode"]) ==
            (tile,factor,band,block["domain"],block["mode"]), "Unexpected confirmation work item")
    master = p.bank.release.bands
    expected = [master.band_ids[i] for i in retained_positions(master,band)]
    require(row["retained_ids"] == expected and (band=="full" or len(expected)<len(master.band_ids)),
            "Omitted columns or full-band information entered the retained-view result")
    actual, _, targets = p.confirmation(checkpoint,tile,factor,band)
    require(row["retained_sha256"] == actual.receipt["spectral"] and row["released"] == targets.summary()
            and row["grid"] == list(actual.lattice.shape) and row["original_pixels"] == actual.receipt["original_pixels"]
            and row["truth"] == [label for _,label in targets.points], "Released point or retained input witness differs")
    require(set(row["heads"]) == set(HEADS) and len(row["repeats"]) == CAPTURED_REPEATS,
            "Independent response needs three heads and the same cost repetitions")
    for head in HEADS:
        probabilities(row["heads"][head],len(row["truth"]),p.classes)
    graph_ok = (row["graph"]["q_shape"] == [actual.receipt["pixels"],0] if p.config.arm=="pixel"
                else row["graph"]["q_shape"][0] == actual.receipt["pixels"]
                and row["graph"]["q_nonempty"] >= fixture.GATES["minimum_nonempty_q"]
                and row["graph"]["q_nnz"] >= actual.receipt["pixels"])
    require(graph_ok, "Confirmation changed grid or did not exercise multiple native regions")
    for sample in row["repeats"]:
        require(all(math.isfinite(sample[key]) and sample[key]>=0 for key in
                    ("total_seconds","prepare_seconds","forward_seconds","graph_seconds"))
                and sample["total_seconds"] >= max(sample["prepare_seconds"],sample["forward_seconds"]),
                "Malformed total cost measurement")


def scored(row):
    truth = row["truth"]
    result = {}
    for head in HEADS:
        decision = (np.asarray(row["heads"][head]).argmax(1)+1).tolist()
        value = classification_metrics(truth,decision,range(1,fixture.CLASSES+1))
        result[head] = dict(oa=value["oa"],aa=value["aa"],macro_f1=value["macro_f1"],
                            per_class=value["per_class"],confusion=value["confusion_matrix"],decisions=decision)
    repeats = row["repeats"]
    cost = {}
    for field in ("total_seconds","prepare_seconds","forward_seconds","graph_seconds"):
        values = [rep[field] for rep in repeats]
        cost[field] = dict(p50=float(np.quantile(values,.5)),p95=float(np.quantile(values,.95)))
    cost["peak_cuda_bytes"] = max((value["peak_cuda_bytes"] or 0 for value in repeats)) or None
    cost["process_peak_rss_kib"] = max(value["process_peak_rss_kib"] for value in repeats)
    return {key:row[key] for key in ("tile","factor","band","domain","mode","truth","grid","original_pixels",
                                    "retained_ids","retained_sha256","released","graph")} | dict(heads=result,cost=cost)


def paired_guard(reference, current, truth):
    require(len(reference["per_class"]) == len(current["per_class"]) == fixture.CLASSES,
            "Average AA cannot substitute for class-specific safeguards")
    by_class = {}
    a,b = reference["decisions"],current["decisions"]
    require(len(a) == len(b) == len(truth), "Paired class safeguards require the identical point roster")
    for before,after,label in zip(reference["per_class"],current["per_class"],range(1,fixture.CLASSES+1),strict=True):
        selected = [i for i,y in enumerate(truth) if y==label]
        require(before["class_id"] == after["class_id"] == label and
                before["support"] == after["support"] == len(selected) > 0,
                "Paired class denominator differs")
        gained = sum(a[i]!=label==b[i] for i in selected)
        lost = sum(a[i]==label!=b[i] for i in selected)
        delta = (gained-lost)/len(selected)
        by_class[str(label)] = dict(n=len(selected),gains=gained,losses=lost,recall_delta=delta,
            passed=(lost<=fixture.GATES["rare_added_errors"] if label==fixture.RARE
                    else delta>=fixture.GATES["ordinary_delta_recall"]))
    aa = current["aa"]-reference["aa"]
    return dict(aa_delta=aa,pass_aggregate=aa>=fixture.GATES["ordinary_delta_aa"],by_class=by_class,
                eligible=aa>=fixture.GATES["ordinary_delta_aa"] and all(v["passed"] for v in by_class.values()))


def compare_cpu_gpu(cpu, gpu):
    result = {}
    for name, (device_rows, _) in gpu.items():
        reference = cpu[name][0]
        require(device_rows["config"] == reference["config"] and device_rows["model"] == reference["model"],
                "CPU/GPU deployment checkpoint differs")
        row_differences=[]
        for a,b in zip(reference["rows"],device_rows["rows"],strict=True):
            require(all(a[k] == b[k] for k in ("tile","band","domain","mode","factor","truth","retained_sha256","retained_ids")),
                    "CPU/GPU released view or labels differ")
            require(a["graph"]["q_shape"] == b["graph"]["q_shape"], "CUDA and CPU graph dimensions differ")
            for head in HEADS:
                p,q = np.asarray(a["heads"][head]),np.asarray(b["heads"][head])
                row_differences.append((float(np.abs(p-q).max()),int((p.argmax(1)!=q.argmax(1)).sum())))
        maximum = max(item[0] for item in row_differences)
        changed = sum(item[1] for item in row_differences)
        result[name] = dict(maximum_probability_difference=maximum,
                            changed_predictions=changed, within_tolerance=maximum<=5e-5 and changed==0,
                            confirmation_views=len(reference["rows"]))
    return result


def read_capture(root,out,plan,link,folder,producer,code,*,subset=None):
    cases = {}
    for path in sorted(folder.glob("*.json")):
        if path.stem=="source-lineage":
            continue
        link_to_row = file_pin(path)
        row = json_pin(root,link_to_row)
        verify_scope(row)
        name = row["case"]
        require(name == path.stem and name in plan["design"] and not name.startswith("source")
                and row["plan"]==link["sha256"] and row["producer"]==producer and row["consumer"]==code["commit"],
                "Confirmation receipt does not belong to the registered production cohort")
        p,receipt,spec,end = endpoints(root,out,plan,link,producer,name)
        require(row["checkpoint"] == receipt["full"]["sha256"] and row["selected"] == receipt["selected"]
                and row["effective_updates"] == end["cursor"] and row["parameters"] == spec["parameters"]
                and row["model"] == state_hash(selected_model(p,receipt,end,"cpu")),
                "Confirmation point scores reference a different selected checkpoint")
        labels = plan["inputs"][plan["design"][name]["input"]]["blocks"]
        for evidence,expected in zip(row["rows"],workload(plan,labels,p.roles),strict=True):
            validate_row(p,evidence,expected,receipt["full"])
        cases[name] = row,link_to_row["sha256"]
    expected = {name for name,cell in plan["design"].items() if not name.startswith("source")
                and (subset is None or cell["seed"]==subset)}
    require(set(cases)==expected,"Independent production cohort or matched CUDA subset incomplete")
    return cases


def audit(root,out,cpu_folder,gpu_folder,code,producer):
    plan, link = campaign.plan_for(root,out,{"commit":producer})
    cpu = read_capture(root,out,plan,link,cpu_folder,producer,code)
    gpu = read_capture(root,out,plan,link,gpu_folder,producer,code,subset=fixture.SEEDS[0]) if gpu_folder else {}
    comparisons, gates, selected = {}, {}, {}
    results = {key:dict(receipt=sha,parameters=row["parameters"],selection=row["selected"],
                  budget=row["effective_updates"],train_time=row["train_time"],recovery_exact=row["recovery_exact"],
                  deployment_device=row["device"], training_device=row["training_device"],
                  views=[scored(item) for item in row["rows"]]) for key,(row,sha) in cpu.items()}
    for seed,k in product(fixture.SEEDS,fixture.SUPPORT):
        prefix = f"s{seed}-k{k}"
        group = {arm:cpu[prefix+"-"+arm][0] for arm in candidate.ARMS}
        require(len({case["optimization_and_selection_inputs"] for case in group.values()}) == 1
                and all(case["effective_updates"] == fixture.STEPS and case["training_device"] == group["native"]["training_device"]
                        for case in group.values())
                and len({group[arm]["initial"]["state"] for arm in candidate.ARMS[:3]})==1
                and len({group[arm]["parameters"] for arm in candidate.ARMS[:3]})==1
                and abs(group["pixel"]["parameters"]/group["native"]["parameters"]-1)<=.005,
                "Production candidate arms lack matched support, capacity, actual exposures or work")
        comparison = []
        for native_row, center_row, shift_row, pixel_row in zip(
                *(results[prefix+"-"+arm]["views"] for arm in candidate.ARMS),strict=True):
            require(all(all(row[key] == native_row[key] for key in ("truth","released","retained_sha256","grid"))
                    for row in (center_row,shift_row,pixel_row)), "Paired arms saw different retained observations")
            a,b,c,d = (row["heads"]["fused"] for row in (native_row,center_row,shift_row,pixel_row))
            view = {key:native_row[key] for key in ("mode","domain","band","factor")}
            contrast = dict(view=view, center_native=paired_guard(a,b,native_row["truth"]),
                            shift_native=paired_guard(a,c,native_row["truth"]),
                            pixel_native=paired_guard(a,d,native_row["truth"]))
            comparison.append(contrast)
        comparisons[prefix]=comparison
    for name,item in results.items():
        selected[name]=item["selection"]
        by_view = []
        for row in item["views"]:
            score = row["heads"]["fused"]
            checks = {}
            if row["band"] == "full" and row["mode"] in ("local","erased"):
                checks["full_positive"] = score["aa"] >= fixture.GATES["full_positive_min_aa"]
                checks["rare_positive"] = score["per_class"][-1]["recall"] >= fixture.GATES["minimum_rare_recall"]
            if row["band"]=="central_gap" and row["mode"]=="erased":
                checks["erased_null"] = score["aa"]<=fixture.GATES["erased_gap_max_aa"]
            if row["mode"]=="context" and name.endswith("-pixel"):
                checks["context_center_cannot_read_neighbors"] = score["aa"]<=fixture.GATES["reference_context_center_max_aa"]
            by_view.append(dict(mode=row["mode"],domain=row["domain"],band=row["band"],factor=row["factor"],
                                checks=checks))
        gates[name]=by_view
    gpu_cost = {name:dict(receipt=sha,device=value["device"],
                         views=[dict(domain=row["domain"],mode=row["mode"],band=row["band"],factor=row["factor"],
                                     cost=scored(row)["cost"]) for row in value["rows"]])
               for name,(value,sha) in gpu.items()}
    require_frozen_code(code)
    return dict(schema=SCHEMA,scope="independent_synthetic_confirmation",producer=producer,consumer=code["commit"],
                plan=link["sha256"],source_lineage=json_pin(root,file_pin(cpu_folder/"source-lineage.json")),
                qualified_pairs=6,model_seeds=list(fixture.SEEDS),support_seeds=list(fixture.SUPPORT),
                cohorts=results,paired_protection=comparisons,control_thresholds=fixture.GATES,
                control_results=gates,device_compare=compare_cpu_gpu(cpu,gpu),gpu_cost=gpu_cost,
                budgets=dict(matched_target=24*fixture.STEPS,sources=2*fixture.SOURCE_STEPS,
                             config_smoke=6,extra_same_device_reference=fixture.STEPS),
                independent_real_accuracy=None,real_updates=0,query_accessed=False,raw_gt_loaded=False)
