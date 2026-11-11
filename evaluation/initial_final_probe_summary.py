"""Describe paired changes and retain every released class and numerical failure."""
from collections import Counter,defaultdict
from fractions import Fraction
from statistics import mean

from evaluation import allscene_probe_registry as scenes,allscene_probe_summary as tables,frozen_probe_summary as base
from research_runtime import write_json


OUTPUTS = ("metrics.csv","class-metrics.csv","contrasts.csv","class-contrasts.csv","regional-dimensions.csv","summary.json")
METRIC_KEYS = ("scene","case","seed","method","mode","layer","role","checkpoint","backbone_stage","backbone_step",
               "selected_step","selected_at_budget_end","strict_decisions","probabilities_close","zero_recall_classes")
COMPARE = ("scene","seed","method","mode","layer","role","checkpoint","contrast","baseline_mode","baseline_layer")
PAIR_KEYS = (*COMPARE,"stage_comparison","strict_decisions")
GROUP = ("scene","method","mode","layer","role","checkpoint")
VALUES = (*base.METRICS,*(f"recall_{i}" for i in range(1,23)),*(f"f1_{i}" for i in range(1,23)))


def endpoints(results, stage):
    for pair in results:
        value = pair[stage]
        for row in base.metric_rows([value],scenes=True):
            step = pair["role"]["initial_step" if stage == "initial" else "terminal_step"]
            yield {**row,"backbone_stage":stage,"backbone_step":step}


def differences(before,after,keys):
    index = {tuple(row[key] for key in keys):row for row in before}
    for row in after:
        other = index.get(tuple(row[key] for key in keys))
        if other is None:
            continue
        strict = [item["strict_decisions"] for item in (other,row)]
        yield {**{key:row[key] for key in keys},"stage_comparison":"change",
            "strict_decisions":all(strict) if None not in strict else None,
            **{key:row[key]-other[key] for key in VALUES if row.get(key) is not None and other.get(key) is not None}}


def contrasts(initial,terminal):
    pair_key = ("scene","seed","method","mode","layer","role","checkpoint")
    for row in differences(initial,terminal,pair_key):
        yield {**row,"contrast":"terminal_minus_initial","baseline_mode":row["mode"],"baseline_layer":row["layer"]}
    summaries = {}
    for stage,rows in (("initial",initial),("terminal",terminal)):
        by_stage = []
        for checkpoint in ("selected","terminal"):
            chosen = [{**row,"checkpoint":"selected"} for row in rows if row["checkpoint"]==checkpoint]
            for row in base.paired_rows(chosen,scenes=True,values=VALUES):
                by_stage.append({**row,"checkpoint":checkpoint,"stage_comparison":stage,
                    "strict_decisions":row["strict_decisions"] if checkpoint=="selected" else None})
        yield from by_stage
        summaries[stage] = by_stage
    yield from differences(summaries["initial"],summaries["terminal"],COMPARE)


def class_rows(rows,keys,*,paired=False):
    for row in rows:
        for class_id in range(1,scenes.CLASSES[row["scene"]]+1):
            support = (10 if row["role"]=="train" else 5) if paired else row[f"support_{class_id}"]
            yield {**{key:row[key] for key in keys},"class_id":class_id,"support":support,
                "recall":row[f"recall_{class_id}"],"f1":row[f"f1_{class_id}"]}


def protection(rows):
    groups = defaultdict(list)
    fields = (*GROUP,"contrast","baseline_mode","baseline_layer","stage_comparison","class_id")
    for row in rows:
        if (row["role"]=="val" and row["checkpoint"]=="selected"
                and row["contrast"] in ("terminal_minus_initial","residual_minus_capacity_control","residual_minus_none")):
            groups[tuple(row[k] for k in fields)].append(row)
    for key,group in groups.items():
        sample = [Fraction(round(row["recall"]*row["support"]),row["support"]) for row in group]
        delta = mean(sample)
        yield {**dict(zip(fields,key,strict=True)),"seed_count":len(group),"mean_recall_delta_pp":float(delta*100),
            "passes_2pp":delta>=Fraction(-2,100) if len({row["seed"] for row in group})==3 else None}


def write_outputs(out,results):
    initial,terminal = (list(endpoints(results,name)) for name in ("initial","terminal"))
    metrics = initial+terminal
    paired = list(contrasts(initial,terminal))
    classes = list(class_rows(metrics,METRIC_KEYS))
    paired_classes = list(class_rows(paired,PAIR_KEYS,paired=True))
    dimensions = [{"scene":pair["role"]["target"],"case":pair["case"],"seed":pair["role"]["seed"],
        "method":pair["role"]["method"],"mode":pair["role"]["mode"],"layer":layer,**row}
        for pair in results for layer,row in pair["role"].get("regional_effective_dimensions",{}).items()]
    for filename, rows,keys in (
            (OUTPUTS[0],metrics,(*METRIC_KEYS,*base.METRICS)),
            (OUTPUTS[1],classes,(*METRIC_KEYS,"class_id","support","recall","f1")),
            (OUTPUTS[2],paired,(*PAIR_KEYS,*base.METRICS)),
            (OUTPUTS[3],paired_classes,(*PAIR_KEYS,"class_id","support","recall","f1")),
            (OUTPUTS[4],dimensions,("scene","case","seed","method","mode","layer","regions","initial","terminal","terminal_minus_initial"))):
        base.write_csv(out/filename,[{key:row[key] for key in keys} for row in rows],keys)
    statuses = Counter(row["initial"]["status"] for row in results)
    write_json(out/OUTPUTS[5],{"cases":dict(statuses),"endpoint_rows":len(metrics),"class_rows":len(classes),
        "contrast_rows":len(paired),"class_contrast_rows":len(paired_classes),"regional_dimension_rows":len(dimensions),
        "numerical_failures":[{"case":pair["case"],"scene":pair["role"]["target"],**failure}
            for pair in results for failure in pair["initial"].get("numerical_failures",[])],
        "three_seed_metrics":base.summarize(metrics,(*GROUP,"backbone_stage"),base.METRICS),
        "three_seed_class_metrics":base.summarize(classes,(*GROUP,"backbone_stage","class_id","support"),("recall","f1")),
        "three_seed_contrasts":base.summarize(paired,(*GROUP,"contrast","baseline_mode","baseline_layer","stage_comparison"),base.METRICS),
        "three_seed_class_contrasts":base.summarize(paired_classes,
            (*GROUP,"contrast","baseline_mode","baseline_layer","stage_comparison","class_id","support"),("recall","f1")),
        "class_protection":list(protection(paired_classes)),
        "scope":"Same-scene fixed development labels. Step-zero target model and selected terminal model; historical terminal probabilities reused."})
    return OUTPUTS
