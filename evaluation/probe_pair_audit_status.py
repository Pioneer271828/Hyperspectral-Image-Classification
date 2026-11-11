"""Versioned endpoint audit lineage for descriptive WP06 paired statistics."""
from collections import Counter, defaultdict
import json

from evaluation import frozen_probe_summary as base, initial_final_probe_summary as pairs
from research_runtime import write_json


POLICY = "wp06_initial_terminal_pair_audit_lineage_v2"
KEYS = ("scene","seed","method","mode","layer","role","checkpoint")
GROUP = (*pairs.GROUP,"contrast","baseline_mode","baseline_layer","stage_comparison")
GATE_FIELDS = ("audit_status","eligible_for_formal_numeric_claim","endpoint_count","passed_endpoint_count",
               "failed_endpoint_count","not_audited_endpoint_count","probability_failed_endpoint_count",
               "decision_failed_endpoint_count")


def audit_endpoint(endpoint, leg):
    decisions = endpoint["strict_decisions"]
    probability = endpoint["probabilities_close"]
    if False in (decisions,probability):
        status = "failed"
    elif decisions is True and probability is True:
        status = "passed"
    else:
        status = "not_audited"
    return {**{key:endpoint[key] for key in ("case","layer","role","checkpoint","backbone_stage")},
        "leg":leg, "audit_status":status, "strict_decisions":decisions, "probabilities_close":probability}


def gate(endpoints, *, complete=True):
    counts = Counter(endpoint["audit_status"] for endpoint in endpoints)
    if counts["failed"]:
        status = "failed"
    elif not complete:
        status = "incomplete"
    elif counts["not_audited"] or not endpoints:
        status = "not_audited"
    else:
        status = "passed"
    return {"audit_status":status, "eligible_for_formal_numeric_claim":status=="passed",
        "endpoint_count":len(endpoints), "passed_endpoint_count":counts["passed"],
        "failed_endpoint_count":counts["failed"], "not_audited_endpoint_count":counts["not_audited"],
        "probability_failed_endpoint_count":sum(endpoint["probabilities_close"] is False for endpoint in endpoints),
        "decision_failed_endpoint_count":sum(endpoint["strict_decisions"] is False for endpoint in endpoints)}


def dependencies(row, endpoints):
    def get(stage, mode, layer, leg):
        key = (row["scene"],row["seed"],row["method"],mode,layer,row["role"],row["checkpoint"],stage)
        return audit_endpoint(endpoints[key],leg)

    if row["contrast"]=="terminal_minus_initial":
        return [get(stage,row["mode"],row["layer"],stage) for stage in ("terminal","initial")]
    stages = ("terminal","initial") if row["stage_comparison"]=="change" else (row["stage_comparison"],)
    return [endpoint for stage in stages for endpoint in (
        get(stage,row["mode"],row["layer"],stage+"_candidate"),
        get(stage,row["baseline_mode"],row["baseline_layer"],stage+"_baseline"))]


def annotate_groups(rows, keys, values):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = base.summarize(rows,keys,values)
    for group in result:
        sample = grouped[tuple(group[key] for key in keys)]
        seeds = {row["seed"] for row in sample}
        if len(seeds)!=len(sample) or len(seeds)>3:
            raise ValueError("Expected at most one dependency set per declared seed")
        ends = [endpoint for row in sample for endpoint in row["endpoint_audits"]]
        legs = defaultdict(list)
        for endpoint in ends:
            legs[endpoint["leg"]].append(endpoint)
        group.update(gate(ends,complete=len(seeds)==3), expected_seed_count=3,
                     dependency_audits={leg:gate(values,complete=len(seeds)==3) for leg,values in legs.items()})
    return result


def build(results):
    initial,terminal = (list(pairs.endpoints(results,name)) for name in ("initial","terminal"))
    all_endpoints = initial+terminal
    index = {tuple(row[key] for key in (*KEYS,"backbone_stage")):row for row in all_endpoints}
    if len(index)!=len(all_endpoints):
        raise ValueError("Duplicate authorized endpoint identity")
    contrasts = list(pairs.contrasts(initial,terminal))
    for row in contrasts:
        endpoints = dependencies(row,index)
        row.update(gate(endpoints),endpoint_audits=endpoints)
    keys = (*pairs.PAIR_KEYS,*GATE_FIELDS,"endpoint_audits")
    classes = list(pairs.class_rows(contrasts,keys,paired=True))
    groups = annotate_groups(contrasts,GROUP,base.METRICS)
    class_groups = annotate_groups(classes,(*GROUP,"class_id"),("recall","f1"))
    class_index = {tuple(row[key] for key in (*GROUP,"class_id")):row for row in class_groups}
    protection = list(pairs.protection(classes))
    for row in protection:
        group = class_index[tuple(row[key] for key in (*GROUP,"class_id"))]
        row.update({key:group[key] for key in (*GATE_FIELDS,"dependency_audits","expected_seed_count")})
        row["formal_passes_2pp"] = row["passes_2pp"] if row["eligible_for_formal_numeric_claim"] else None
    return contrasts,classes,groups,class_groups,protection


def write_outputs(out, results, source_receipt_sha):
    contrasts,classes,groups,class_groups,protection = build(results)
    out.mkdir(exist_ok=False)
    keys = (*pairs.PAIR_KEYS,*GATE_FIELDS,"endpoint_audits")
    for filename, rows, fields in (("contrasts.csv",contrasts,(*keys,*base.METRICS)),
                                   ("class-contrasts.csv",classes,(*keys,"class_id","support","recall","f1"))):
        base.write_csv(out/filename,[{key:json.dumps(row[key],ensure_ascii=False,separators=(",",":"))
                                      if key=="endpoint_audits" else row[key] for key in fields}
                                     for row in rows],fields)
    write_json(out/"summary.json",{"policy":POLICY,"schema_version":2,
        "source_v1_review_sha256":source_receipt_sha,
        "contrast_rows":len(contrasts),"class_contrast_rows":len(classes),
        "contrast_status_counts":dict(Counter(row["audit_status"] for row in contrasts)),
        "class_protection_status_counts":dict(Counter(row["audit_status"] for row in protection)),
        "three_seed_contrasts":groups,"three_seed_class_contrasts":class_groups,
        "class_protection":protection,"new_probe_updates":0,
        "scope":"Original v1 development metrics and numerical lineage; budget-end endpoints were not independently audited."})
    return ("contrasts.csv","class-contrasts.csv","summary.json")
