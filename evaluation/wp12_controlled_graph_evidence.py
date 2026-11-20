"""Join a bounded synthetic graph study as a supplement to production evidence."""

from collections import Counter
import csv
import math
from pathlib import Path

from research_runtime import hash_file
from transfer.registered_controlled_graph import FORMAT, MAXIMUM, STAGES, roster
from transfer.registered_wp11_learning import read


PRODUCER = "8caba56a1537aeb45b5ad9ab3c06b61d584f4dca"


def adapt(spec):
    if set(spec)!={"registration","audit","oracle","development_guard"}:
        raise ValueError("Pinned controlled registration, audit, independent oracle and development guard required")
    reg,audit,oracle,guard=(read(spec[k]) for k in ("registration","audit","oracle","development_guard"))
    frozen=read(audit["frozen"])
    claim=read(audit["claim"])
    cases={c["key"]:c for c in roster()}
    root=Path(reg["root"])
    if (reg["commit"]!=PRODUCER or reg["format"]!=FORMAT or reg["scope"]!="synthetic_regional_graph_only" or reg["maximum"]!=MAXIMUM
            or reg["cases"]!=list(cases.values()) or spec["registration"]["path"]!=str(root/"registration.json")
            or spec["audit"]["path"]!=str(root/"audit.json") or audit["registration"]!=spec["registration"]
            or audit["status"]!="passed_controlled_synthetic_graph" or audit["commit"]!=reg["commit"]
            or audit["actual"]!=MAXIMUM or audit["confirmation_cases"]!=len(cases) or set(audit["cases"])!=set(cases)
            or audit["frozen"]["path"]!=str(root/"frozen.json") or audit["claim"]["path"]!=str(root/"confirmation-claim.json")
            or frozen["registration"]!=spec["registration"] or frozen["commit"]!=reg["commit"] or frozen["actual"]!=MAXIMUM
            or frozen["confirmation_accesses"]!=0 or frozen["completed_reentry_updates"]!=0 or set(frozen["models"])!=set(cases)
            or claim!=dict(registration=spec["registration"],frozen=audit["frozen"],once=True)
            or oracle["status"]!="passed_independent_controlled_graph" or oracle["registration"]!=spec["registration"]
            or oracle["audit"]!=spec["audit"] or oracle["producer"]!=reg["commit"] or oracle["actual_updates"]!=MAXIMUM
            or guard["status"]!="passed_development_schema_guard" or guard["role_scope"]!=["devtrain","devval"]
            or guard["data_bytes_read"]!=0 or guard["model_updates"]!=0
            or any(row[k]!=0 for row in (reg,audit,frozen,oracle,guard) for k in ("real_updates","gt","query"))):
        raise ValueError("Synthetic graph producer, freeze, roster, real permissions or budget differs")
    actual=Counter()
    engineering={}
    for name,case in cases.items():
        item=audit["cases"][name]
        witness=frozen["models"][name]
        if (item["case"]!=case or witness["case"]!=case or item["selected"]!=witness["selected"]
                or item["last_support"]!=witness["last_support"] or item["confirmed_once"] is not True
                or hash_file(witness["selected_weights"]["path"])!=witness["selected_weights"]["sha256"]
                or len(item["history"])!=case["budget"] or set(item["phase_costs"])!=set(STAGES)):
            raise ValueError("Selected graph state, history or fixed confirmation boundary differs")
        for stage,expected in zip(STAGES,(case["budget"],7,case["budget"]-7,0),strict=True):
            status=read(witness["attempts"][stage])
            phase=item["phase_costs"][stage]
            if (status["status"]!="passed" or phase["updates"]!=expected or status["updates"]!=expected
                    or phase["cost"]!=status["cost"] or phase["wall_seconds"]!=status["wall_seconds"]
                    or not math.isfinite(phase["wall_seconds"]) or phase["wall_seconds"]<=0
                    or any(not math.isfinite(v) or v<0 for v in phase["cost"].values())):
                raise ValueError("Actual graph interval or timing differs")
            actual[case["kind"]]+=phase["updates"]
        if case["kind"]=="evolution":
            engineering[name]=dict(nnz=[r["nnz"] for r in item["history"]],
                budgets=[r["budget"] for r in item["history"]],events=item["graph_events"],costs=item["phase_costs"],
                selected_nnz=len(item["selected_support"]),last_nnz=len(item["last_support"]),fixed_donor=witness["donor"])
    if dict(actual)!=MAXIMUM:
        raise ValueError("Summed synthetic graph budget differs")
    return dict(scope="synthetic_regional_graph_only",**spec,producer=reg["commit"],
        actual_optimizer_updates=dict(actual),completed_reentry_updates=0,real_model_rows_added=0,
        controlled_learning=dict(cases=36,aggregate={k:v for k,v in audit["aggregate"].items() if k.startswith("signal/")}),
        support_engineering=dict(cases=18,trajectory=engineering,aggregate={
            k:v for k,v in audit["aggregate"].items() if k.startswith("evolution/")}),
        real_efficacy=dict(status="NOT_RUN",real_updates=0,reason="Fixed-final matched real development geometry and efficacy are pending"),
        gpu=dict(status="NOT_RUN"),optimizer_updates=0,gt=0,query=0)


def write(out,summary):
    with (out/"controlled-graph-synthetic.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(("kind","topology","comparison","metric","mean","sample_sd","scope"))
        for group in ("controlled_learning","support_engineering"):
            for key,values in summary[group]["aggregate"].items():
                for metric,row in values.items():
                    writer.writerow((*key.split("/"),metric,row["mean"],row["sample_sd"],summary["scope"]))
