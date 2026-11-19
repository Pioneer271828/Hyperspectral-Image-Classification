"""Read-only inventory and executable-shape preregistration for three real cohorts."""

from collections import Counter
from math import isfinite
from statistics import median

from evaluation.frozen_training_stages import json_pin, require
from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import _write
from transfer.wp_paired_structure import REAL, SCHEMA, VIEWS
from transfer.wp_paired_structure_fixture import OFF, SCALES, designs, message, pair

SOURCES = {
    "prior_comparison": ("wp05-wp08-cross-backend-20261120-qa/history7.json",
                         "0fe12fa2c84091a4f05abc6af2d7f72d419ee0a5531a34bc5957afd37576cf2d"),
    "pavia_gpu5": ("wp08-paviau-gpu5-rerun-20261118/audit/comparison-audit.json",
                   "aa786164e85638cda6b57b4b1af3c0708dd0db2b10c526aeff400e0d98cc061c"),
}
COST_BANDS = {"xz":(.65,2.),"pavia":(.8,2.2),"hh":(.5,2.4)}
SEEDS = (64,512,2048)


def proposed_groups():
    xz = designs([])["xz-s64-f1"]
    arms = {k:v for k,v in xz["arms"].items() if k!="fixed1"}
    routes = [p for p in xz["pairs"] if p["candidate"]!="fixed1"]
    scale = dict(off=OFF,scale=SCALES,message=message("combined"))
    alternatives = [pair("off","scale"),pair("off","message"),pair("scale","message")]
    panels = (("xz",(1,),256,32,arms,routes),
              ("pavia",(1,2),120,15,scale,alternatives),
              ("hh",(1,),256,32,scale,alternatives))
    return {f"{release}-s{seed}-f{f}":dict(release=release,seed=seed,factor=f,steps=steps,
                clock=list(range(0,steps+1,interval)),arms=models,pairs=pairs,device=None,purpose="science")
            for release,factors,steps,interval,models,pairs in panels for seed in SEEDS for f in factors}


def predicted_window(groups, rates, cards):
    loads = [[0.,0.] for _ in range(cards)]
    for label,g in sorted(groups.items(),key=lambda row:-row[1]["steps"]*len(row[1]["arms"])*rates[row[1]["release"]][1]):
        worker = min(loads,key=lambda item:item[1])
        for i,value in enumerate(rates[g["release"]]):
            worker[i] += value*g["steps"]*len(g["arms"])
    return [round(max(row[i] for row in loads)/3600,2) for i in range(2)]


def draft(root,out,code,evidence):
    pins = {key:dict(path=str(root/path),sha256=sha) for key,(path,sha) in SOURCES.items()}
    inventory = {key:json_pin(root,value) for key,value in pins.items()}
    history,pavia = inventory["prior_comparison"],inventory["pavia_gpu5"]
    require(history["new_real_optimizer_updates"]==0 and pavia["optimizer_updates"]==1260
            and not pavia["query_authorized"],"Reference does not bind prior released runs")
    real_xz = [item["training_cost"]["per_stage"]["full"]["update_loop_seconds"]/256
               for item in history["xuzhou_full"]["arms"].values()]
    real_pavia = [sum(stage["wall_seconds"] for stage in case["phase_costs"][:2])/60
                  for case in pavia["cases"]]
    actual = {"xz":dict(scope="historical_real_cpu_256_full_band",arms=len(real_xz),seconds_per_update=median(real_xz)),
              "pavia":dict(scope="historical_real_gpu5_two_views_prefix_resume",arms=len(real_pavia),
                           seconds_per_update=median(real_pavia)),
              "hh":dict(scope="historical_real_cpu_native_24_step_call_mean",arms=None,
                        seconds_per_update=16.56/24,
                        report="research/reports/WP05HongHu_完整预算多种子与批次接受工程_20261111.md")}
    result = json_pin(root,evidence)
    require(result["producer"]==code["commit"] and result["scope"]=="test_external_format_only"
            and result["real_optimizer_updates"]==0 and not result["query_accessed"],"Engineering rates need the final pinned run")
    panels = {}
    for name,arms in result["arms"].items():
        if not name.endswith(("-f1","-f2")):
            continue
        for arm,row in arms.items():
            rate = sum(v["wall_seconds"] for v in row["timings"])/row["budget"]
            require(rate>0 and isfinite(rate),"Engineering timing is not finite")
            panels.setdefault(name.split("-")[0],[]).append(
                dict(case=f"{name}/{arm}",seconds_per_update=rate,device=row["device"]["physical_uuid"] or "cpu"))
    groups = proposed_groups()
    rates = {s:[actual[s]["seconds_per_update"]*x for x in COST_BANDS[s]] for s in actual}
    counts,updates = Counter(),Counter()
    for group in groups.values():
        counts[group["release"]] += len(group["arms"])
        updates[group["release"]] += len(group["arms"])*group["steps"]
    prerequisites = {
        "xz":dict(primary="same_direction_local_versus_combined_at_factor1",
                  sources="Original shared initial tensors may be reused as a source only after a new Bank binds the exact 29-tensor loading.",
                  missing=["common verified drop-view and three-role independent Bank","separate real execution authority"]),
        "pavia":dict(primary="both_enabled_backends_versus_the_same_off_within_each_factor",
                     sources="The archived held-out source may supply the four shared tensor families; reset target graph and heads.",
                     missing=["independent intact class and band release","common factor1 and factor2 training approval"]),
        "hh":dict(primary="both_enabled_backends_versus_native_with_class15_and21_protection",
                  sources="Scratch cohort is preregistered; the native archives inform timing and failure scope.",
                  missing=["22-class independently released roles with adequate 15/21 denominators","matched structure execution permit"]),
    }
    template = dict(schema=SCHEMA,scope=REAL,groups=groups,releases={
        key:dict(manifest=None,authority=None,source=None,k=5) for key in counts})
    require_frozen_code(code)
    return _write(out,dict(schema="wp05_wp08_pending_real_preregistration_v1",producer=code["commit"],
        execution_authorized=False,source_pins=pins,engineering_evidence=evidence,observed_engineering=panels,
        observed_real=actual,hypothetical_sensitivity_multipliers=COST_BANDS,seconds_per_update_projection=rates,
        scheduled_wall_hours={str(c):predicted_window(groups,rates,c) for c in (1,2,3)},
        estimates_exclude=["resource wait","unpublished geometry costs","heavy confirmation and owner review"],
        budgets=dict(models=dict(counts),optimizer_updates=dict(updates),total=sum(updates.values())),
        retained_views=list(VIEWS),prerequisites=prerequisites,request_template=template,approval=None,
        query_accessed=False,raw_gt_loaded=False,optimizer_updates=0))
