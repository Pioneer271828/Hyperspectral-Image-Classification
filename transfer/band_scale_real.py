"""Frozen, released-only compatibility curves and the complete historical WP07 intake."""

from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from evaluation.wp12_p4_released_evidence import paviau_prepared, paviau_trained
from model.research_config import digest
from scripts import run_frozen_band_region as inherited
from scripts.run_frozen_band_region import _read, _state, check, pin
from transfer.band_scale import BANDS, FACTORS, POLICIES, can_cross, evidence, measure, released_view, validate_cross
from transfer.g3_training import state_hash
from transfer.native_released_preflight import target_head_baseline
from transfer.spatial_multiscale import initialize

PAVIA = inherited.PAVIA_PRIOR
PARDIR = Path(PAVIA).parent
SEEDS = (64,512,2048)
FOCUS = {"xuzhou":[5,6],"paviaU":[2,6],"whu-honghu":[15,21]}
COMPETITORS = {"xuzhou":{5:8,6:3}}
SOURCES = {
    "pavia_matrix": (f"{PARDIR}/aggregate/results/comparison-audit.json",
                     "728076ffe4751662cea2580e66c3b915d3fc5b30cecc1308ff1d761a3b8cc1bb"),
    "pavia_preflight": ("wp02-paviau-final-nov18/preflight/paviaU.json",
                        "61a5264dc021979803cda0d7bd279d9644f8a50505f256aeded876ab39a7b77d"),
    "pavia_64": (f"{PAVIA}/paviaU-64-2-joint.json",
                 "6b4a7cf9521ada7b8673864c40dea154bb24634b257681207708bf655ee3e073"),
    "pavia_512": (f"{PAVIA}/paviaU-512-2-joint.json",
                  "4462bfadf9c38985ad783422dc3bd6445362122e23e485886f812394836830ae"),
    "pavia_2048": (f"{PAVIA}/paviaU-2048-2-joint.json",
                   "c52a57860b19ec5ae56d0ba07b022dced1ebee6f713f66422cee5a339174f315"),
    "xuzhou_regions": ("wp07-xuzhou-frozen-regions03/study/analysis.json",
                       "286cf9bd043042a671cfe3f355c8f379a36e64f827867e002759cd7b035b9b67"),
    "older_region_readout": ("wp07-trained-region-analysis02/result/analysis.json",
                             "5d746f3ae2afe75949b51de1ad29d3e58500fcbbce9352cd62c64f5b69367c18"),
}
CASES = {**{f"paviaU-{s}":("paviaU",s,"joint") for s in SEEDS},
         "xuzhou":("xuzhou",64,"both_content"),
         "whu-honghu":("whu-honghu",None,"s512-k5-shared_transfer-none")}


def source_pins(root):
    return {key:pin(root/path,sha) for key,(path,sha) in SOURCES.items()}


def history(root, links, docs):
    current = {key:_read(value) for key,value in links.items()}
    trained = paviau_trained(links["pavia_matrix"],root,paviau_prepared(links["pavia_preflight"],root))
    check(trained["phase_receipts"]==117,"PaviaU historical phases differ")
    xz = docs["xz_audit"]
    check(xz["status"]=="passed" and xz["raw_gt"] is xz["query"] is False,"Foreign Xuzhou archive")
    old_xz = {arm:{end:{key:result[key] for key in ("step","metrics")}
                   for end,result in item["endpoints"].items()}
              for arm,item in xz["results"].items()}
    hh_receipt = docs["hh_predictions"]
    folder = root/Path(inherited.PINS["hh_predictions"][0]).parent
    archive = {}
    for row in docs["hh_scope"]["cases"]:
        key = row["case"]["id"]
        name = key+".json"
        item = _read(pin(folder/name,hh_receipt["artifacts"][name]))
        observations = {k:{h:v["metrics"] if "metrics" in v else v for h,v in value["heads"].items()}
                        for k,value in item["readouts"].items() if k.startswith("obs__")}
        archive[key] = dict(case=row["case"],selected=row["state"]["selected_step"],
                            sha256=hh_receipt["artifacts"][name],observations=observations)
    return dict(pavia=trained,xuzhou=old_xz,honghu=archive,
                archive_hashes={name:digest(current[name]) for name in ("xuzhou_regions","older_region_readout")},
                real_optimizer_updates=0, raw_gt_loaded=False, query_accessed=False)


def open_pavia(document,bank,intake,seed):
    reg = _read(document["registration"])
    phase = _read(document["stages"][-1])
    packet = _state(phase["checkpoint"])
    check(document["case"]==f"paviaU/{seed}/2/joint" and document["registration"]==phase["registration"]
          and reg["seed"]==seed and reg["factor"]==2 and reg["arm"]=="joint" and reg["bank"]==bank.pin
          and packet["registration"]==digest(reg) and packet["cursor"]==document["last_step"]
          and document["audit_optimizer_updates"]==document["completed_resume_updates"]==0,
          "Pavia selected device archive or source bank differs")
    baseline = target_head_baseline(intake.baseline,"paviaU",seed)
    model = initialize(baseline,"joint",reg["module_seed"])
    model.load_state_dict(packet["selected"]["state"],strict=True)
    model_identity = state_hash(model)
    check(model_identity==document["selected_model_sha256"] and packet["selected"]["step"]==document["selected_step"],
          "Pavia frozen weights disagree with verified manifest")
    return model,seed,model_identity,packet["selected"]["step"],phase["checkpoint"]["sha256"]


@torch.no_grad()
def run(root, code, name):
    started = perf_counter()
    scene,seed,arm = CASES[name]
    parents,docs = inherited.source_documents(root)
    links = source_pins(root)
    bank,intake = inherited._freeze_bank(root,scene,{"sources":parents})
    if scene=="paviaU":
        model,seed,ident,step,checkpoint = open_pavia(_read(links[f"pavia_{seed}"]),bank,intake,seed)
    elif scene=="xuzhou":
        model,seed,ident,step,_,checkpoint = inherited._xuzhou(docs,arm)
    else:
        model,seed,ident,step,_,checkpoint = inherited._new_honghu(root,docs,arm,inherited._honghu_context(docs["hh_scope"]))
    model.eval()
    model.backbone.adapter.cache_attention = False
    model.backbone.adapter.clear_attention_cache()
    policies = POLICIES if can_cross(model) else ("native",)
    population, lines, offset = [], [], 0
    for tile in bank.role_rows("devval"):
        points = [i for i,(_,label) in enumerate(tile["samples"]) if label in FOCUS[scene]]
        if not points:
            continue
        population.extend(tile["samples"][i][1] for i in points)
        for band,factor in product(BANDS,FACTORS):
            start = perf_counter()
            actual,_,targets = released_view(bank,tile,code["commit"],band,factor)
            obs = evidence(actual,targets)
            prep = perf_counter()-start
            check(obs["physical"]["nominal_gsd_m"] is None and actual.view.bands.wavelength_quality!="verified",
                  "Existing real release does not authorize physical selector metadata")
            reports = {policy:measure(model,actual,targets,code["commit"],policy=policy,groups=(policy=="native"))
                       for policy in policies}
            if len(policies)>1:
                validate_cross(reports)
            for policy,row in reports.items():
                selected = {head:[values[i] for i in points] for head,values in row.pop("scores").items()}
                groups = reports["native"].get("groups")
                lines.append(dict(tile=offset, event=dict(band=band,factor=factor,policy=policy),
                                  evidence=obs, timings=dict(materialize_seconds=prep if policy=="native" else 0.),
                                  points=[offset*100+i for i in range(len(points))],
                                  labels=[tile["samples"][i][1] for i in points], probabilities=selected,
                                  groups=[groups[i] for i in points], computation={k:v for k,v in row.items() if k!="groups"}))
        offset+=1
    check(all(population.count(cls)==5 for cls in FOCUS[scene]) and state_hash(model)==ident,
          "Released class denominator or frozen model differs")
    return dict(scene=scene,case=name, model=ident, architecture=arm, model_seed=seed,checkpoint=checkpoint,
                selected_step=step, preflight_population=bank.pin["sha256"],labels=population,
                competitor=COMPETITORS.get(scene,{}), supported_policies=list(policies),
                research_boundary_scope="sparse_released_labels_only_full_boundary_and_thin_unknown",
                no_cross_reason=None if len(policies)>1 else "checkpoint_does_not_own_both_trained_selector_banks",
                role="devval",rows=lines, elapsed_seconds=perf_counter()-started,
                real_updates=0,raw_gt_loaded=False, query_accessed=False)
