"""Index exact historical paired-point and confusion-only evidence without inference."""

from collections import defaultdict
from pathlib import Path
import re
import subprocess

from evaluation.frozen_training_stages import HEADS, json_pin, pinned, require
from evaluation.wp12_external_paired import CAPTURE, REVIEW
from evaluation.wp12_paired_head_review import pair_scores
from evaluation.wp12_release_results import P4_REVIEW, csv_rows, p4_partial, proof
from model.research_config import digest
from scripts import run_frozen_band_region as archive
from transfer import native_external_paired as external

SCHEMA = "wp12_paired_cross_protocol_history_v1"
MECHANISM = "wp05-wp06-mechanism-20261119-qa/accepted"
PAIRED = "wp05-wp06-paired-heads-20261119-qa"
PINS = {
    "mechanism": (MECHANISM+"/mechanism.json", "656a955b7c01a36594b177536c031d4f92eadf0860fae9195e7ef86bbd9367cc"),
    "mechanism_points": (MECHANISM+"/development-points.csv", "0f9fcc38637018c74cd88b341ba302579de19866144d1620b915915c86b93f73"),
    "paired": (PAIRED+"/accepted/paired-review.json", "d5f0355355688c50013a10279044ae9783b7262aacaa448318e110e444ed49db"),
    "paired_points": (PAIRED+"/accepted/paired-points.csv", "1c7826a36cbc5ddc2b85e5e07559a622ae06c4e9523b80f16c9ff5a4d88b3179"),
    "inventory": (PAIRED+"/proofs/inventory.json", "0c2c7a490a75e5dcfa0a1abfc3ddee9acc8fc4aac918df9ab2daece261ffe13a"),
}
XZ_STAGES = {"initial":0,"prefix":7,"selected":347,"last":512}
FIVE_ARMS = {"none","anchor_only","binary","continuous","shuffled"}


def focus_from_confusion(confusion, focus):
    result = {}
    for cls in focus:
        n = sum(confusion[cls-1])
        count = confusion[cls-1][cls-1]
        result[str(cls)] = dict(n=n,correct=count,wrong=n-count)
    return result


def focus_from_pairs(classes, focus):
    result = {}
    for row in classes:
        cls = row["class_id"]
        if cls in focus:
            n = row["denominator"]
            result[str(cls)] = dict(n=n,before=row["source_correct"],after=row["target_correct"],
                baseline_wrong=n-row["source_correct"],gains=row["error_to_correct"],losses=row["correct_to_error"],
                lost_to=row["lost_to"],gained_from=row["gained_from"])
    return result


def frozen_views(root, links):
    report = json_pin(root,links["mechanism"])
    require(report["development"]["csv_sha256"]==links["mechanism_points"]["sha256"]
            and report["real_updates"]==0 and report["query_accessed"] is report["raw_gt_loaded"] is False,
            "Foreign frozen development point receipt")
    stages = {**XZ_STAGES,"whu-honghu":24}
    baseline = defaultdict(dict)
    for row in csv_rows(root,links["mechanism_points"]):
        case = row["case"]
        if case not in stages or row["site"]!="original" or row["control"]!="base":
            continue
        key = (case,row["target"],row["head"])
        ordinal = int(row["point"])
        require(ordinal not in baseline[key] and int(row["step"])==stages[case],
                "Repeated or foreign frozen point or step")
        require(row["scene"]==("whu-honghu" if case=="whu-honghu" else "xuzhou"),
                "Foreign frozen point scene")
        baseline[key][ordinal] = (int(row["truth"]),int(row["predicted"]))
    views = ("central_gap","matched_dispersed")
    output = {}
    for name,step in stages.items():
        rec = report["development"]["cases"][name]
        scene = rec["scene"]
        width = 9 if scene=="xuzhou" else 22
        n = rec["total"]
        metrics = {}
        population = None
        for target in ("full",*views):
            metrics[target] = {}
            for head in HEADS:
                full = baseline[name,"full",head]
                retained = baseline[name,target,head]
                require(sorted(full)==sorted(retained)==list(range(1,n+1))
                        and all(full[i][0]==retained[i][0] for i in full),
                        "Frozen views do not share the same released point order")
                truth,old,new = zip(*[(full[i][0],full[i][1],retained[i][1]) for i in sorted(full)],strict=True)
                population = digest(truth) if population is None else population
                require(digest(truth)==population,"Diagnostic heads disagree on released point order")
                seen = pair_scores(truth,old,new,width,scene,head_kind=head,guard_population=20)
                parent = rec["baseline"][target][head]
                require(all((value["n"],value["before"],value["after"],value["losses"])==
                            tuple(parent[cls][field] for field in ("n","full_correct","after","new_errors_from_full"))
                            and value["n"]-value["before"]==parent[cls]["already_wrong_in_full"]
                            for cls,value in seen["transitions"].items()),
                        "Frozen point counts differ from the accepted parent's exact split")
                metrics[target][head] = dict(
                    total=n, before_aa=seen["original_aa"], after_aa=seen["candidate_aa"],
                    focus={str(c):{**{field: seen["transitions"][str(c)][field]
                                  for field in ("n","before","after","gains","losses")},
                                  "baseline_wrong":seen["transitions"][str(c)]["n"]-
                                      seen["transitions"][str(c)]["before"]} for c in rec["focus"]})
        output[name] = dict(scene=scene,step=step,checkpoint=rec["checkpoint"],source=rec["source"],
                            authority=rec["authority"],population_order=population,role="archived_devval",views=metrics)
    return output


def honghu_short(root):
    links,docs = archive.source_documents(root)
    scope = docs["hh_scope"]
    receipt = docs["hh_predictions"]
    require(receipt["scope_sha256"]==links["hh_scope"]["sha256"]
            and scope["query_accessed"] is scope["raw_gt_loaded"] is False and len(scope["cases"])==40,
            "Foreign HongHu short archived source")
    ctx = archive._honghu_context(scope)
    cohorts = defaultdict(dict)
    for saved in scope["cases"]:
        case = saved["case"]
        state = saved["state"]
        pinned(root,dict(path=state["path"],sha256=state["state_sha256"]))
        require(Path(state["path"]).name=="state.pt","Foreign short-run saved checkpoint")
        archived = archive._archive(root,docs,case["id"],ctx)
        require(archived["model"]==state["selected"] and archived["selected"]==state["selected_step"]
                and archived["population"]==saved["populations"]["val"]["sha256"],
                "HongHu archived same-point predictions lost their checkpoint or population identity")
        key = f's{case["support_seed"]}-k{case["k"]}-{case["method"]}'
        cohorts[key][case["arm"]] = dict(case=case["id"], support_seed=case["support_seed"], k=case["k"],
            method=case["method"], model_seed=archived["model_seed"], selected_step=state["selected_step"],
            selected_state=state["selected"], initial_state=state["initial"], saved_weights=state["state_sha256"],
            pair=saved["pair_sha256"], support=saved["populations"]["train"]["sha256"], validation=archived["population"],
            role="historical_selection_devval", archived_predictions=archived["source"],
            views={name:{head:dict(total=value["denominator"], before_aa=value["source_aa"], after_aa=value["target_aa"],
                                    focus=focus_from_pairs(value["classes"],(15,21))) for head,value in heads.items()}
                   for name,heads in archived["paired"].items()})
    require(len(cohorts)==8 and all(set(arms)==FIVE_ARMS for arms in cohorts.values()),
            "Incomplete historical HongHu five-arm cohorts")
    for arms in cohorts.values():
        for identity in ("support","validation","pair","role"):
            require(len({arm[identity] for arm in arms.values()})==1,"Historical short cohort identity differs by arm")
    return dict(pins={key:links[key] for key in ("hh_scope","hh_predictions","hh_audit")},
                scopes=dict(classification="historical_devval", transitions="ordered_points_same_selected_state",
                            across_optimizer_arms=None),cohorts=dict(cohorts))


def honghu_120(root):
    rows,gaps = p4_partial(root)
    parent = json_pin(root,proof(root,"p4"))
    require(parent["committed_optimizer_updates"]==1200
            and all(item["optimizer_updates"]==120 for item in parent["cases"]),
            "Authorized long review no longer covers ten cases with 120 aggregate updates across two paths")
    fields = ("support_seed","k","method","model_seed","arm")
    identities = {tuple(item["case"][field] for field in fields):item for item in parent["cases"]}
    groups = defaultdict(dict)
    for row in rows:
        item = identities[(row["support"],row["k"],row["method"],row["seed"],row["arm"])]
        require(item["paired"]["state"]==row["model"],"Long reviewed model differs from its intake")
        cfg = item["case"]
        require(item["populations"]["val"]["sha256"] and item["populations"]["train"]["sha256"],
                "Long reviewed case lost its frozen train/validation population identity")
        key = f's{row["support"]}-k{row["k"]}-{row["method"]}-m{row["seed"]}'
        groups[key][row["arm"]] = dict(case=cfg["id"],support_seed=row["support"],k=row["k"],method=row["method"],
            model_seed=row["seed"],model_state=row["model"]["model"],selected_state=row["model"]["best"],
            selected=item["paired"]["selected_step"],
            support=item["populations"]["train"]["sha256"],validation=item["populations"]["val"]["sha256"],
            roles=row["role"],device=row["device"],heads={head:dict(total=sum(map(sum,value["confusion_matrix"])),
                  oa=value["oa"],aa=value["aa"],focus=focus_from_confusion(value["confusion_matrix"],(15,21)))
                  for head,value in row["heads"].items()})
    for roster in groups.values():
        require(set(roster)==FIVE_ARMS and len({v["validation"] for v in roster.values()})==1
                and len({v["support"] for v in roster.values()})==1,"Long compared cohort has foreign support or role")
        baseline = roster["none"]
        for row in roster.values():
            for head in HEADS:
                for cls,current in row["heads"][head]["focus"].items():
                    old = baseline["heads"][head]["focus"][cls]
                    require(current["n"]==old["n"],"Long confusion rows changed the class denominator")
                    current.update(delta_correct=current["correct"]-old["correct"],gains=None,losses=None)
    return dict(review=proof(root,"p4"),archive=P4_REVIEW,scope="reviewed_confusion_margins_only",
                paths_per_case=2,optimizer_steps_per_path=60,aggregate_optimizer_updates_per_case=120,
                point_transition_join_available=False,cohorts=dict(groups),coverage=gaps)


def xuzhou_protection(root, links):
    parent = json_pin(root,links["paired"])
    require(parent["point_csv_sha256"]==links["paired_points"]["sha256"]
            and parent["real_updates"]==0 and parent["independent_real_accuracy"] is None,
            "Foreign paired synthetic point receipt")
    records = defaultdict(dict)
    for row in csv_rows(root,links["paired_points"]):
        if (not row["case"].startswith("xuzhou-") or row["arm"]!="rotating_support"
                or row["mechanism"]!="retained" or row["band"]!="central_gap" or row["head"]!="fused"):
            continue
        key = (row["source"],int(row["block"]),int(row["point_ordinal"]))
        require(key not in records[row["case"]],"Duplicate paired point")
        records[row["case"]][key] = tuple(int(row[name]) for name in ("truth","baseline","candidate"))
    require(len(records)==6,"Six historical Xuzhou paired cohorts are required")
    profiles = {}
    for case,rows in records.items():
        truth,old,new = zip(*rows.values(),strict=True)
        observed = pair_scores(truth,old,new,9,"xuzhou",head_kind="fused",guard_population=5)
        previous = parent["cases"][case]["arms"]["rotating_support"]["primary"]["fused"]
        require(observed==previous and len(rows)==90,"Recounted protected point gates do not match source")
        violations = {}
        lost = {}
        for cls,item in observed["ordinary_and_rare"].items():
            if not item["passed"]:
                violations[cls] = item
            for (site,block,ordinal),(y,before,after) in rows.items():
                if y==int(cls) and before==y and after!=y:
                    lost.setdefault(cls,[]).append(dict(site=site,block=block,ordinal=ordinal,predicted=after))
        profiles[case] = dict(violations=violations,lost_points=lost,adequate=observed["adequate"],
                              aggregate_aa_delta=observed["delta_aa"],target_gain=observed["target_gain"],
                              classes=observed["ordinary_and_rare"],accepted=observed["accepted"])
    totals = {str(c): {field:sum(row["classes"][str(c)][field] for row in profiles.values())
              for field in ("n","old","new","corrected","introduced")} for c in range(1,10)}
    return dict(scope="historical_synthetic_central_gap_retained_signal",pairs=profiles,totals=totals,
                rejected=sum(not row["accepted"] for row in profiles.values()),retuning=False)


def review_attachment(root,link,consumer):
    review = json_pin(root,link)
    require(review["schema"]==REVIEW and review["raw_gt_loaded"] is review["query_accessed"] is False,
            "Foreign independent paired review")
    raw = json_pin(root,review["raw"])
    selected = json_pin(root,raw["selected"])
    registration = json_pin(root,selected["registration"])
    require(raw["schema"]==CAPTURE and selected["schema"]==external.PUBLISH
            and registration["schema"]==external.SCHEMA
            and review["scope"]==raw["scope"]==selected["scope"]==registration["scope"]
            and review["scope"] in (external.TEST,external.REAL)
            and review["runtime"]==raw["producer"]==selected["producer"]==registration["producer"]
            and set(raw["arms"])==set(selected["members"])
            and set(review["results"])==set(registration["releases"]),"Independent publication chain differs")
    for ancestor,descendant in ((review["runtime"],review["consumer"]),(review["consumer"],consumer)):
        require(re.fullmatch("[0-9a-f]{40}",ancestor),"Expected a full historical source commit")
        subprocess.run(["git","merge-base","--is-ancestor",ancestor,descendant],check=True)
    require(review["scope"]!=external.TEST or review["real_accuracy_available"] is False
            and review["external_real_updates"]==0,"Test fixture was promoted as real evidence")
    return dict(pin=link,scope=review["scope"],groups=len(review["results"]),
                real_accuracy_available=review["real_accuracy_available"])


def index(root):
    links = {key:dict(path=str(root/path),sha256=sha) for key,(path,sha) in PINS.items()}
    inventory = json_pin(root,links["inventory"])
    require(inventory["scope"]=="read_only_inventory" and inventory["real_updates"]==0
            and inventory["query_accessed"] is inventory["raw_gt_loaded"] is False,
            "Historical authorization inventory was promoted")
    hh = inventory["honghu"]
    require(hh["physical_saved_cursors"]==[24,24] and hh["k"]==5 and hh["guarded_support_seed"]==512,
            "Foreign same-protocol HongHu recovery evidence")
    cursors = {step:{head:focus_from_confusion(row["confusion"],(15,21)) for head,row in heads.items()}
               for step,heads in hh["observations"].items()}
    full_scene, hardware = (json_pin(root,proof(root,item)) for item in ("old_combined","independent"))
    require(len(full_scene["combined"]["rows"])==hardware["case_count"]==54 and hardware["new_query_reads"]==0
            and full_scene["preregistered_benefit_conclusion"]==hardware["preregistered_benefit_conclusion"],
            "Foreign full-scene WP12 evidence")
    return dict(schema=SCHEMA,pins=links,
                frozen_actual_retained_views=frozen_views(root,links),
                issue184_frozen54=dict(scope="historical_full_scene_full_spectrum",approved_cases=54,
                    matched_to_guarded_K5=False,full_report=proof(root,"old_combined"),
                    independent_hardware=proof(root,"independent"),
                    conclusion=full_scene["preregistered_benefit_conclusion"]),
                honghu_historical_short=honghu_short(root),honghu_reviewed_120=honghu_120(root),
                honghu_exact_support_checkpoint_scope=dict(physical_cursors=hh["physical_saved_cursors"],views=cursors,
                    history_22_weight_materialized=hh["history_22_weight_materialized"],
                    score_only_cursor_delta=True,protocol=hh["protocol"],support_seed=512,k=5),
                xuzhou_historical_protection_failures=xuzhou_protection(root,links),
                nonmatching_scopes=["short_vs_reviewed_two_60_step_paths","K1_vs_K5","scratch_vs_shared_transfer",
                                    "full_scene_frozen54_vs_guarded_K5","historical_vs_external_independent"],
                real_optimizer_updates=0,raw_gt_loaded=False,query_accessed=False,
                wp04_executed=False,independent_real_accuracy=None)
