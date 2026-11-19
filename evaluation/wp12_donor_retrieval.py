"""Admit and recompute sealed WP06/WP08 donor evidence without model updates."""

from collections import Counter
import csv
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.classification_metrics import classification_metrics
from scripts.run_frozen_band_region import _read, _write, check, pin
from transfer.donor_retrieval_campaign import CASES, PHASES, registration, verify_state
from transfer.donor_retrieval_graph import ARMS, BATCH, PREFIX, RANDOM_SEED, UPDATES
from transfer.donor_retrieval_training import EVAL_DRAWS, HEADS


def verify_evaluation(value, roster, writer, *, case, arm, condition):
    labels = value["labels"]
    negatives = value["null_labels"]
    check(len(labels)==len(negatives)==len(roster)*4*EVAL_DRAWS and Counter(labels)==Counter(negatives)
          and Counter(labels)[0]==3*Counter(labels)[1],"Unexpected independent evaluation denominator")
    check([row["id"] for row in value["per_graph"]]==roster and
          sum(r["donor_hits"] for r in value["per_graph"])==value["retrieval"]["hits"]
          and value["retrieval"]["total"]==len(labels),"Donor retrieval exposure changed")
    for head in HEADS:
        probabilities = value["probabilities"][head]
        predictions = [max(range(2), key=p.__getitem__) for p in probabilities]
        for truth_name, truth in (("heads",labels),("null_heads",negatives)):
            metrics = classification_metrics(truth,predictions,[0,1])
            check(all(metric==value[truth_name][head][k] for k,metric in metrics.items()),"WP12 classification evidence differs")
        for point,(y,n,p) in enumerate(zip(labels,negatives,probabilities,strict=True)):
            check(len(p)==2 and all(0.<=v<=1. for v in p) and abs(sum(p)-1.)<2e-6,"Probability simplex violated")
            writer.writerow((case,arm,condition,head,point,y,n,p[0],p[1]))
    return len(labels)*len(HEADS)


def verify_matching(record):
    same, independent = 0, 0
    for info in record["independent_matching"].values():
        plan = info["plan"]
        masks = info["masks"]
        weak = masks["low_weight"]
        natural = masks[f"random:{RANDOM_SEED}"]
        conditioned = masks[f"conditional_random:{RANDOM_SEED}"]
        check(info["feasible"]==24 and info["independent_alternatives"]==22 and weak!=conditioned
              and info["natural_equals_weak"]==(natural==weak)
              and info["weak_differs_conditioned"] and plan["separate_weak"],"Independent matched arms collapsed")
        for name in ("important","low_weight",f"random:{RANDOM_SEED}",f"conditional_random:{RANDOM_SEED}"):
            original = info["matched_support"]["none"]
            actual = info["matched_support"][name]
            check(original["nnz"]-actual["nnz"]==len(masks[name]),"Physical deletion count differs")
        same += natural==weak
        independent += conditioned!=weak
    return same, independent


def outcome(case):
    arms = case["evaluation"]
    chosen = {name:value["chosen"] for name,value in arms.items()}
    def value(name, *, head="logit_fused", section="heads", metric="aa"):
        return chosen[name][section][head][metric]
    retrieval = lambda name: chosen[name]["retrieval"]["hits"]/chosen[name]["retrieval"]["total"]
    matches = {name:{"retrieval":retrieval(name), "fused_oa":value(name,metric="oa"),"fused_aa":value(name),
                     "graph_aa":value(name,head="logit_graph"), "null_fused_aa":value(name,section="null_heads"),
                     "minority_recall":chosen[name]["heads"]["logit_fused"]["per_class"][1]["recall"]}
               for name in ARMS}
    oracle = value("oracle")>=.85 and value("local")<=.65
    learn = (retrieval("learned")>=.75 and value("learned")-value("random")>=.10
             and value("learned",section="null_heads")<=.65)
    return dict(oracle=oracle,learned=learn,arms=matches)


def publish(root,out,code,expected):
    scope = registration(root,code,expected)["sha256"]
    check(not out.exists(),"Existing donor WP12 output")
    checks, pinmap, scores = {}, {}, {}
    updates, rows, natural_equals_weak, independent = 0, 0, 0, 0
    with TemporaryDirectory(prefix=".donor-wp12-",dir=out.parent) as work:
        staging = Path(work)/"evidence"
        staging.mkdir()
        with (staging/"predictions.csv").open("w",encoding="utf-8",newline="") as f:
            w = csv.writer(f)
            w.writerow(("case","arm","condition","head","point","truth","null_truth","p0","p1"))
            for name in CASES:
                base = root/name
                sealed = pin(base/"receipt.json")
                entry = _read(sealed)
                check(entry["scope"]==scope and entry["case"]==name and set(entry["arms"])==set(ARMS)
                      and entry["real_updates"]==0 and entry["raw_gt_loaded"] is entry["query_accessed"] is False,
                      "Donor evidence belongs to a foreign run")
                same,count = verify_matching(entry)
                natural_equals_weak += same
                independent += count
                checks[name] = entry
                pinmap[name] = sealed["sha256"]
                for arm,detail in entry["arms"].items():
                    scientific = {}
                    for phase,wanted in zip(PHASES,(UPDATES,PREFIX,UPDATES-PREFIX,0),strict=True):
                        path = base/arm/phase
                        evidence = _read(pin(path/"receipt.json",detail["phases"][phase]))
                        check(evidence["performed"]==wanted and evidence["phase"]==phase and evidence["scope"]==scope
                              and evidence["roster"]==entry["roster"] and evidence["parent"]==detail["parent"],
                              "Phase budget or input lineage differs")
                        saved = verify_state(path/"state.pt",evidence["checkpoint"],evidence["state"])
                        check(saved["cursor"]==evidence["cursor"] and len(saved["trajectory"])==saved["cursor"],
                              "Actual update cursor differs")
                        scientific[phase] = saved["state_sha256"]
                        updates += wanted
                    check(all(scientific[phase]==detail["state"] for phase in ("full","resume","completed")),
                          "Failed independent staged state replay")
                    for condition,value in entry["evaluation"][arm].items():
                        rows += verify_evaluation(value,entry["roster"]["confirmation"],w,case=name,arm=arm,condition=condition)
                scores[name] = outcome(entry)
        passed_oracle = sum(r["oracle"] for r in scores.values())
        passed_learning = sum(r["learned"] for r in scores.values())
        check(updates==len(CASES)*len(ARMS)*2*UPDATES,"Bad campaign update coverage")
        _write(staging/"cases.json",checks)
        _write(staging/"summary.json",dict(cases=scores, known_graph_supported=passed_oracle>=9,
               learned_remote_supported=passed_learning>=9, oracle_pairs=passed_oracle, learned_pairs=passed_learning,
               matching_contexts=independent, natural_equals_weak=natural_equals_weak))
        sealed = {item.name:pin(item)["sha256"] for item in staging.iterdir()}
        _write(staging/"receipt.json",dict(status="accepted_controlled_synthetic_only",scope=scope,
               producer=code["commit"], cases=pinmap, fits=len(CASES)*len(ARMS), prediction_rows=rows,
               synthetic_updates=updates, completed_reentry_updates=0, real_updates=0, query_accessed=False,
               raw_gt_loaded=False, independent_matched_masks=independent, files=sealed))
        os.rename(staging,out)
    return pin(out/"receipt.json")
