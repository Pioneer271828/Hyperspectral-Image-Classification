"""Regression for the formal class floor on physical external-format Bank publications."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from evaluation.frozen_training_stages import require
from evaluation.wp12_paired_head_review import pair_scores
from evaluation.wp12_release_qualification import classification_status
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_frozen_band_region import _write
from transfer import native_independent_release as independent
from transfer.native_external_paired import SCENES
from transfer.native_external_paired_fixture import release_bank


def exercise(root,out,producer):
    target = {"xuzhou":5,"honghu":15}
    test = (("Below",19,True,True,True),("AtNoGain",20,True,True,False),
            ("AtPass",20,True,False,True),("Above",21,True,True,True),
            ("CostOnly",20,False,True,False))
    outcomes = {}
    for kind,scene in SCENES.items():
        place = out/kind
        place.mkdir(parents=True)
        history,_,release = release_bank(place/"prior",kind,1,64,"h")
        for name,n,accuracy,cost,gain in test:
            counts = {cls:(n if cls==target[kind] else 20) for cls in range(1,release.classes+1)}
            entry,roles,_ = release_bank(place/name,kind,5,512,"a",evaluation_counts=counts)
            manifest = _write(place/(name+"-manifest.json"),dict(schema=independent.MANIFEST,candidate=entry,
                              roles=roles,minimum_buffer=8,history=[history]))
            permission = _write(place/(name+"-authority.json"),dict(schema=independent.AUTHORITY,
                approved_manifest_sha256=manifest["sha256"],historical_bank_sha256=[history["pin"]["sha256"]],
                source_mode="synthetic",uses=dict(independent_accuracy=accuracy,cost=cost,frozen_diagnostics=False)))
            bank,intake,assigned = independent.open_independent(root,manifest,permission,test_history=[])
            earned = intake["uses"]
            require(earned["independent_accuracy"]["eligible"]==accuracy and earned["cost"]["eligible"]==cost
                    and intake["role_reports"]["evaluation"]["per_class"]==list(counts.values())
                    and bank.reads==0,"Selective role gate or physical support changed")
            metrics = None
            if accuracy:
                truth = []
                for member in roles["evaluation"]:
                    _,_,points = independent.materialize_reviewed(bank,intake,assigned,"evaluation",member,producer,
                                                                    "full",use="independent_accuracy")
                    truth.extend(y for _,y in points.points)
                before = list(truth)
                before[truth.index(target[kind])] = 1
                after = truth if gain else before
                metrics = pair_scores(truth,before,after,release.classes,scene,head_kind="fused",guard_population=20)
            if cost:
                for member in roles["evaluation"]:
                    _,labels,points = independent.materialize_reviewed(bank,intake,assigned,"evaluation",member,producer,
                                                                         "full",use="cost")
                    require(labels is points is None,"Cost-only materialization reached labels")
            actual = classification_status(metrics,accuracy_allowed=accuracy,cost_allowed=cost,real=False)
            # Verify the real qualification arithmetic with the fixture's measured denominator and controlled scores.
            counterfactual = classification_status(metrics,accuracy_allowed=accuracy,cost_allowed=cost,real=True)
            require(actual["raw_metrics"]==accuracy and actual["protection_evaluable"]==(accuracy and n>=20)
                    and actual["qualified"] is False and actual["cost_eligible"]==cost
                    and counterfactual["qualified"]==(accuracy and gain and n>=20),"Mixed classification gate states")
            outcomes[f"{kind}/{name}"] = dict(manifest=manifest,permission=permission,bank=bank.pin,
                evaluation_points=sum(counts.values()),focused_class_count=n,
                actual_test_status=actual,formula_at_real_floor=counterfactual,source="external_format_test_only")
    return _write(out/"classification-floor.json",dict(schema="wp12_formal_class_floor_regression_v1",producer=producer,
                  cases=outcomes,new_real_labels=False,actual_real_accuracy_qualified=False,optimizer_updates=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--expected-commit",required=True)
    args = parser.parse_args()
    root,out = args.runs.resolve(strict=True),external_directory(args.out)
    code = code_snapshot()
    require(root in out.parents and code["commit"]==args.expected_commit and not out.exists(),"Invalid fixture scope")
    require_frozen_code(code)
    result = exercise(root,out,code["commit"])
    require_frozen_code(code)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
