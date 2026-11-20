"""Run the fixed source-swap boundary test on the predeclared blind release blocks."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from evaluation import wp06_causal_boundary as boundary, wp12_cross_backend as compare
from evaluation.frozen_training_stages import file_pin, json_pin, require
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_frozen_band_region import _write
from scripts.run_native_released_points import arithmetic
from transfer import native_independent_release as independent, wp_cross_backend as backend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--campaign",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--expected-commit",required=True)
    a=p.parse_args()
    start=perf_counter()
    root=a.runs.resolve(strict=True)
    out=external_directory(a.out)
    require(root in out.parents and root in a.campaign.resolve().parents and not out.exists(),"Wrong bounded output roots")
    code=code_snapshot()
    require(code["commit"]==a.expected_commit,"Wrong causal boundary source")
    require_frozen_code(code)
    marker=file_pin(a.campaign/"publication.json")
    publication=json_pin(root,marker)
    require(publication["producer"]==code["commit"] and publication["scope"]=="test_external_format_only"
            and publication["query_accessed"] is publication["raw_gt_loaded"] is False,
            "Boundary test only consumes complete engineering publications")
    results={}
    with arithmetic(root,None) as execution:
        for release,pin in sorted(publication["releases"].items()):
            bank,intake,roles,groups=compare.open_case(root,pin,test_history=[])
            group=groups.get(release+"-s64")
            if group is None:
                continue
            protocol=group["causal_design"]
            model=backend.load_export(root,group["models"]["original"]["checkpoint"])
            chosen=next(t for t in roles["evaluation"] if protocol["tiles"][str(t)]["regime"]=="new_path")
            actual,labels,targets=independent.materialize_reviewed(bank,intake,roles,"evaluation",chosen,
                                                                 code["commit"],"full",use="cost")
            require(labels is targets is None,"Boundary audit acquired confirmation labels")
            observed=boundary.native_boundary(model,actual,code["commit"],execution,protocol)
            pure=boundary.pure_boundary(model,bank.release.scene,protocol["layout"])
            results[release]=dict(native=observed,pure= pure,endpoint=group["arms"]["original"])
    require(results,"No predeclared source-swap cohort")
    require_frozen_code(code)
    receipt=_write(out,dict(schema="wp06_separated_graph_boundary_v1",source=code["commit"],
        publication=marker,rows=results,raw_gt_loaded=False,query_accessed=False,optimizer_updates=0,
        elapsed_seconds=perf_counter()-start))
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    main()
