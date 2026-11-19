"""Read-only audit of sealed checkpoints and matched supports, without solving again."""

from collections import Counter
from itertools import product

from evaluation.graph_edge_intervention import delete_support
from evaluation.registered_matched_route_audit import signature
from research_runtime import tensor_hash
from scripts.run_frozen_band_region import _read, _write, check, pin
from transfer.band_graph_campaign import admitted
from transfer.band_graph_control import BINS, controlled_state, observation_id
from transfer.band_graph_fixture import LAYOUTS, SEEDS, TASKS, VIEWS
from transfer.native_scale_training import _exact
from transfer.spatial_coverage_protocol import bank_view
from transfer.spatial_coverage_training import load


def audit(folder, output, expected_receipt):
    receipt=_read(pin(folder/"wp12/receipt.json",expected_receipt))
    check(receipt["status"]=="accepted_controlled_synthetic_only" and receipt["real_updates"]==0
          and receipt["raw_gt_loaded"] is receipt["query_accessed"] is False,"Unexpected frozen WP12 declaration")
    details=Counter()
    for task,layout,seed in product(TASKS,LAYOUTS,SEEDS):
        case=f"{task}-{layout}-{seed}"
        path=folder/case
        record=_read(pin(path/"receipt.json",receipt["cases"][case]))
        check((path/"bank/bank.json").is_file(),"Audit requires complete immutable synthetic bank")
        bank,witness=admitted(path,task,layout,seed)
        check(witness==record["truth"],"Controlled truth or role differs")
        for band in VIEWS:
            result=_read(pin(path/band/"receipt.json",record["views"][band]))
            fits=result["fits"]
            expected_nnzs={}
            for name,matched in result["matches"].items():
                role,tid=name.split("/")
                source=bank.role_row(role,int(tid))
                actual,_,_=bank_view(bank,source,receipt["producer"],band)
                state,xy,geometry=controlled_state(actual,layout)
                check(observation_id(actual)==matched["view"] and geometry==matched["topology"],
                      "Actual source-band graph identity changed")
                control=signature(matched["baseline"]["removed"],9,xy.tolist(),BINS)
                details["graphs"]+=1
                for arm in fits:
                    if arm=="native":
                        continue
                    removed=([] if arm=="fixed_none" else matched["baseline"]["removed"] if arm=="important"
                             else matched["controls"][arm]["removed"])
                    a=delete_support(state.A0,removed)
                    if arm!="fixed_none":
                        check(signature(removed,9,xy.tolist(),BINS)==control,
                              "Independent degree/distance count differs")
                        if layout=="vertical":
                            check({(e%9)*9+e//9 for e in removed}==set(removed),"Reciprocal direction missing")
                    expected_nnzs[name,arm]=(state.Q._nnz(),a._nnz(),tensor_hash(state.Q),tensor_hash(a))
                    details["mask_checks"]+=1
                details["duplicate_weak_random"]+=matched["controls"]["low_weight"]["removed"]==matched["controls"]["random:7719"]["removed"]
            for arm,sha in fits.items():
                name=arm.replace(":","-")
                directory=path/band/name
                candidate=_read(pin(directory/"receipt.json",sha))
                full,resumed,complete=(load(directory/f"{variant}.pt") for variant in ("full","resume","completed"))
                check(_exact(full,resumed) and _exact(full,complete),"End-state restart or no-op recovery differs")
                step=min(full["validation"],key=lambda values:values["fused_nll"])["step"]
                check(step==candidate["selected"] and full["cursor"]==12,"First minimum or update clock differs")
                for trace in candidate["endpoint"]["traces"]:
                    if arm!="native":
                        expected=expected_nnzs[f'devval/{trace["tile"]}',arm]
                        check((trace["q_nnz"],trace["nnz"],trace["q"],trace["a"])==expected,"Graph used at endpoint differs")
                        details["inferred_graph_checks"]+=1
                details["recovery_checks"]+=1
            for arm,prediction in result["frozen"].items():
                details["frozen_heads"]+=3
                for trace in prediction["traces"]:
                    expected=expected_nnzs[f'devval/{trace["tile"]}',arm]
                    check((trace["q_nnz"],trace["nnz"],trace["q"],trace["a"])==expected,"Frozen intervention graph differs")
    check(details["graphs"]==324 and details["recovery_checks"]==receipt["fit_cases"],
          "Incomplete independent finite matrix")
    return _write(output,dict(status="passed",scope=receipt["scope"],counts=dict(details),wp12=expected_receipt,
                              optimizer_updates=0,raw_gt_loaded=False,query_accessed=False))
