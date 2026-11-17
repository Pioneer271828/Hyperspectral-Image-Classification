"""Cross-check all historic train/val confusion matrices with fixed TorchMetrics."""

import argparse
import ast
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.audit_oss_a_history import CAMPAIGNS, PHASES, digest, labels, read
from scripts.audit_oss_b_spectral import MANIFEST


PIN = "219b4af29637ea9a2c356ec550402a56241d192f5a98edebd35897504db7a80c"


def oracle(path):
    if digest(path)!=PIN:
        raise ValueError("Fixed TorchMetrics source bytes required")
    source = ast.parse(path.read_bytes())
    name = "_multiclass_confusion_matrix_update"
    function = next(node for node in source.body if isinstance(node,ast.FunctionDef) and node.name==name)
    namespace = dict(torch=torch,Tensor=torch.Tensor,_bincount=torch.bincount)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),"exec"),namespace)
    return namespace[name]


def audit(runs,source):
    matrix = oracle(source)
    truth = labels(runs,MANIFEST)
    counts = {}
    for campaign,(expected,producer) in CAMPAIGNS.items():
        cases = sorted((runs/campaign/"cases").iterdir())
        if len(cases)!=expected:
            raise ValueError("Frozen case roster differs")
        checked = 0
        for case in cases:
            result = read(case/"result.json")
            if result["start_code"]["commit"]!=producer or result["test_scoring"] is not False:
                raise ValueError("Wrong frozen study identity")
            path = case/"selected-probabilities.pt"
            if digest(path)!=result["artifacts"][path.name]:
                raise ValueError("Changed frozen predictions")
            probabilities = torch.load(path,map_location="cpu",mmap=True,weights_only=True)
            recorded = result["selected_validation"]["metrics"]
            for scene,roles in probabilities.items():
                if set(roles)!=set(PHASES):
                    raise ValueError("Unexpected role")
                for role,heads in roles.items():
                    labels_ = truth[scene][role]
                    target = labels_[labels_>0]-1
                    for head,scores in heads.items():
                        observed = matrix(scores.argmax(1),target,scores.shape[1])
                        if observed.tolist()!=recorded[scene][role][head]["confusion"]:
                            raise ValueError(f"Confusion mismatch for {case.name}/{scene}/{role}/{head}")
                        checked += 1
        counts[campaign]=checked
    return dict(confusion_matrices=counts,source_sha256=PIN,heldout_reads=0,optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,required=True)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    result = audit(args.runs,args.source)
    with args.out.open("x") as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=="__main__":
    main()
