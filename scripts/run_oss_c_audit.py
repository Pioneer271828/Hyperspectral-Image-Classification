"""Standalone CPU audit of frozen metrics, stock pruning axes and PyG receipts."""

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from scripts.audit_oss_a_history import CAMPAIGNS, PREPARATION, PHASES, SCENES, digest, read
from scripts.audit_oss_b_spectral import MANIFEST
from transfer.oss_c_oracles import (HEADS, confusion, qualification, structural,
                                    torchmetrics_oracle, validate_indices)


def score_history(runs, torchmetrics_file=None):
    root = runs/PREPARATION
    manifest = root/"manifest.json"
    if digest(manifest) != MANIFEST:
        raise ValueError("Frozen historical manifest changed")
    entries = read(manifest)["datasets"]
    targets = {}
    for scene in SCENES:
        entry = entries[scene]
        path = root/(scene+"-prepared.pt")
        if path.is_symlink() or entry["prepared_file"]!=path.name or digest(path)!=entry["prepared_sha256"]:
            raise ValueError("Frozen labeled payload changed")
        item = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if any(key.startswith(("test_", "query_")) for key in item):
            raise ValueError("Held-out content is prohibited")
        targets[scene] = (item["class_num"],{})
        for role in PHASES:
            label = item[role+"_samples_gt"].long().flatten()
            ids = (label>0).nonzero().flatten()
            if len(ids) != int(item[role+"_label_mask"][:,0].sum()):
                raise ValueError("Historical role labels differ from mask")
            targets[scene][1][role] = (ids,label[ids])
    crosscheck = None if torchmetrics_file is None else torchmetrics_oracle(torchmetrics_file)
    checked = {}
    empty = []
    for study,(case_count,producer) in CAMPAIGNS.items():
        cases = sorted((runs/study/"cases").iterdir())
        if len(cases) != case_count:
            raise ValueError("Unexpected frozen case roster")
        checked[study] = 0
        for case in cases:
            result = read(case/"result.json")
            if result["start_code"]["commit"]!=producer or result["test_scoring"] is not False:
                raise ValueError("Wrong historical study identity")
            path = case/"selected-probabilities.pt"
            if path.is_symlink() or digest(path)!=result["artifacts"][path.name]:
                raise ValueError("Frozen prediction bytes changed")
            values = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if tuple(values) != SCENES:
                raise ValueError("Unexpected or reordered prediction scene roster")
            loss = result["settings"]["config"]["loss"]
            active = HEADS[:1] if loss["cnn_loss_weight"]==loss["graph_branch_loss_weight"]==0 else HEADS
            for scene, roles in values.items():
                if tuple(roles) != PHASES:
                    raise ValueError("Unexpected or reordered prediction roles")
                classes, known = targets[scene]
                for role, heads in roles.items():
                    ids, labels = known[role]
                    # The frozen producer serialized rows in ascending flat-index order.
                    # The prediction-file digest binds that ordering; the file carries no separate ID vector.
                    validate_indices(ids, torch.sort(ids).values)
                    observed = confusion(labels, heads, classes, role=role,active_heads=active,crosscheck=crosscheck)
                    for head, row in observed.items():
                        if row["confusion"]!=result["selected_validation"]["metrics"][scene][role][head]["confusion"]:
                            raise ValueError("Frozen confusion matrix differs")
                        if row["empty_classes"]:
                            empty.append([study,case.name,scene,role,head,row["empty_classes"]])
                        checked[study] += 1
    return dict(counts=checked, empty_classes=empty, heldout_reads=0, optimizer_updates=0,
                qualification=qualification("torchmetrics" if crosscheck else "project_confusion",
                    test_execution_passed=True,numeric_scope="CPU integer confusion only"))


def structural_audit(upstream_root):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = MultiDatasetVCASAHGCN({"scene":2,"other":3},norm_type="domain_bn").eval()
        witness = torch.rand(1,64,5,6)
    receipt = structural(model,witness,[2,95,126],upstream_root=upstream_root)
    receipt["qualification"] = qualification(receipt["backend"],test_execution_passed=True,
                                              numeric_scope="stock stage-1 tensor axes only")
    return receipt


def pyg_qualification(path):
    rows = {}
    for line in path.read_text().splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if "failed_parameter_tensors" in row:
            rows[row["device"]] = row
    if set(rows)!={"cpu","cuda:0"}:
        raise ValueError("CPU and strict CUDA paired evidence required")
    failure = {}
    for device,row in rows.items():
        if (row["candidate_recovery"]!="exact" or row["production_authorized"] is not False
                or row["training_trajectory_compatible"]!= (not row["failed_parameter_tensors"])):
            raise ValueError("Inconsistent engineering witness")
        failure.update({f"{device}/{name}":value for name,value in row["failed_parameter_tensors"].items()})
    return qualification("pyg_coo_sum",test_execution_passed=True,failed_tensors=failure,
                         numeric_scope="same-device native/candidate 12-update trajectory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    mode = parser.add_subparsers(dest="mode",required=True)
    score = mode.add_parser("history")
    score.add_argument("--runs",type=Path,required=True)
    score.add_argument("--torchmetrics-source",type=Path)
    shape = mode.add_parser("structure")
    shape.add_argument("--torch-pruning-root",type=Path)
    qualify = mode.add_parser("pyg")
    qualify.add_argument("--evidence",type=Path,required=True)
    args = parser.parse_args()
    if args.mode == "history":
        value = score_history(args.runs,args.torchmetrics_source)
    elif args.mode == "structure":
        value = structural_audit(args.torch_pruning_root)
    else:
        value = pyg_qualification(args.evidence)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2)
        stream.write("\n")
    print(json.dumps(value.get("qualification",value)))


if __name__=="__main__":
    main()
