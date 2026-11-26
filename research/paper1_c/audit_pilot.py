"""Zero-update independent devval reconstruction from selected author weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from adapter import load_author_model
from pilot import BANK_CODE, RECEIPTS, SEEDS, git_clean, load_bank, sha, write_new


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--bank-code", required=True)
    p.add_argument("--author-root", required=True)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--execution", required=True)
    p.add_argument("--registration", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    git_clean(Path(__file__).resolve().parents[2], args.expected_commit)
    git_clean(args.bank_code, BANK_CODE)
    sys.path.insert(0, str(Path(args.bank_code).resolve()))
    torch.set_num_threads(1)
    author = load_author_model(args.author_root)
    reg = json.loads(Path(args.registration).read_text())
    folder = Path(args.execution)
    summary = json.loads((folder/"summary.json").read_text())
    if summary["registration_sha256"] != sha(args.registration):
        raise ValueError("Independent audit registration identity differs")
    protected_before = {path: sha(path) for path in reg["protected_inputs"]}
    if protected_before != reg["protected_inputs"]:
        raise ValueError("Protected development inputs differ")
    audited = []
    for scene in RECEIPTS:
        bank, receipt = load_bank(args, scene)
        if receipt != reg["scenes"][scene]["receipt"]:
            raise ValueError("Wrong scene release")
        for seed in SEEDS:
            root = folder/f"{scene}-{seed}"
            result = json.loads((root/"result.json").read_text())
            if result["status"] != "PASSED":
                raise ValueError("All failures must remain visible, not audited as passes")
            n = reg["scenes"][scene]["updates_per_arm_seed"]
            journal = [json.loads(line) for line in (root/"updates.jsonl").read_text().splitlines()]
            if [(r["arm"], r["step"]) for r in journal] != [(a, s) for s in range(1, n+1) for a in (0, 1)]:
                raise ValueError("Actual update journal differs from frozen exposure")
            if result["updates"] != [n, n] or journal[-1]["completed_updates"] != [n, n]:
                raise ValueError("Completed update count differs")
            trajectory = result["trajectory"]
            if [r["step"] for r in trajectory] != reg["scenes"][scene]["events"]:
                raise ValueError("Wrong selection events")
            selected = min(trajectory, key=lambda r: r["metrics"]["nll"])["step"]
            if selected != result["selected_step"]:
                raise ValueError("First minimum selection differs")
            checkpoint = root/"adapter-selected.pt"
            if sha(checkpoint) != result["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint changed")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["selected_step"] != selected:
                raise ValueError("Selected weight metadata differs")
            bands, classes = bank.release.shape[-1], bank.release.classes
            model = author.DSNet(bands, classes, 7, "conv2d_unmix").eval()
            model.load_state_dict({name.removeprefix("model."): value for name, value in saved["state"].items()}, strict=True)
            confusion = np.zeros((classes, classes), dtype=np.int64)
            predictions, nll = [], 0.
            for row in bank.role_rows("devval"):
                if not row["samples"]:
                    continue
                _, _, view, _ = bank.materialize(row, {"mode": "none"})
                array = view.observation.spectral
                h, w, _ = array.shape
                top, _, left, _ = row["box"]
                width = bank.release.shape[1]
                batch, labels = [], []
                for pixel, label in row["samples"]:
                    y, x = pixel//width-top, pixel%width-left
                    yy = np.clip(np.arange(y-3, y+4), 0, h-1)
                    xx = np.clip(np.arange(x-3, x+4), 0, w-1)
                    batch.append(array[yy[:, None], xx[None, :]].transpose(2, 0, 1))
                    labels.append(label-1)
                with torch.no_grad():
                    logits = model(torch.from_numpy(np.stack(batch)))[2].cpu().double().numpy()
                if not np.isfinite(logits).all():
                    raise ValueError("Nonfinite independent logits")
                labels = np.asarray(labels, dtype=np.int64)
                prediction = logits.argmax(1)
                predictions.extend(prediction.tolist())
                np.add.at(confusion, (labels, prediction), 1)
                shifted = logits-logits.max(1, keepdims=True)
                nll += float((np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(labels)), labels]).sum())
            expected = result["selected_metrics"]
            if confusion.tolist() != expected["confusion"]:
                raise ValueError("Independent confusion differs")
            prediction_hash = hashlib.sha256(np.asarray(predictions, dtype="<i8").tobytes()).hexdigest()
            if prediction_hash != expected["prediction_sha256"]:
                raise ValueError("Independent selected predictions differ")
            support = confusion.sum(1)
            recall = confusion.diagonal()/support
            oa = float(confusion.trace()/support.sum())
            aa = float(recall.mean())
            if (abs(oa-expected["oa"]) > 1e-7 or abs(aa-expected["aa"]) > 1e-12
                    or abs(nll/support.sum()-expected["nll"]) > 2e-6
                    or not np.array_equal(support, expected["support"])):
                raise ValueError("Independent metrics differ")
            expected_exposure = sum(reg["scenes"][scene]["geometry"]["coverage"]["devtrain"]["released_per_class"])
            if result["train_label_exposure_per_arm"] != expected_exposure:
                raise ValueError("Released point exposure differs")
            audited.append({"scene": scene, "seed": seed, "updates": [n, n], "exposure": expected_exposure,
                            "selected_step": selected, "oa": oa, "aa": aa, "recall": recall.tolist(),
                            "support": support.tolist(), "confusion_cells": classes**2,
                            "prediction_sha256": prediction_hash, "numpy_nll": nll/int(support.sum()),
                            "parameter_count": sum(p.numel() for p in model.parameters())})
    unchanged = protected_before == {path: sha(path) for path in protected_before}
    if not unchanged:
        raise ValueError("Independent audit changed an input")
    write_new(args.output, {"status": "PASSED", "auditor_code": args.expected_commit,
              "producer_code": summary["code"], "registration_sha256": sha(args.registration),
              "cases": audited, "actual_updates_audited": sum(sum(r["updates"]) for r in audited),
              "new_updates": 0, "query_reads": 0, "raw_gt_reads": 0, "inputs_unchanged": unchanged,
              "protected_files": len(protected_before), "independent_confirmation": False})
    print(json.dumps({"cases": len(audited), "updates_audited": sum(sum(r["updates"]) for r in audited),
                      "output": args.output, "sha256": sha(args.output)}))


if __name__ == "__main__":
    main()
