"""Register and run an isolated, one-pass author/adapter development pairing.

All data are consumed through the existing authenticated development bank.
No MAT/GT/query reader is defined here. CPU only for this bounded pilot.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

from adapter import Adapter, UPSTREAM, first_minimum, load_author_model, objective, patches

BANK_CODE = "49bce8c5fa2d8b79c0e58b2a70693c7b897d4eb7"
RECEIPTS = {"xuzhou": "wp02-bank-xuzhou03.json", "whu-honghu": "wp02-bank-honghu03.json",
            "paviaU": "wp02-paviau-bank-nov18.json"}
SEEDS = (64, 512, 2048)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def git_clean(root, expected):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    if git("rev-parse", "HEAD") != expected or git("status", "--porcelain"):
        raise ValueError("Expected clean frozen checkout: " + str(root))


def load_bank(args, scene):
    from transfer import development_bank as banks, development_bank_custody as custody
    from transfer import paviau_development_bank as pavia
    runs = Path(args.runs_root).resolve()
    path = runs / RECEIPTS[scene]
    receipt = json.loads(path.read_text())
    if scene == "paviaU":
        opened, _, intake = pavia.open_publication(runs, {"path": str(path), "sha256": sha(path)})
    else:
        intake = custody.PROVIDERS[scene](runs)
        opened = banks.Bank(receipt, runs, intake.release)
    return opened, {"path": str(path), "sha256": sha(path)}


def geometry(bank):
    train, val = [bank.role_rows(role) for role in ("devtrain", "devval")]
    overlaps, separations = 0, []
    for a in train:
        for b in val:
            ay, ae, ax, az = a["box"]
            by, be, bx, bz = b["box"]
            overlaps += max(0, min(ae, be)-max(ay, by)) * max(0, min(az, bz)-max(ax, bx))
            separations.append(max(by-ae, ay-be, bx-az, ax-bz, 0))
    parents = sorted({r["parent_tile"] for r in train} & {r["parent_tile"] for r in val})
    return {"scene": bank.release.scene, "coverage": bank.release.coverage(),
            "cross_role_box_overlap_pixels": overlaps, "minimum_excluded_axis_gap_pixels": min(separations),
            "shared_parents": parents, "normalization_fit": "frozen training statistics",
            "context": "one role-local bank tile, no external halo or full-scene graph",
            "independent_confirmation_eligible": False,
            "reason": "Previously used development roles; no independent custodian confirmation release",
            "graph": "DSNet: none. Native comparators require tile-local Q/A and unchanged access."}


def protected(bank, receipt):
    files = {str(path): sha(path) for path in bank.root.iterdir() if path.is_file()}
    files[receipt["path"]] = receipt["sha256"]
    return files


def prepare(args):
    rows, protect = {}, {}
    for scene in RECEIPTS:
        bank, receipt = load_bank(args, scene)
        roster = [r["id"] for r in bank.role_rows("devtrain") if r["samples"]]
        n = len(roster)
        if not n:
            raise ValueError("Empty released training roster")
        rows[scene] = {"receipt": receipt, "pin": bank.pin, "geometry": geometry(bank),
                       "train_roster": roster, "events": sorted(set((0, (n+1)//2, n))),
                       "updates_per_arm_seed": n}
        protect.update(protected(bank, receipt))
    value = {"schema": "paper1_c_one_pass_author_adapter_v1", "code": args.expected_commit,
             "bank_code": BANK_CODE, "author_code": UPSTREAM, "seeds": list(SEEDS), "scenes": rows,
             "maximum_updates": 2 * len(SEEDS) * sum(r["updates_per_arm_seed"] for r in rows.values()),
             "purpose": "numerical adaptation equivalence and development learning only",
             "device": "cpu", "threads": 1, "optimizer": "Adam lr1e-4, no scheduler, no clipping",
             "loss": "original unclamped CE+SAD", "selection": "first minimum devval CE NLL",
             "patch": "7x7 tile-local replicate", "estimated_minutes": [5, 30],
             "training_gate": "all engineering tests passed; register before real updates",
             "query_reads": 0, "raw_gt_reads": 0, "protected_inputs": protect}
    write_new(args.output, value)
    print(json.dumps({"registration": str(args.output), "sha256": sha(args.output),
                      "maximum_updates": value["maximum_updates"],
                      "scenes": {n: {k: r[k] for k in ("updates_per_arm_seed", "events")} for n, r in rows.items()}}))


def batch(bank, row):
    _, _, view, context = bank.materialize(row, {"mode": "none"})
    obs = view.observation
    y0, _, x0, _ = row["box"]
    width = bank.release.shape[1]
    points = [(int(i)//width-y0, int(i)%width-x0) for i, _ in row["samples"]]
    x = torch.from_numpy(obs.spectral.copy())
    actual = patches(x, points)
    # Independent coordinate-clamping reference: no F.pad or unfold.
    reference = torch.stack([torch.stack([
        torch.stack([x[min(max(y+dy, 0), x.shape[0]-1), min(max(z+dx, 0), x.shape[1]-1)]
                     for dx in range(-3, 4)]) for dy in range(-3, 4)]).permute(2, 0, 1)
        for y, z in points])
    if not torch.equal(actual, reference):
        raise ValueError("Patch extraction differs from direct local coordinate reference")
    labels = torch.tensor([label-1 for _, label in row["samples"]], dtype=torch.long)
    return actual, labels, context


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()+str(array.dtype).encode()+str(array.shape).encode()+array.tobytes())
    return h.hexdigest()


def reference_objective(outputs, x, y):
    nonlinear, linear, pred = outputs
    count = linear.shape[1]//2
    reconstructed = nonlinear + (linear[:, :count] + linear[:, count:count*2])
    sad = torch.mean(torch.acos(torch.sum(x * reconstructed, dim=1) / (
        torch.norm(reconstructed, dim=1, p=2) * torch.norm(x, dim=1, p=2))))
    value = F.cross_entropy(pred, y) + sad
    if not torch.isfinite(value):
        raise FloatingPointError("Unchanged author objective failed")
    return value


@torch.no_grad()
def evaluate(model, data, classes):
    model.eval()
    before = state_hash(model)
    matrix = torch.zeros(classes, classes, dtype=torch.long)
    nll = 0.
    predicted = []
    for x, y, _ in data:
        logits = model(x)[2]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite author logits")
        nll += float(F.cross_entropy(logits, y, reduction="sum"))
        pred = logits.argmax(1)
        matrix += torch.bincount(y*classes+pred, minlength=classes**2).reshape(classes, classes)
        predicted.extend(pred.tolist())
    support = matrix.sum(1)
    if int(support.min()) == 0 or state_hash(model) != before:
        raise ValueError("Missing class or evaluation mutated state")
    recall = matrix.diag().double()/support
    n = int(support.sum())
    return {"nll": nll/n, "oa": float(matrix.diag().sum()/n), "aa": float(recall.mean()),
            "recall": recall.tolist(), "support": support.tolist(), "confusion": matrix.tolist(),
            "prediction_sha256": hashlib.sha256(np.asarray(predicted, dtype="<i8").tobytes()).hexdigest()}


def execute_pair(args, reg, scene, seed, folder):
    bank, receipt = load_bank(args, scene)
    row = reg["scenes"][scene]
    if receipt != row["receipt"] or bank.pin != row["pin"] or geometry(bank) != row["geometry"]:
        raise ValueError("Registered observation or geometry changed")
    started = time.perf_counter()
    train = {r["id"]: batch(bank, r) for r in bank.role_rows("devtrain") if r["samples"]}
    validation = [batch(bank, r) for r in bank.role_rows("devval") if r["samples"]]
    materialize_seconds = time.perf_counter()-started
    classes, bands = bank.release.classes, bank.release.shape[-1]
    author = load_author_model(args.author_root)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    direct = author.DSNet(bands, classes, 7, "conv2d_unmix")
    adapted = Adapter(author, bands, classes)
    adapted.model.load_state_dict(copy.deepcopy(direct.state_dict()), strict=True)
    construction_seconds = time.perf_counter()-t0
    if state_hash(direct) != state_hash(adapted.model):
        raise ValueError("Initial states differ")
    opts = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in (direct, adapted)]
    trajectory, times, records = [], [0., 0.], [[], []]
    states = [None, None]
    order = list(np.random.default_rng(seed).permutation(row["train_roster"]))
    total_updates = [0, 0]
    for step in range(len(order)+1):
        if step in row["events"]:
            metrics = []
            for arm, model in enumerate((direct, adapted)):
                t0 = time.perf_counter()
                result = evaluate(model, validation, classes)
                times[arm] += time.perf_counter()-t0
                records[arm].append([step, result["nll"]])
                if first_minimum(records[arm]) == step:
                    states[arm] = copy.deepcopy(model.state_dict())
                metrics.append(result)
            if metrics[0] != metrics[1]:
                raise ValueError("Validation, probabilities or selection differ")
            trajectory.append({"step": step, "metrics": metrics[0], "equal_arms": True})
        if step == len(order):
            break
        x, y, _ = train[int(order[step])]
        losses = []
        for arm, (model, opt) in enumerate(zip((direct, adapted), opts)):
            t0 = time.perf_counter()
            model.train()
            opt.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = (reference_objective if arm == 0 else objective)(outputs, x, y)
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Missing or nonfinite author gradient")
            opt.step()
            times[arm] += time.perf_counter()-t0
            losses.append(float(loss.detach()))
            total_updates[arm] += 1
        if losses[0] != losses[1] or state_hash(direct) != state_hash(adapted.model):
            raise ValueError("Training trajectories differ")
    direct.load_state_dict(states[0])
    adapted.load_state_dict(states[1])
    selected = evaluate(direct, validation, classes)
    if selected != evaluate(adapted, validation, classes):
        raise ValueError("Selected readouts differ")
    checkpoint = folder / "adapter-selected.pt"
    torch.save({"state": adapted.state_dict(), "selected_step": first_minimum(records[1])}, checkpoint)
    restored = Adapter(author, bands, classes)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)["state"])
    if selected != evaluate(restored, validation, classes):
        raise ValueError("Selected checkpoint readout differs after reload")
    return {"scene": scene, "seed": seed, "status": "PASSED", "updates": total_updates,
            "train_label_exposure_per_arm": sum(len(b[1]) for b in train.values()),
            "classes": classes, "bands": bands, "parameter_count": sum(p.numel() for p in direct.parameters()),
            "initialization": "from scratch, identical author weights and BN state", "graph": None,
            "selected_step": first_minimum(records[0]), "selected_metrics": selected, "trajectory": trajectory,
            "source_and_adapter_exact_at_every_update": True, "checkpoint_sha256": sha(checkpoint),
            "materialize_and_patch_seconds": materialize_seconds, "construction_seconds": construction_seconds,
            "arm_train_validation_seconds": times, "elapsed_seconds": time.perf_counter()-started,
            "limitations": "One traversal, development readout only; no accuracy benefit or convergence claim"}


def run(args):
    registration = Path(args.registration)
    if sha(registration) != args.registration_sha256:
        raise ValueError("Registration hash differs")
    reg = json.loads(registration.read_text())
    if reg["code"] != args.expected_commit or reg["seeds"] != list(SEEDS) or reg["device"] != "cpu":
        raise ValueError("Frozen execution contract differs")
    for path, expected in reg["protected_inputs"].items():
        if sha(path) != expected:
            raise ValueError("Protected input changed")
    root = Path(args.output)
    root.mkdir(exist_ok=False)
    results = []
    for scene in RECEIPTS:
        for seed in SEEDS:
            folder = root/f"{scene}-{seed}"
            folder.mkdir()
            try:
                row = execute_pair(args, reg, scene, seed, folder)
            except Exception as error:
                row = {"scene": scene, "seed": seed, "status": "FAILED", "error": str(error),
                       "traceback": traceback.format_exc(), "no_automatic_retry": True}
            write_new(folder/"result.json", row)
            results.append(row)
            print(json.dumps({k: row.get(k) for k in ("scene", "seed", "status", "updates", "elapsed_seconds", "error")}), flush=True)
    unchanged = all(sha(path) == expected for path, expected in reg["protected_inputs"].items())
    value = {"registration_sha256": args.registration_sha256, "code": args.expected_commit,
             "author": UPSTREAM, "bank_code": BANK_CODE, "results": results,
             "all_passed": all(r["status"] == "PASSED" for r in results), "inputs_unchanged": unchanged,
             "query_reads": 0, "raw_gt_reads": 0, "independent_confirmation_reads": 0,
             "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    write_new(root/"summary.json", value)
    if not unchanged or not value["all_passed"]:
        raise RuntimeError("Pilot has retained failed cases")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "run"))
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--bank-code", required=True)
    p.add_argument("--author-root", required=True)
    p.add_argument("--runs-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--registration")
    p.add_argument("--registration-sha256")
    args = p.parse_args()
    if sys.platform != "linux" or subprocess.check_output(["systemd-detect-virt"], text=True).strip() != "lxc":
        raise RuntimeError("Run only in authorized LXC")
    git_clean(Path(__file__).resolve().parents[2], args.expected_commit)
    git_clean(args.bank_code, BANK_CODE)
    # This is a pinned external execution dependency, not a source merge.
    sys.path.insert(0, str(Path(args.bank_code).resolve()))
    load_author_model(args.author_root)
    if torch.cuda.is_initialized() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This bounded CPU pilot requires CUDA_VISIBLE_DEVICES='' before import")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    runs = Path(args.runs_root).resolve()
    output = Path(args.output).resolve()
    if not output.is_relative_to(runs) or output == runs:
        raise ValueError("Output must stay outside checkout in authorized runs")
    if args.mode == "prepare":
        prepare(args)
    else:
        if not args.registration or not args.registration_sha256:
            raise ValueError("Frozen registration and digest required")
        run(args)
    git_clean(Path(__file__).resolve().parents[2], args.expected_commit)
    git_clean(args.bank_code, BANK_CODE)


if __name__ == "__main__":
    main()
