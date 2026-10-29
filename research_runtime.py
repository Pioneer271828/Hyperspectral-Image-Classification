"""Run ownership and provenance for C0. No optimizer/checkpoint resume is implied."""

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from model.research_config import canonical_json, digest


ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def hash_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def external_directory(path):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError("Run outputs must be outside the source checkout")
    return path


def code_snapshot(root=ROOT):
    root = Path(root)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args]).decode("utf-8").strip()

    files = {}
    for name in git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0"):
        if name.endswith(".py") or (name.startswith("configs/") and name.endswith(".json")):
            files[name] = hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
            "source_hash": digest(files), "files": files}


def require_frozen_code(snapshot):
    current = code_snapshot()
    if current["dirty"] or current["commit"] != snapshot["commit"] or current["source_hash"] != snapshot["source_hash"]:
        raise RuntimeError("Code changed since planning; commit and synchronize before running")


def tensor_hash(value):
    import torch

    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().cpu()
    header = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    if tensor.layout != torch.strided:
        tensor = tensor.to_sparse_coo().coalesce()
        return digest({**header, "indices": tensor_hash(tensor.indices()), "values": tensor_hash(tensor.values())})
    result = hashlib.sha256(canonical_json(header).encode("utf-8"))
    result.update(memoryview(tensor.contiguous().numpy()).cast("B"))
    return result.hexdigest()


def prepared_fingerprints(datasets):
    result = {}
    data_fields = ("net_input", "wavelengths", "fwhm", "Q", "A")
    supervision = [f"{split}_{suffix}" for split in ("train", "val", "test")
                   for suffix in ("samples_gt", "gt_onehot", "label_mask")]
    for name, item in datasets.items():
        data = {key: tensor_hash(item.get(key)) for key in data_fields}
        data.update({key: item[key] for key in ("height", "width", "bands", "class_num", "superpixel_scale")})
        splits = {key: tensor_hash(item[key]) for key in supervision}
        result[name] = {"data_hash": digest(data), "split_hash": digest(splits),
                        "data_fields": data, "supervision_fields": splits, "seed": item.get("seed")}
    return result


def parameter_counts(model):
    groups = defaultdict(int)
    for name, parameter in model.named_parameters():
        group = ".".join(name.split(".")[:2]) if name.startswith("backbone.") else name.split(".")[0]
        groups[group] += parameter.numel()
    return {"total": sum(groups.values()), "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "groups": dict(groups), "research_parameters": {
                "M6": groups.get("backbone.interaction", 0), "M7": groups.get("backbone.edge_updater", 0)}}


class AlreadyCompleted(Exception):
    """The previous success has been verified, including current prepared data."""


class RunRecord:
    def __init__(self, root, config, code, resume=False):
        self.config, self.code, self.resume = config, code, resume
        identity = {"schema_version": 1, "config_hash": config.config_hash, "source_hash": code["source_hash"]}
        self.identity_hash = digest(identity)
        self.run_id = f"{config.name}-s{config.training.seed}-{self.identity_hash[:20]}"
        self.directory = external_directory(external_directory(root) / self.run_id)
        self.identity = {**identity, "identity_hash": self.identity_hash, "run_id": self.run_id}
        self.previous = None
        self.lock = self.directory / ".lock"

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(canonical_json({"pid": os.getpid(), "started_at": now()}))
        try:
            identity_path = self.directory / "identity.json"
            if identity_path.exists():
                if json.loads(identity_path.read_text(encoding="utf-8")) != self.identity:
                    raise ValueError("Run identity differs; refusing to reuse directory")
            else:
                write_json(identity_path, self.identity)
            previous = sorted((self.directory / "attempts").glob("*/manifest.json"))
            if previous and not self.resume:
                raise FileExistsError("Run exists; use --resume to verify success or retry failure")
            for path in reversed(previous):
                item = json.loads(path.read_text(encoding="utf-8"))
                if item["status"] == "completed":
                    self.previous = (path.parent, item)
                    break
            attempts = self.directory / "attempts"
            attempts.mkdir(exist_ok=True)
            numbers = [int(p.name) for p in attempts.iterdir() if p.is_dir()]
            self.attempt = attempts / f"{max(numbers, default=0) + 1:04d}"
            self.attempt.mkdir()
            self.save_path = self.attempt / "model.pt"
            self.metrics_path = self.attempt / "metrics.json"
            self.manifest = {"schema_version": 1, **self.identity, "code": self.code, "status": "running",
                             "started_at": now(), "attempt": self.attempt.name,
                             "budget": self.config.budget, "resume_requested": self.resume,
                             "command": list(sys.argv), "python": sys.version, "executable": sys.executable}
            write_json(self.attempt / "resolved_config.json", self.config.to_dict())
            self.save()
            return self
        except BaseException:
            self.lock.unlink()
            raise

    def save(self):
        write_json(self.attempt / "manifest.json", self.manifest)

    def record_data(self, datasets):
        self.manifest["prepared_data"] = prepared_fingerprints(datasets)
        count = len(datasets)
        steps = self.config.training.steps_per_epoch or count * 3
        self.manifest["budget"] = {**self.config.budget, "steps_per_epoch": steps,
                                   "optimizer_steps": (self.config.training.epochs + 1) * steps}
        self.save()
        if self.previous:
            directory, prior = self.previous
            if self.manifest["prepared_data"] != prior["prepared_data"]:
                raise ValueError("Data or split changed; cannot skip/reuse the previous success")
            for name, expected in prior["artifacts"].items():
                path = directory / name
                if not path.is_file() or hash_file(path) != expected:
                    raise ValueError(f"Completed artifact missing or changed: {path}")
            self.manifest.update(status="skipped", verified_attempt=directory.name)
            raise AlreadyCompleted(self.run_id)

    def record_model(self, model):
        import torch

        self.manifest["parameters"] = parameter_counts(model)
        self.manifest["framework"] = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                                      "device": str(next(model.parameters()).device)}
        self.save()

    def __exit__(self, kind, error, traceback):
        try:
            if kind is AlreadyCompleted:
                self.manifest["status"] = "skipped"
            elif kind is not None:
                self.manifest.update(status="failed", error=f"{kind.__name__}: {error}")
            else:
                required = [self.metrics_path, self.save_path, Path(str(self.save_path) + ".json")]
                if not all(p.is_file() for p in required) or "parameters" not in self.manifest or "prepared_data" not in self.manifest:
                    self.manifest.update(status="failed", error="Required training artifacts or provenance missing")
                    raise RuntimeError(self.manifest["error"])
                self.manifest.update(status="completed", artifacts={
                    str(p.relative_to(self.attempt)): hash_file(p) for p in sorted(self.attempt.rglob("*"))
                    if p.is_file() and p.name != "manifest.json"})
        finally:
            self.manifest["finished_at"] = now()
            self.save()
            self.lock.unlink()
        return kind is AlreadyCompleted
