"""Validate a frozen baseline in the remote training container; stdlib orchestration."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True, help="Full 40-character commit SHA")
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory outside checkout")
    parser.add_argument("--skip-cuda", action="store_true", help="Explicit CPU-only validation")
    args = parser.parse_args()
    if sys.platform != "linux" or not any(Path(p).exists() for p in ("/.dockerenv", "/run/.containerenv")):
        parser.error("Run inside the remote Linux training container via ssh rmvision")
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_commit):
        parser.error("--expected-commit must be a full lowercase SHA")
    if git("rev-parse", "HEAD") != args.expected_commit:
        parser.error("HEAD differs from requested commit")
    if git("status", "--porcelain", "--untracked-files=all"):
        parser.error("Checkout must be clean; synchronize and commit current changes first")
    out = args.out_dir.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error("Artifacts must be outside the code checkout")
    if sys.flags.optimize or os.environ.get("PYTHONOPTIMIZE"):
        parser.error("Assertions must be enabled for the existing smoke tests")
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "module": "baseline",
        "branch": git("branch", "--show-current"),
        "commit": args.expected_commit,
        "initial_dirty": False,
        "hostname": platform.node(),
        "code_directory": str(ROOT),
        "python": sys.version,
        "executable": sys.executable,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_requested": not args.skip_cuda,
        "status": "running",
        "checks": [],
    }
    manifest_path = out / "manifest.json"

    def save():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        inventory = json.loads((ROOT / "docs/migration/baseline-source.json").read_text(encoding="utf-8"))
        for entry in inventory["files"]:
            content = (ROOT / entry["path"]).read_bytes().replace(b"\r\n", b"\n")
            if hashlib.sha256(content).hexdigest() != entry["sha256_lf"]:
                raise RuntimeError(f"Frozen source changed: {entry['path']}")
        manifest["source_files_verified"] = len(inventory["files"])
        manifest["packages"] = {}
        for name in ("torch", "numpy", "scipy", "scikit-learn", "scikit-image", "spectral", "h5py", "Pillow"):
            manifest["packages"][name] = importlib.metadata.version(name)
        import torch

        manifest["cuda_runtime"] = torch.version.cuda
        manifest["cuda_available"] = torch.cuda.is_available()
        if manifest["cuda_available"]:
            manifest["gpu"] = torch.cuda.get_device_name(0)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1"}
        commands = [
            ("regression", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_interaction_modules.py", "-v"]),
            ("smoke_cpu", [sys.executable, "smoke_test.py", "--device", "cpu"]),
        ]
        if not args.skip_cuda:
            commands.append(("smoke_cuda", [sys.executable, "smoke_test.py", "--device", "cuda"]))
        for name, command in commands:
            check = {"name": name, "command": command, "status": "running", "log": f"{name}.log"}
            manifest["checks"].append(check)
            save()
            started = time.monotonic()
            with (out / check["log"]).open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            check.update(exit_code=result.returncode, duration_seconds=round(time.monotonic() - started, 3),
                         status="passed" if result.returncode == 0 else "failed")
            save()
            print(f"{name}: {check['status']} ({check['duration_seconds']}s)", flush=True)
            if result.returncode:
                raise RuntimeError(f"{name} failed; see {out / check['log']}")
        manifest["final_dirty"] = bool(git("status", "--porcelain", "--untracked-files=all"))
        if git("rev-parse", "HEAD") != args.expected_commit or manifest["final_dirty"]:
            raise RuntimeError("Code changed during validation")
        manifest["status"] = "passed"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        for check in manifest["checks"]:
            if check["status"] == "running":
                check["status"] = "failed"
        raise
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(f"Validation record: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
