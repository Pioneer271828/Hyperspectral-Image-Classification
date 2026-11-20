"""Immutable runtime identity for the support-head deployment and its archives."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import subprocess

from evaluation.frozen_training_stages import require
from research_runtime import ROOT

LEGACY_ANCHOR = "9102a9632e13c688d68739019936c571fc243612"
BRIDGE_FILE = "docs/architecture/wp12-support-head-runtime-bridge.json"
BRIDGE_SCHEMA = "wp12_support_head_runtime_bridge_v1"
# Neither reporting entry point is imported by the deployment worker.
REPORT_ONLY = {
    "README.md", "evaluation/wp12_classification_review.py", "scripts/ingest_wp12_evidence.py",
    "research/当前搭建状态与审查整改清单_20261102.md",
}


def git(root, *args):
    return ["git","-C",str(root.resolve()),*args]


def full_commit(value):
    return isinstance(value,str) and re.fullmatch("[0-9a-f]{40}",value) is not None


def ancestor(older, newer, *, root=ROOT):
    require(full_commit(older) and full_commit(newer),"Complete plan and execution Git SHAs required")
    status = subprocess.run(git(root,"merge-base","--is-ancestor",older,newer),
                            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
    require(status.returncode==0,"Historical deployment identity is outside the checked-out ancestry")


@lru_cache(maxsize=24)
def runtime_sha256(commit, *, root=ROOT):
    require(full_commit(commit),"A complete runtime Git SHA is required")
    tree = subprocess.check_output(git(root,"ls-tree","--full-tree","-rz",commit))
    members = []
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        _,filename = entry.split(b"\t",1)
        path = filename.decode("utf-8")
        if path not in REPORT_ONLY and not path.startswith(("docs/","tests/","research/reports/")):
            members.append(entry)
    require(members,"Empty support deployment runtime tree")
    return hashlib.sha256(b"\0".join(members)).hexdigest()


def compatibility(plan_producer, execution_producer, *, root=ROOT):
    """Permit equal runtime trees or a single explicitly reviewed predecessor bridge."""
    ancestor(plan_producer,execution_producer,root=root)
    plan_runtime = runtime_sha256(plan_producer,root=root)
    execution_runtime = runtime_sha256(execution_producer,root=root)
    mode = "identical_runtime"
    if plan_runtime != execution_runtime:
        additions = subprocess.check_output(git(root,"log","--full-history","--diff-filter=A",
                                               "--format=%H",execution_producer,"--",BRIDGE_FILE),
                                            text=True).splitlines()
        require(len(additions)==1,"Historical support head requires unchanged deployment runtime or one fixed bridge")
        registered = subprocess.check_output(git(root,"show",f"{additions[0]}:{BRIDGE_FILE}"))
        current = subprocess.check_output(git(root,"show",f"{execution_producer}:{BRIDGE_FILE}"))
        require(current==registered,"An existing legacy bridge cannot be amended")
        bridge = json.loads(registered)
        require(bridge.keys()=={"schema","legacy_anchor","legacy_runtime_sha256","execution_anchor",
                               "execution_runtime_sha256"}
                and bridge["schema"]==BRIDGE_SCHEMA and bridge["legacy_anchor"]==LEGACY_ANCHOR
                and full_commit(bridge["execution_anchor"])
                and plan_runtime==bridge["legacy_runtime_sha256"]==runtime_sha256(LEGACY_ANCHOR,root=root)
                and execution_runtime==bridge["execution_runtime_sha256"]==
                    runtime_sha256(bridge["execution_anchor"],root=root),
                "Historical support head requires unchanged deployment runtime or the exact reviewed legacy bridge")
        ancestor(LEGACY_ANCHOR,plan_producer,root=root)
        ancestor(bridge["execution_anchor"],execution_producer,root=root)
        mode = "pinned_legacy_bridge"
    return dict(mode=mode,plan_runtime_sha256=plan_runtime,execution_runtime_sha256=execution_runtime)
