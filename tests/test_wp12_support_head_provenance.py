"""Regression coverage for immutable plan, runtime, receipt and P4 grid boundaries."""

import json
from pathlib import Path
import subprocess

import pytest

from evaluation import wp12_support_head_cost as cost, wp12_support_head_provenance as provenance
from evaluation.wp12_classification_review import p4_grid


def checkout(tmp_path):
    root = tmp_path/"repo"
    root.mkdir()
    subprocess.run(["git","-C",str(root),"init","-q"],check=True)
    return root


def commit(root, changes):
    for path,text in changes.items():
        file = root/path
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text(text,encoding="utf-8")
    git = ["git","-C",str(root)]
    subprocess.run([*git,"add","."],check=True)
    subprocess.run([*git,"-c","user.name=Fixture","-c","user.email=fixture@example.invalid",
                    "commit","-qm","fixture"],check=True)
    return subprocess.check_output([*git,"rev-parse","HEAD"],text=True).strip()


ENTRY = "scripts/run_wp12_support_head.py"
RUNTIME_PATHS = (ENTRY,"model/VCASAHGCN.py","transfer/spatial_coverage_protocol.py",
                 "transfer/development_bank.py","evaluation/wp12_support_head_cost.py")


def test_runtime_matches_from_nonroot_and_ignores_report_and_test_updates(tmp_path,monkeypatch):
    root = checkout(tmp_path)
    first = commit(root,{name:"fixed" for name in RUNTIME_PATHS})
    harmless = commit(root,{"tests/test_review.py":"pass","docs/guides/report.md":"guide",
                            "research/reports/report.md":"report","scripts/ingest_wp12_evidence.py":"report"})
    monkeypatch.chdir(tmp_path)
    assert provenance.compatibility(first,harmless,root=root)["mode"]=="identical_runtime"
    for relative in RUNTIME_PATHS:
        changed = commit(root,{relative:"different execution"})
        with pytest.raises(ValueError,match="unchanged deployment runtime"):
            provenance.compatibility(first,changed,root=root)
        commit(root,{relative:"fixed"})


def test_bridge_binds_exact_entry_and_is_not_a_mutable_exemption(tmp_path,monkeypatch):
    root = checkout(tmp_path)
    legacy = commit(root,{ENTRY:"legacy entry"})
    new = commit(root,{ENTRY:"reviewed adapter"})
    monkeypatch.setattr(provenance,"LEGACY_ANCHOR",legacy)
    bridge = dict(schema=provenance.BRIDGE_SCHEMA,legacy_anchor=legacy,
        legacy_runtime_sha256=provenance.runtime_sha256(legacy,root=root),
        execution_anchor=new,execution_runtime_sha256=provenance.runtime_sha256(new,root=root))
    recorded = commit(root,{provenance.BRIDGE_FILE:json.dumps(bridge)})
    proof = provenance.compatibility(legacy,recorded,root=root)
    assert proof["mode"]=="pinned_legacy_bridge"
    drift = commit(root,{ENTRY:"changed timing boundary"})
    with pytest.raises(ValueError,match="exact reviewed legacy bridge"):
        provenance.compatibility(legacy,drift,root=root)
    rewritten = commit(root,{provenance.BRIDGE_FILE:json.dumps(dict(bridge,execution_anchor=drift))})
    with pytest.raises(ValueError,match="cannot be amended"):
        provenance.compatibility(legacy,rewritten,root=root)


def test_legacy_producer_does_not_prove_measurement_and_v2_binds_new_writer(monkeypatch):
    old,new = "a"*40,"b"*40
    proof = dict(mode="pinned_legacy_bridge",plan_runtime_sha256="1"*64,execution_runtime_sha256="2"*64)
    monkeypatch.setattr(provenance,"ancestor",lambda a,b: None)
    monkeypatch.setattr(provenance,"compatibility",lambda a,b: proof)
    legacy = cost.measurement_identity(dict(schema=cost.LEGACY_COST,producer=old,device="cuda:0"),old,new)
    assert legacy["execution"] is None and legacy["plan_producer"]==old and legacy["reported_device"]=="cuda:0"
    hardware = dict(device="cpu",name="Xeon",machine="x86_64",torch="2.7",torch_threads=1,cpu_affinity=[0],
                    physical_uuid=None,driver=None,cuda_runtime=None)
    record = dict(schema=cost.COST,producer=new,plan_producer=old,compatibility=proof,device="cpu",
                  timing_scope=cost.TIMING_SCOPE,cuda_peak_bytes=None,
                  execution=dict(commit=new,runtime_sha256="2"*64,hardware=hardware))
    approved = cost.measurement_identity(record,old,new)
    assert approved["execution"]["commit"]==new and approved["plan_producer"]==old
    with pytest.raises(ValueError,match="misstates"):
        cost.measurement_identity(dict(record,producer=old),old,new)
    with pytest.raises(ValueError,match="cannot claim"):
        cost.measurement_identity(dict(record,cuda_peak_bytes=1),old,new)
    wrong = dict(record,execution=dict(record["execution"],runtime_sha256="4"*64))
    with pytest.raises(ValueError,match="differs from its Git provenance"):
        cost.measurement_identity(wrong,old,new)


@pytest.mark.parametrize("complete",[False,True])
def test_publication_consumes_actual_p4_grid_size_without_filling_missing_cases(complete):
    roster = [dict(k=k,method=m,accepted_arms=list(range(5)) if complete or (k==5)==(m=="scratch") else [])
              for k in (1,5) for m in ("scratch","shared_transfer")]
    count = 20 if complete else 10
    receipt = dict(roster_cells=roster,cases=list(range(count)),pending=list(range(count,20)),
                   combined_acceptance={"sha256":"synthetic-fixture"} if complete else None)
    assert sorted(p4_grid(receipt).values())==([5,5,5,5] if complete else [0,0,5,5])
    broken = dict(receipt,combined_acceptance=None if complete else {"sha256":"unexpected"})
    with pytest.raises(ValueError,match="partial or complete"):
        p4_grid(broken)
