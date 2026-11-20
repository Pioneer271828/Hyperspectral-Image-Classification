import pytest
import torch

from transfer import native_independent_release as independent
from scripts.run_frozen_band_region import _read, _write


def roles_for(bank):
    support=sorted(row["id"] for row in bank.role_rows("devtrain"))
    val=sorted(row["id"] for row in bank.role_rows("devval"))
    return dict(support=support,calibration=val[::2],evaluation=val[1::2])


def test_independent_release_distinguishes_roles_from_spatial_reuse(tmp_path):
    torch.set_num_threads(1)
    prior,_=independent.synthetic_bank(tmp_path,"prior",0)
    current,_=independent.synthetic_bank(tmp_path,"different",40)
    roles=roles_for(current)
    permitted=dict.fromkeys(independent.USES,True)
    report=independent.assess(current,[prior],roles,gap=3,permitted=permitted)
    assert all(item["eligible"] for item in report["uses"].values())
    assert all(min(row["per_class"])>=1 for row in report["role_reports"].values())
    with pytest.raises(ValueError,match="crosses"):
        independent.materialize_reviewed(current,report,roles,"calibration",roles["calibration"][0],
                                         "fixture","full",use="cost")
    _, labels, targets=independent.materialize_reviewed(current,report,roles,"evaluation",roles["evaluation"][0],
                                                       "fixture","central_gap",use="cost")
    assert labels is targets is None
    reused=independent.assess(prior,[prior],roles_for(prior),gap=3,permitted=permitted)
    assert reused["uses"]["cost"]["eligible"] and reused["uses"]["frozen_diagnostics"]["eligible"]
    assert not reused["uses"]["independent_accuracy"]["eligible"]
    assert reused["role_reports"]["evaluation"]["shared_historical_samples"]>0
    with pytest.raises(ValueError,match="narrower"):
        independent.assess(current,[prior],roles,gap=2,permitted=permitted)
    with pytest.raises(ValueError,match="reuse"):
        independent.assess(current,[prior],{**roles,"evaluation":roles["calibration"]},gap=3,permitted=permitted)


def test_external_intake_rejects_stale_authority_history_and_roleless_labels(tmp_path):
    prior,_=independent.synthetic_bank(tmp_path,"prior",0)
    current,entry=independent.synthetic_bank(tmp_path,"different",40)
    roles=roles_for(current)
    manifest=_write(tmp_path/"manifest.json",dict(schema=independent.MANIFEST,candidate=entry,history=[],
                                                  roles=roles,minimum_buffer=3))
    approval=dict(schema=independent.AUTHORITY,approved_manifest_sha256=manifest["sha256"],
                  historical_bank_sha256=[current.pin["sha256"]],source_mode="synthetic",
                  uses=dict.fromkeys(independent.USES,True))
    with pytest.raises(ValueError,match="bind the history"):
        independent.open_independent(tmp_path,manifest,_write(tmp_path/"stale.json",approval),test_history=[prior])
    approval["historical_bank_sha256"]=[prior.pin["sha256"]]
    authority=_write(tmp_path/"valid.json",approval)
    changed={**_read(manifest),"roles":{**roles,"calibration":[]}}
    opposite=_write(tmp_path/"opposite.json",changed)
    with pytest.raises(ValueError,match="not authorized"):
        independent.open_independent(tmp_path,opposite,authority,test_history=[prior])
    _, review, _=independent.open_independent(tmp_path,manifest,authority,test_history=[prior])
    assert review["uses"]["independent_accuracy"]["eligible"]


def test_end_to_end_independent_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(independent,"require_frozen_code",lambda _:None)
    result=_read(independent.synthetic_demo(tmp_path,tmp_path/"demo",{"commit":"fixture"}))
    assert result["label_blind_cost_view"] and result["frozen_diagnostic_view"]
    assert result["report"]["optimizer_updates"]==0
    assert all(row["scores"] is None for row in result["roles"]["calibration"])
    assert sum(row["scores"]["fused"]["count"] for row in result["roles"]["evaluation"])>0
