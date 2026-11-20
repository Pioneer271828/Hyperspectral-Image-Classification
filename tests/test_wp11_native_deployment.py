import pytest
import torch

from evaluation import wp11_deployment_cost as costs
from evaluation.wp11_deployment_synthetic import models
from inference.native_deployment import Runtime, prepare
from transfer.native_independent_release import synthetic_bank
from transfer.spatial_coverage_protocol import bank_observation


def test_native_request_rejects_wrong_identity_and_temperature_hooks(tmp_path):
    torch.set_num_threads(1)
    bank,_=synthetic_bank(tmp_path,"synthetic",0)
    row=bank.role_rows("devval")[0]
    actual=bank_observation(bank,row,"consumer","central_gap")
    options=dict(key="row",device="cpu",original_band_count=bank.release.shape[-1])
    req=prepare(actual,"consumer",**options)
    assert .4<req.coverage<.8 and req.known
    with pytest.raises(ValueError,match="fixed native"):
        prepare(actual,"consumer",expected="another_band",**options)
    other=dict(row,parent_tile=row["parent_tile"]+1)
    with pytest.raises(ValueError,match="registered bank context"):
        bank_observation(bank,other,"consumer","central_gap")
    full,light=models(64,"xuzhou")
    hook=full.heads["xuzhou"].register_forward_hook(lambda _,args,x:x/2)
    try:
        with pytest.raises(ValueError,match="temperature"):
            Runtime(device="cpu",full=full)
    finally:
        hook.remove()
    rt=Runtime(device="cpu",full=full,light=light)
    fast=rt.run([req],"light")
    assert fast["counts"]["graph_builds"]==0 and fast["counts"]["full_calls"]==0
    with pytest.raises(ValueError,match="registered pair"):
        Runtime(device="cpu",full=full,identity={"full":"foreign"})
    another,_=models(512,"xuzhou")
    unrelated=Runtime(device="cpu",full=another,light=light)
    with pytest.raises(ValueError,match="one frozen source"):
        unrelated.run([req],"adaptive",threshold=0.)


def test_accepted_stage_resumes_after_failed_attempt_without_repeating(tmp_path, monkeypatch):
    producer={"commit":"synthetic-stage"}
    out=tmp_path/"fixture"
    out.mkdir()
    plan_sha="7"*64
    parent={"path":str(out/"measurement-plan.json"),"sha256":plan_sha}
    plan={"repeats":dict(zip(costs.SCOPES,costs.REPEATS,strict=True))}
    monkeypatch.setattr(costs,"read_plan",lambda *args:(parent,plan))
    monkeypatch.setattr(costs,"require_frozen_code",lambda _:None)
    calls=[]
    def run(*args):
        calls.append(1)
        if len(calls)==1:
            raise RuntimeError("first worker failed")
        return [dict(wall_ms=float(i),graph_ms=1.,rss_kib=10,cuda_peak_bytes=None,prediction="frozen") for i in (1,2,3)],None
    monkeypatch.setattr(costs,"external_cold",run)
    args=(tmp_path,out,producer,"xuzhou",costs.SCOPES[0],"cpu","cpu",{},plan_sha)
    with pytest.raises(RuntimeError,match="worker failed"):
        costs.stage(*args)
    case=out/"stages"/"cpu"/"xuzhou"/costs.SCOPES[0]
    assert not (case/"accepted.json").exists()
    assert (case/"attempt-1"/"failure.json").is_file()
    won=costs.stage(*args)
    assert costs.stage(*args)==won and len(calls)==2
    assert (case/"attempt-2").is_dir()
