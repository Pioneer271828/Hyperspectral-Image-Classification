from dataclasses import replace

import numpy as np
import pytest
import torch

from evaluation import wp06_capacity_released as old
from evaluation.frozen_training_stages import file_pin
from evaluation.wp11_deployment_synthetic import models
from evaluation.wp12_support_head import comparison
from evaluation.wp12_support_head_cost import original_pixel_ids
from inference import support_readout as readout
from inference.native_deployment import Runtime, prepare
from research_runtime import write_json
from transfer.native_independent_release import synthetic_bank
from transfer.spatial_coverage_protocol import bank_observation
from transfer.native_scale_observation import Lattice


def test_portable_affine_matches_independent_solver_and_has_no_fit_examples():
    rng = np.random.default_rng(17)
    train,dev = rng.normal(size=(15,64)),rng.normal(size=(43,64))
    train[:,3] = 2
    y = np.repeat([1,2,3],5)
    fit = readout.fit(train,y,np.arange(len(y)))
    expected = old.primal_predictions({"fused":train},{"fused":dev},y)["fused"]
    assert np.array_equal(readout.decisions(readout.scores(dev,fit)),expected)
    assert fit.keys() == {"mean","scale","weight","intercept"}


def test_reload_rejects_tamper_changed_source_and_nonfinite_affine(tmp_path):
    rng = np.random.default_rng(33)
    head = readout.fit(rng.normal(size=(6,64)),np.repeat([1,2],3),np.arange(6))
    source = dict(scene="xuzhou",checkpoint="a"*64,state="b"*64,authority="c"*64,
                  fit_receipt="d"*64,class_ids=[1,2],bands=dict(full=[0,1,2,3],central_gap=[0,3],matched_dispersed=[1,3]))
    packet = dict(schema=readout.SCHEMA,policy=readout.POLICY,source=source,selections={
        "all":dict(support_sha256="e"*64,counts=[3,3],heads=dict.fromkeys(readout.VIEWS,head))})
    path = tmp_path/"head.json"
    write_json(path,packet)
    pin = file_pin(path)
    assert readout.load(tmp_path,pin,source)["source"] == source
    with pytest.raises(ValueError,match="source binding"):
        readout.load(tmp_path,pin,dict(source,checkpoint="f"*64))
    head["scale"][0] = -1
    write_json(path,packet)
    with pytest.raises(ValueError,match="hash differs"):
        readout.load(tmp_path,pin,source)
    with pytest.raises(ValueError,match="coefficients"):
        readout.load(tmp_path,file_pin(path),source)


def test_block_runtime_preserves_native_heads_and_rejects_foreign_requests(tmp_path):
    torch.set_num_threads(1)
    bank,_ = synthetic_bank(tmp_path,"synthetic",0)
    row = bank.observation_row("devval",bank.role_rows("devval")[0]["id"])
    model,_ = models(64,"xuzhou")
    classes = model.heads["xuzhou"].out_features
    runtime = Runtime(device="cpu",full=model)
    requests,bands = {},{}
    for band in readout.VIEWS:
        actual = bank_observation(bank,row,"unit",band,observation_only=True)
        requests[band] = prepare(actual,"unit",key=band,device="cpu",original_band_count=bank.release.shape[-1])
        bands[band] = list(actual.view.bands.band_ids)
    rng = np.random.default_rng(18)
    labels = np.repeat(np.arange(1,classes+1),2)
    base = rng.normal(size=(len(labels),64))
    heads = {v:readout.fit(base*(i+1),labels,np.arange(len(labels))) for i,v in enumerate(readout.VIEWS)}
    artifact = dict(source=dict(scene="xuzhou",state=runtime.identity["full"],class_ids=list(range(1,classes+1)),
                               bands=bands),selections={"all":{"heads":heads}})
    adapted = readout.SupportRuntime(runtime,artifact,"all")
    for band in ("full","central_gap","full"):
        req = requests[band]
        native,adapt = (adapted.run(req,m) for m in ("native","same_view"))
        assert all(torch.equal(a,b) for a,b in zip(native["native"],adapt["native"],strict=True))
        assert adapt["counts"]["graph_builds"] == 1 and not runtime.cache
    with pytest.raises(ValueError,match="foreign|Foreign"):
        adapted.run(replace(req,x=req.x.to("meta")),"same_view")
    with pytest.raises(ValueError,match="retained columns"):
        adapted.run(replace(req,band="central_gap"),"same_view")
    assert not model.heads["xuzhou"]._forward_pre_hooks


def test_native_block_pixel_mapping_keeps_original_ids_and_refuses_coarsened_cells():
    grid = Lattice.make((2,4,3,6),(8,10),1)
    assert original_pixel_ids(grid).tolist()==[[23,24,25],[33,34,35]]
    with pytest.raises(ValueError,match="original native pixel grid"):
        original_pixel_ids(Lattice.make((2,4,3,6),(8,10),2))


def test_focus_gain_can_hide_other_class_loss_below_confirmation_floor():
    labels = np.repeat([1,2,3],5)
    before,after = labels.copy(),labels.copy()
    before[0] = 2
    after[10] = 2
    got = comparison(labels,before,after,np.arange(15)//5,(1,2),3)
    assert got["gained_points"] == got["lost_points"] == 1
    assert got["delta_oa"] == 0
    assert got["ordinary_class_recall_drops"] == ["3"]
    assert not got["registered_limits_passed"]
