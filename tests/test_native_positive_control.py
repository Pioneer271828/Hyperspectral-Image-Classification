from copy import deepcopy

import numpy as np
import pytest
import torch

from evaluation.native_positive_oracle import oracle
from evaluation.native_spectral_control import column_positions, native_signatures, verify_loader
from evaluation.wp06_observation_audit import verify_fit_statistics
from evaluation.wp12_positive_control import positive_gate, selected_step
from transfer import development_bank as custody
from transfer.native_long_campaign import synthetic_definition
from transfer.native_positive_fixture import CLOCK, CLASSES, SCIENTIFIC, STEPS, bands, centers, definition, profiles, publish_bank
from transfer.native_positive_training import admitted_forward, make_model
from transfer.spatial_coverage_protocol import bank_view


def test_physical_controls_preserve_amplitude_and_identifiable_classes():
    book, axis = centers(),bands()
    a = column_positions(axis.valid,"central_gap")
    b = column_positions(axis.valid,"matched_dispersed")
    assert len(a) == len(b) == 5 and 2 not in a and 2 not in b
    assert np.array_equal(book[0,a],book[1,a])
    assert not np.array_equal(book[2,a],book[3,a])
    assert np.unique(book[:,column_positions(axis.valid,"full")],axis=0).shape[0] == CLASSES
    for arrangement in ("iid","block"):
        profile = profiles()[f"row-{arrangement}-local1-context0"]
        fit,_,roles,correlation = definition(profile)
        assert "confirmation" not in roles
        held,_,other,_ = definition(profile,confirm=True)
        assert not set(roles["calibration"]) & set(other["confirmation"])
        assert all(fit.statistics[k] == held.statistics[k] for k in ("minimum","maximum"))
        assert (correlation >= .30) if arrangement=="block" else (correlation <= .20)


def test_wrong_label_order_and_polluted_statistics(tmp_path):
    selected = profiles()["row-block-local1-context0"]
    bank,roles,_ = publish_bank(tmp_path/"fit",selected)
    data,proof = oracle(bank,roles,"test","calibration",selected)
    assert all(proof["gates"].values())
    source = bank.role_row("devval",roles["calibration"][0])
    actual,labels,targets = bank_view(bank,source,"test","full")
    release,raw,_,_ = definition(selected)
    minimum,maximum = [release.statistics[key] for key in ("minimum","maximum")]
    columns = column_positions(release.bands.valid,"full")
    with pytest.raises(ValueError,match="band order"):
        verify_loader(source,release.shape,raw[source["id"]],minimum,maximum,columns[::-1],
                      actual,labels,targets,release.bands)
    wrong = labels.copy()
    wrong[0,1] = (wrong[0,1] % CLASSES)+1
    with pytest.raises(ValueError,match="supervision identities"):
        verify_loader(source,release.shape,raw[source["id"]],minimum,maximum,columns,
                      actual,wrong,targets,release.bands)
    contaminated = deepcopy(release.statistics)
    contaminated["maximum"][0] += 10
    with pytest.raises(ValueError,match="support alone"):
        verify_fit_statistics(raw,roles["support"],contaminated)
    pixel = make_model("pixel",64).eval()
    original,_,_ = admitted_forward(pixel,actual,"test","devval",None)
    shuffled,_,_ = admitted_forward(pixel,actual,"test","devval",None,order=np.arange(actual.receipt["pixels"])[::-1].copy())
    assert all(torch.equal(a,b) for a,b in zip(original,shuffled,strict=True))


def test_single_region_collapse_is_reported(tmp_path):
    release,raw,roles = synthetic_definition("xuzhou")
    pin = custody.publish(tmp_path/"old",tmp_path,release,lambda row:raw[row["id"]])
    bank = custody.Bank({k:pin[k] for k in ("root","sha256","registration")},tmp_path,release)
    row = bank.role_row("devval",roles["calibration"][0])
    observation,_,_ = bank_view(bank,row,"test","full")
    model = make_model("native",64).eval()
    representation,_,_ = native_signatures(model,observation,"test","devval")
    assert representation["regions"] == 1
    assert representation["pool_variance_ratio"] < 1e-9
    assert np.var(representation["pre_pool"],axis=0).sum() > 0


def test_wrong_selection_and_false_confirmation_are_not_evidence():
    spec = dict(selection=SCIENTIFIC["selection"],steps=STEPS,validation=[{"band":"full"}])
    history = [dict(step=epoch,fused_nll=0.5 if epoch in (120,180) else 1.) for epoch in CLOCK]
    assert selected_step(spec,history) == 120
    with pytest.raises(ValueError,match="full-band"):
        selected_step({**spec,"validation":[{"band":"central_gap"}]},history)
    null = dict(local=False,context=False)
    issue = positive_gate(null,dict(local=True,context=True),.2,.26,.3,0.)
    assert issue["null_detected_false_positive"] and not issue["center_supported"]
    weak = positive_gate(dict(local=True,context=False),dict(local=True,context=True),.6,.2,.95,0.)
    assert not weak["local_reference_valid"] and not weak["center_supported"]
