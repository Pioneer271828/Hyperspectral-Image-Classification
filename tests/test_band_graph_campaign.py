import pytest
import torch

from evaluation.matched_edge_controls import freeze, verify
from transfer.band_graph_campaign import admitted, compatibility, endpoint
from transfer.band_graph_fixture import BandGraphFixture
from transfer.band_graph_control import ViewControl, row_coo
from transfer import native_scale_training as native


@pytest.mark.parametrize("layout",("horizontal","vertical"))
def test_real_band_certificate_matches_support_and_preserves_frozen_cnn(tmp_path,layout):
    torch.set_num_threads(1)
    bank, _ = admitted(tmp_path,"remote",layout,64)
    a,b = (BandGraphFixture(bank,64,layout,band,"fixed_none","fixture") for band in ("full","continuous"))
    assert compatibility(a,b)
    model=a.baseline(64)
    original=endpoint(a,model)
    important=endpoint(a.variant("important"),model)
    assert original["scores"]["cnn"]==important["scores"]["cnn"]
    assert all(x["nnz"]-y["nnz"]==(8 if layout=="vertical" else 4)
               and x["q"]==y["q"] for x,y in zip(original["traces"],important["traces"],strict=True))
    actual=a.views["devval",4][0]
    scope=a[actual]
    assert scope.removal("low_weight")!=scope.removal("important")
    verify(scope.plan,scope.state.A0,scope.coordinates,expected_hash=scope.plan["plan_hash"])
    empty=freeze(scope.state.A0,scope.coordinates,per_row=0,distance_bins=[36],seeds=[7719],
                 symmetric=layout=="vertical")
    assert all(c["status"]=="no_distinct_control" and c["removed"] is None for c in empty["controls"].values())


def test_bound_control_rejects_other_split_or_tampered_view(tmp_path):
    torch.set_num_threads(1)
    bank,_=admitted(tmp_path,"spectral","horizontal",64)
    core=BandGraphFixture(bank,64,"horizontal","full","fixed_none","fixture")
    first=core.views["devtrain",0][0]
    foreign=core.views["devval",4][0]
    with pytest.raises(ValueError,match="different band"):
        core[first].actual(foreign,"fixed_none")
    first_control=ViewControl.create(first,"horizontal")
    assert first_control.fingerprint==core[first].fingerprint
    run=native.Run(core.contract(9,64,"fixed_none",1,"fixture"),"fixture",released=core)
    run.start()
    run.update(run.registered["training"][0])
    with pytest.raises(ValueError,match="registration"):
        alternative=core.variant("important")
        other=native.Run(alternative.contract(9,64,"important",1,"fixture"),"fixture",released=alternative)
        other.restore(run.pack(),1)

def test_random_and_weak_are_distinguishable_when_the_support_has_choices():
    torch.set_num_threads(1)
    edges = [(i,i) for i in range(8)]
    weights = [1.] * 8
    for node in range(4):
        for offset in range(4):
            edges.append((node,4+(node+offset)%4))
            weights.append((2.,.1,.6,.7)[offset])
    adjacency=row_coo(edges,weights,8)
    xy=torch.tensor([[float(i%4),float(i//4)*25.] for i in range(8)])
    result=freeze(adjacency,xy,per_row=1,distance_bins=[12,36],seeds=list(range(6)))
    verify(result,adjacency,xy,expected_hash=result["plan_hash"])
    controls=result["controls"]
    assert all(value["status"]=="optimal" for value in controls.values())
    assert any(controls[f"random:{i}"]["removed"] != controls["low_weight"]["removed"] for i in range(6))
