"""Protect single-site donor and exact-geometry mechanism contracts."""

import numpy as np
import pytest
import torch

from evaluation import wp12_mechanism_factorial as factorial, wp12_mechanism_real as real
from evaluation.wp12_mechanism_review import factorial_contrast, scope
from transfer.band_graph_control import fixed_graph
from transfer.native_mechanism_edges import EdgeProvider, exact_control, publish


def test_invalid_tensor_and_real_donor_leak_are_rejected():
    with pytest.raises(ValueError,match="dimensions"):
        real.switch(torch.ones(3,32),torch.ones(3,16))
    accepted=dict(schema=real.SCHEMA,scope="released_development_offline_only",
                  query_accessed=False,raw_gt_loaded=False,independent_accuracy=None,optimizer_updates=0)
    scope(accepted,real.SCHEMA,"released_development_offline_only",updates=0)
    with pytest.raises(ValueError,match="Independent accuracy"):
        scope({**accepted,"independent_accuracy":.9},real.SCHEMA,"released_development_offline_only",updates=0)


def test_metadata_control_preserves_actual_columns_and_statistics():
    sample=dict(truth=[1,2],point_set="points",bands=[10,12],spectral="spectral",grid=[2,2],
                factor=1,wavelength_quality="approximate",metadata_fingerprint="original")
    fallback=sample|dict(wavelength_quality="index_only",metadata_fingerprint="missing")
    factorial.verify_input_pair(sample,fallback)
    with pytest.raises(ValueError,match="changed actual retained"):
        factorial.verify_input_pair(sample,fallback|dict(bands=[10,11]))


def test_independent_factorial_correctly_separates_interaction():
    from itertools import product

    cells={(p,m,g):3*p-2*m+g+4*p*g for p,m,g in product((0,1),repeat=3)}
    result=factorial_contrast(cells,float)
    assert result=={"placement":5.,"metadata":-2.,"grid":3.,
                    "placement*metadata":0.,"placement*grid":4.,"metadata*grid":0.,
                    "placement*metadata*grid":0.}


def test_exact_graph_preserves_distance_degree_and_rejects_foreign_band(tmp_path):
    bank,roles,geometry=publish(tmp_path/"private","remote",64)
    provider=EdgeProvider(tmp_path,bank,roles,64,"fixed_none","fixture")
    tile=roles["evaluation"][0]
    assert geometry[str(tile)]["boundary_points"]==4
    with pytest.raises(ValueError,match="unauthorized"):
        provider.materialize("devval",dict(tile=tile,factor=1,band="central_gap"),"fixture")
    native,_,_=provider.confirmation("selected",tile,1,"central_gap")
    base=exact_control(native)
    graph=fixed_graph(native,base,"low_weight")
    graph.verify(native)
    full,_,_=provider.confirmation("selected",tile,1,"full")
    with pytest.raises(ValueError,match="escapes"):
        graph.verify(full)
    high=base.removal("important")
    low=base.removal("low_weight")
    assert high!=low and len(high)==len(low)==4
    points=base.coordinates.numpy()

    def signature(edges):
        ij=np.asarray([(e//9,e%9) for e in edges])
        return (sorted(ij[:,0].tolist()),sorted(ij[:,1].tolist()),
                sorted(np.square(points[ij[:,0]]-points[ij[:,1]]).sum(1).tolist()))

    assert signature(high)==signature(low)
