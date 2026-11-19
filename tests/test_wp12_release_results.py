import pytest

from evaluation.wp12_release_results import HEADS, cohorts, cost_missing, metrics, select


def sample(seed, device, workload, matrix):
    return dict(source="fixture",protocol="released",scene="xuzhou",seed=seed,support=512,
                method="native",arm="none",role="devval",band="full",factor=1,endpoint="selected",
                model=str(seed),device=device,workload=workload,measurement_scope=None,
                cost=cost_missing("not timed"),heads={head:metrics(matrix,2) for head in HEADS})


def test_aggregate_requires_matched_scope_and_keeps_undefined_recall():
    rows=[sample(64,"GPU-A","resident",[[5,0],[0,0]]),sample(512,"GPU-A","resident",[[4,1],[0,0]]),
          sample(2048,"GPU-B","cold",[[1,4],[0,0]])]
    groups=cohorts(rows)
    supports=[row for row in groups if row["head"]=="fused"]
    assert [row["model_seeds"] for row in supports]==[[64,512],[2048]]
    assert supports[0]["oa"]["mean"]==.9 and supports[0]["oa"]["sample_sd"]==pytest.approx(.141421356)
    assert supports[1]["oa"]["sample_sd"] is None
    assert supports[0]["per_class"][1]["recall"]=={"mean":None,"sample_sd":None}
    assert select(rows,scene="xuzhou",device="GPU-B")==[rows[-1]]
    with pytest.raises(ValueError,match="filter"):
        select(rows,protocol_family="foreign")
    with pytest.raises(ValueError,match="duplicates"):
        cohorts([rows[0],rows[0]])
