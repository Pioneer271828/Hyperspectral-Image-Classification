from copy import deepcopy

import numpy as np
import pytest
import torch

from evaluation.frozen_training_stages import file_pin
from model.research_config import digest
from transfer.native_long_campaign import DESIGN, STEPS, SCENES, design, synthetic_definition
from transfer.native_long_run import checkpoints, register, write_checkpoint
from transfer.spatial_coverage_protocol import counts, retained_positions


def test_two_scenes_synthetic_roles_are_sealed_and_k5():
    assert len(design()) == 12 and STEPS*len(design()) == 18000
    for scene, classes in SCENES.items():
        release, data, assigned = synthetic_definition(scene)
        assert release.identity()["custody"]["roles_sha256"] == digest(assigned)
        assert not set(assigned["calibration"]) & set(assigned["evaluation"])
        by_role = {name: [row for row in release.rows if row["id"] in ids] for name, ids in assigned.items()}
        assert counts(by_role["support"], classes) == [5]*classes
        assert counts(by_role["calibration"], classes) == [5]*classes
        assert counts(by_role["evaluation"], classes) == [5]*(classes-2)+[2,2]
        assert len(retained_positions(release.bands, "central_gap")) == len(
            retained_positions(release.bands, "matched_dispersed"))
        changed = deepcopy(data)
        for row in by_role["evaluation"]:
            changed[row["id"]][:] = np.nan
        from transfer.spatial_coverage_protocol import release_statistics
        assert release.statistics == release_statistics(changed, release.rows, release.plan_hash)


def test_real_approval_rejects_before_any_training_output(tmp_path):
    work = tmp_path/"training"
    code = {"commit":"a"*40,"source_hash":"b"*64}
    with pytest.raises(ValueError,match="separate execution"):
        register(tmp_path,work,code,real=True)
    assert not work.exists()
    asked, approved = tmp_path/"request.json",tmp_path/"approval.json"
    asked.write_text('{"schema":"external-develop-only"}')
    approved.write_text('{"decision":"approved","optimizer_updates":0}')
    with pytest.raises(ValueError,match="roster"):
        register(tmp_path,work,code,request=file_pin(asked),approval=file_pin(approved),real=True)
    assert not work.exists()


def test_checkpoint_predecessor_and_authoritative_cursor(tmp_path):
    def value(step,start,parent,binding):
        return dict(journal=dict(step=step,start=start,parent=parent,registration=binding,updates=step-start),
                    training=dict(registration=binding,cursor=step,history=list(range(step)),
                                  validation=[],model={"a":torch.ones(1)}))
    binding = digest(DESIGN)
    start = write_checkpoint(tmp_path/"clock-0000.pt",value(0,0,None,binding))
    image = write_checkpoint(tmp_path/"clock-0003.pt",value(3,0,start["sha256"],binding))
    assert checkpoints(tmp_path,tmp_path,binding,read_all=True)[-1]["pin"] == image
    write_checkpoint(tmp_path/"clock-0007.pt",value(7,3,start["sha256"],binding))
    with pytest.raises(ValueError,match="lineage"):
        checkpoints(tmp_path,tmp_path,binding)
