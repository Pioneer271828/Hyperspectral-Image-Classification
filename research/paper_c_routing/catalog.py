"""Materialize all ordered-direction gates without loading data or model tensors."""
import argparse
import csv
import json
import math
from pathlib import Path

CLASSES={"paviaU":9,"xuzhou":9,"whu-honghu":22,"whu-longkou":9,"whu-hanchuan":16,"laoyuhesidi":8}
SOURCES={
 "paviaU":{"device":"reported ROSIS-03","evidence":"https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes","gap":"103-band version to original detector/band calibration binding; current provider fetch403"},
 "xuzhou":{"device":"HySpex VNIR-1600 + SWIR-384; resampled/merged acquisition product",
            "evidence":"https://www.3sobs.cn/share#Xuzhou",
            "original_paper":"https://doi.org/10.1109/JSTARS.2019.2892975",
            "paper_page":634,"gap":"per-band centers/SRF/retained-band mapping and local version binding; original paper class3/4=coal/cement differs from historical project class5/6 names"},
 "whu-honghu":{"device":"Headwall Nano-Hyperspec17mm,2017-11-20","evidence":"https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm","gap":"class21 independent geography; detector calibration and retained map"},
 "whu-longkou":{"device":"Headwall Nano-Hyperspec8mm,2018-07-17","evidence":"https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm","gap":"new role release and local version/semantic binding; serial device unknown"},
 "whu-hanchuan":{"device":"Headwall Nano-Hyperspec17mm,2016-06-17","evidence":"https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm","gap":"admitted image payload and role release; serial device unknown"},
 "laoyuhesidi":{"device":"PROVISIONAL candidate: OHS satellite product, not bound to local file",
                "evidence":"https://doi.org/10.3390/rs17050763",
                "primary_origin_reference":"https://doi.org/10.1109/JSTARS.2024.3403863",
                "gap":"AFGNet experiment reports391x591x32/8classes/10m/May2024 but project historical inventory391x550x32; no exact local payload/label-map/license/role binding, referenced origin not independently verified"}}


def build(table):
    with Path(table).open(encoding="utf-8-sig") as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=30 or {(r['source'],r['target']) for r in rows}!={(s,t) for s in CLASSES for t in CLASSES if s!=t}:
        raise ValueError("Complete30-direction roster required")
    out=[]
    for row in rows:
        src,target=row["source"],row["target"]
        n=CLASSES[target];ready=row["status"]=="READY_DEVELOPMENT_ONLY"
        executable=(src,target)==("paviaU","xuzhou")
        out.append({"source":src,"target":target,"data_gate":row["status"],
                    "source_evidence":SOURCES[src],"target_evidence":SOURCES[target],
                    "model_gate":"G3_P2_three_frozen_run_seeds_with_CNN_copy" if executable else "NO_VERIFIED_SINGLE_SOURCE_FROZEN_LIGHT_FULL_PAIR",
                    "development_execution":executable and ready,"independent_calibration":False,"independent_confirmation":False,
                    "missing_source_target_pair":not executable,"target_classes":n,
                    "additional_roles_engineering_floor":{"new_calibration_points":5*n,"new_confirmation_points":5*n,
                        "per_class_per_role":5,"independent_parent_components_per_class_per_role":2,
                        "is_power_calculation":False,"current_V5_reusable_for_these_roles":False},
                    "statistical_required_sample_size":None,
                    "iid_zero_error_illustration_per_class":math.ceil(math.log(.05/n)/math.log(.98)),
                    "illustration_scope":"Bonferroni one-sided95% upper error<=2% with zero errors under IID. NOT a paired recall noninferiority sample size; adjacent pixels are not IID.",
                    "required_authorization":"Independent trusted custodian + project user release, pinned region/labels/observations/history; no author webpage releases project confirm labels",
                    "minimal_gap":row["minimum_gap"],"source_training_budget":"inherit exact source checkpoint, never target-scratch-as-transfer",
                    "no_unlabeled_confirm_access_during_fit":True})
    return {"schema":"paper_c_30_direction_runtime_eligibility_v2","directions":out,"target_local_control":"WP11 scratch xuzhou→xuzhou, outside cross-device roster", "new_optimizer_updates":0}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--table",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    with Path(a.out).open("x") as f:json.dump(build(a.table),f,indent=2,allow_nan=False)
