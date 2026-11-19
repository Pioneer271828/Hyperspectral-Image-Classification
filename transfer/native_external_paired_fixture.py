"""Publish explicit external-format fixtures with scene-sized coordinates."""

from dataclasses import asdict

import numpy as np

from evaluation.frozen_training_stages import file_pin, require
from model.checkpoint_io import save_model_checkpoint
from model.research_config import digest
from scripts.run_frozen_band_region import _write
from transfer import development_bank as custody, native_external_paired as paired
from transfer import native_independent_release as independence, native_paired_head_fixture as layout
from transfer.development_bank_training import _baseline
from transfer.g3_training import state_hash
from transfer.spatial_coverage_protocol import release_statistics

SCHEMA = "wp05_wp06_external_format_fixture_v1"
COHORTS = (
    ("OneA", 1, 64, "a", (64,), "inherit_head", True, True),
    ("FiveA", 5, 512, "a", (64, 512, 2048), "seeded_head", True, True),
    ("FiveB", 5, 512, "b", (64,), "inherit_head", True, False),
    ("TenB", 10, 2048, "b", (64,), "inherit_head", False, True),
)


def definition(kind, k, draw, spatial, *, omit_support=False):
    scene = paired.SCENES[kind]
    shape, classes = custody.SCENES[scene]
    offset, index = {"a": (0, 0), "b": (165, 1), "h": (330, 2)}[spatial]
    h, w = layout.SPECS[kind][5]
    bands = layout.master(kind, shape[-1])
    native_seed = layout.SPECS[kind][4] + 29*index
    roles = {role: [] for role in independence.ROLES}
    rows, arrays = [], {}
    for i, role in enumerate(["support"]*4+["calibration"]*2+["evaluation"]*3):
        key = 100*index+i
        y0, x0 = offset+i//3*48, i%3*59+(3 if spatial=="b" else 0)
        values, truth = layout.field(kind, bands, classes, h, w,
                                     "erased" if i in (1, 5, 7) else "retained",
                                     key, native_seed, spatial=="b" or i==8)
        points = []
        for label in range(1, classes+1):
            locations = np.argwhere(truth==label)
            rng = np.random.default_rng([native_seed, key, label, 409])
            if role=="support":
                if omit_support and label==classes:
                    continue
                count = k//4 + int((i-label)%4<k%4)
                start = (0, 1, 3)[paired.SEEDS.index(draw)]
            else:
                count = 3
                start = 0
            positions = rng.permutation(len(locations))[start:start+count]
            require(len(positions)==count,"Fixture class does not have its declared release population")
            points.extend([[(y0+int(y))*shape[1]+x0+int(x),label] for y,x in locations[positions]])
        box = [y0, y0+h, x0, x0+w]
        rows.append(dict(id=key, role=independence.ROLES[role], partition="train" if role=="support" else "val",
                         box=box, parent_tile=key, parent_box=box, samples=sorted(points),
                         representation="raw_fp32", origin=dict(array_sha256=custody.array_hash(values),
                         simulation=SCHEMA), blocked=None))
        arrays[key] = values
        roles[role].append(key)
    rows.sort(key=custody._row_key)
    plan = digest([SCHEMA, scene, shape, roles, [row["box"] for row in rows]])
    stats = release_statistics(arrays, rows, plan)
    release = custody.BankRelease(scene, "synthetic", shape, classes, plan, bands, stats,
        dict(shared_state=digest([SCHEMA,kind,native_seed]), kind=SCHEMA),
        dict(minimum_buffer=8, roles=digest(roles), support_seed=draw, spatial_repeat=spatial, k=k), tuple(rows))
    return release, arrays, roles


def release_bank(path, kind, k, draw, spatial, *, omit_support=False):
    release, arrays, roles = definition(kind, k, draw, spatial, omit_support=omit_support)
    stored = custody.publish(path,path.parent,release,lambda row:arrays[row["id"]],resume=path.exists())
    return dict(pin={name:stored[name] for name in ("root","sha256","registration")},release=asdict(release)), roles, release


def make(out, destination, code, *, gpu=None):
    out.mkdir(parents=True)
    request = dict(schema=paired.SCHEMA, scope=paired.TEST, releases={})
    negatives = {}
    draw_points = {}
    for kind in paired.SCENES:
        basedir = out/kind
        basedir.mkdir()
        historical, _, hist_release = release_bank(basedir/"history",kind,1,64,"h")
        model = _baseline(hist_release,64)
        weights = basedir/"external-test-source.pt"
        save_model_checkpoint(model,weights)
        source = dict(checkpoint=file_pin(weights)["sha256"],state=state_hash(model),weights=file_pin(weights),
                      sidecar=file_pin(weights.with_suffix(".pt.json")))

        def create(name,k,draw,spatial,accuracy,cost,*,gap=8,omit_support=False):
            entry, roles, actual = release_bank(basedir/name,kind,k,draw,spatial,omit_support=omit_support)
            manifest = _write(basedir/(name+"-manifest.json"),dict(schema=independence.MANIFEST,candidate=entry,
                              roles=roles,minimum_buffer=gap,history=[historical]))
            approval = _write(basedir/(name+"-owner.json"),dict(schema=independence.AUTHORITY,
                              approved_manifest_sha256=manifest["sha256"],source_mode="synthetic",
                              historical_bank_sha256=[historical["pin"]["sha256"]],
                              uses=dict(independent_accuracy=accuracy,cost=cost,frozen_diagnostics=False)))
            return manifest, approval, actual

        for name,k,draw,spatial,seeds,mode,accuracy,cost in COHORTS:
            manifest,approval,actual = create(name,k,draw,spatial,accuracy,cost)
            key = kind+name
            draw_points[key] = {sample for row in actual.rows if row["role"]=="devtrain" for sample,_ in row["samples"]}
            request["releases"][key] = dict(kind=kind,k=k,support_seed=draw,spatial_repeat=spatial,
                                          manifest=manifest,authority=approval,source=source,model_seeds=list(seeds),
                                          initialization=mode,device=gpu if gpu and name=="OneA" else paired.DEVICE_CPU)
        for left in (name for name in draw_points if name.startswith(kind)):
            for right in (name for name in draw_points if name.startswith(kind) and name!=left):
                require(not draw_points[left].intersection(draw_points[right]),"Fixture reused a released support point")
        negatives[kind] = {}
        for name,gap,omission in (("shortBuffer",25,False),("missingSupportClass",8,True)):
            manifest,approval,_ = create(name,5,512,"a",True,True,gap=gap,omit_support=omission)
            negatives[kind][name] = dict(manifest=manifest,authority=approval)
    request_pin = _write(out/"test-request.json",request)
    approval_pin = _write(out/"test-only-execution.json",paired.expected_approval(request,request_pin,destination,code,test=True))
    return _write(out/"fixture.json",dict(request=request_pin,approval=approval_pin,negative=negatives,
                          source_scope=SCHEMA,real_approval_created=False))
