"""Paired synthetic spectral/context interventions on the native released grid."""

from itertools import product
from pathlib import Path

import numpy as np

from evaluation.frozen_training_stages import file_pin, require
from evaluation.native_spectral_control import column_positions
from loadData.observation_sampling import BandMetadata
from model.research_config import digest
from transfer import development_bank as custody
from transfer.spatial_coverage_protocol import release_statistics

SCHEMA = "wp05_native_spectral_positive_controls_v2"
SEEDS = (64,512,2048)
LAYOUTS = ("row","column")
ARRANGEMENTS = ("iid","block")
ARMS = ("pixel","native","center")
TRAIN_VIEWS = ("full","central_gap","matched_dispersed")
VIEWS = (*TRAIN_VIEWS,"position_shuffled_full")
CLASSES = 9
STEPS = 240
CLOCK = (0,60,120,180,240)
SCENE = "xuzhou"
SHAPE = (64,88,11)
ROLE_SEEDS = dict(support=83219,calibration=94111,confirmation=709871)
SCIENTIFIC = dict(oracle_positive_aa=.95, negative_upper_aa=.25, spectral_reference_valid_aa=.85,
                  native_positive_aa=.65, candidate_delta_aa=.08, maximum_rare_loss=.20,
                  clustered_min=.30, iid_max=.20, deleted_pair_ceiling=.50,
                  steps=STEPS, clock=list(CLOCK), native_graph_scale=16, class_count=CLASSES,
                  train_each_class=5, calibration_each_class=6, confirmation_each_class=6,
                  confirmation_rare_each=3, fit_seed=ROLE_SEEDS["support"],
                  selection_seed=ROLE_SEEDS["calibration"], confirmation_seed=ROLE_SEEDS["confirmation"],
                  sigma=.055, signal_amplitude=.92, one_view_per_update=True,
                  selection="first_minimum_full_band_fused_calibration_nll",
                  candidate="fixed_center_spectrum_residual_only_to_fused_feature")


def profiles():
    return {f"{l}-{s}-local{int(a)}-context{int(b)}":dict(layout=l,arrangement=s,local=a,context=b)
            for l,s,a,b in product(LAYOUTS,ARRANGEMENTS,(False,True),(False,True))}


def cases():
    return {f"{family}-m{seed}-{arm}":dict(family=family,seed=seed,arm=arm)
            for family,seed,arm in product(profiles(),SEEDS,ARMS)}


def bands():
    return BandMetadata("synthetic-positive-control-v2",tuple(157+11*i for i in range(SHAPE[-1])),
                        tuple(440+40*i for i in range(SHAPE[-1])),"nm","verified",
                        tuple(i != 2 for i in range(SHAPE[-1])))


def centers():
    axis = bands()
    valid = column_positions(axis.valid,"full")
    outside = column_positions(axis.valid,"central_gap")
    central = np.setdiff1d(valid,outside)
    matrix = np.zeros((CLASSES,SHAPE[-1]),dtype=np.float32)
    masks = ((outside,(0,0,3,5,6,9,18,20,29)),(central,(1,2,4,8,16,6,10,17,24)))
    for columns,codes in masks:
        expanded = (np.asarray(codes)[:,None] >> np.arange(len(columns))) & 1
        matrix[:,columns] = (2*expanded-1)*SCIENTIFIC["signal_amplitude"]
    return matrix


def _tile(index, role, row, col, profile, book):
    # Roles use disjoint seeds; a given role/tile shares randomness across
    # all physical signal conditions, and transposition changes only geometry.
    rng = np.random.default_rng([ROLE_SEEDS[role],index,721])
    order = rng.permutation(np.arange(CLASSES))
    labels = np.repeat(order,2)
    random_labels = labels[rng.permutation(len(labels))]
    field_labels = (labels if profile["arrangement"] == "block" else random_labels).reshape(3,6)
    # Independent balanced carriers preserve the exact per-tile class
    # histogram of a useful carrier. Their assignments do not use truth.
    local_null = rng.permutation(labels).reshape(3,6)
    context_null = rng.permutation(labels).reshape(3,6)
    local = field_labels if profile["local"] else local_null
    ring = field_labels if profile["context"] else context_null
    field = np.repeat(np.repeat(book[ring],3,axis=0),3,axis=1)
    field[1::3,1::3] = book[local]
    field += rng.normal(0,SCIENTIFIC["sigma"],size=field.shape).astype(np.float32)
    same = (int((field_labels[:,1:] == field_labels[:,:-1]).sum())+
            int((field_labels[1:,:] == field_labels[:-1,:]).sum()))
    edges = field_labels.shape[0]*(field_labels.shape[1]-1)+(field_labels.shape[0]-1)*field_labels.shape[1]
    if profile["layout"] == "column":
        field = field.transpose(1,0,2).copy()
        field_labels = field_labels.T
    a,c = 2+row*21,2+col*21
    box = [a,a+field.shape[0],c,c+field.shape[1]]
    points = [(int((a+3*y+1)*SHAPE[1]+c+3*x+1),int(field_labels[y,x])+1)
              for y,x in np.ndindex(field_labels.shape)]
    physical = "devtrain" if role == "support" else "devval"
    record = dict(id=index,role=physical,partition="train" if physical == "devtrain" else "val",
                  parent_tile=index,box=box,parent_box=box,samples=[],
                  origin=dict(array_sha256=custody.array_hash(field),sealed_seed=ROLE_SEEDS[role]),
                  representation="raw_fp32",blocked=None)
    return record,field,points,(same,edges)


def definition(profile, *, confirm=False):
    require(profile in profiles().values(),"Unregistered synthetic factor combination")
    rows, arrays, all_points, roles, spatial = [],{},{},{},{}
    book = centers()
    for row,col in product(range(3),range(4)):
        role = ("support","calibration","confirmation")[(row+col)%3]
        if role == ("calibration" if confirm else "confirmation"):
            continue
        index = 1+row*4+col
        item,field,points,links = _tile(index,role,row,col,profile,book)
        rows.append(item)
        roles.setdefault(role,[]).append(index)
        arrays[index],all_points[index],spatial[index] = field,points,links
    by_id = {row["id"]:row for row in rows}
    for role,ids in roles.items():
        rng = np.random.default_rng([ROLE_SEEDS[role],93191])
        for klass in range(1,CLASSES+1):
            pool = [(tile,p) for tile in ids for p,c in all_points[tile] if c == klass]
            n = 5 if role == "support" else (3 if role == "confirmation" and klass > CLASSES-2 else 6)
            for at in rng.permutation(len(pool))[:n]:
                tile,p = pool[at]
                by_id[tile]["samples"].append([p,klass])
    for row in rows:
        row["samples"].sort()
    rows.sort(key=custody._row_key)
    plan = digest([SCHEMA,profile,SHAPE,roles,[row["box"] for row in rows]])
    fit = release_statistics(arrays,rows,plan)
    authority = dict(minimum_buffer=3,condition=profile,scope="confirmation" if confirm else "fit",
                     source_roster=digest(roles),parent_generator=SCHEMA)
    release = custody.BankRelease(SCENE,"synthetic",SHAPE,CLASSES,plan,bands(),fit,
               dict(kind="fixed_seeded_native_backbone",shared_state=digest([SCHEMA,"fresh_seeded_models"])),
               authority,tuple(rows))
    same, edges = map(sum,zip(*spatial.values(),strict=True))
    ratio = same/edges
    boundary = SCIENTIFIC["clustered_min"] if profile["arrangement"] == "block" else SCIENTIFIC["iid_max"]
    require((ratio >= boundary) if profile["arrangement"]=="block" else (ratio <= boundary),
            "The synthetic layout failed its predeclared spatial separation")
    return release,arrays,roles,ratio


def publish_bank(directory, profile, *, confirm=False):
    release, arrays, roles, ratio = definition(profile,confirm=confirm)
    manifest, reg = directory/"bank.json",directory/"registration.json"
    if not manifest.is_file():
        custody.publish(directory,directory.parent,release,lambda row:arrays[row["id"]],
                        resume=directory.exists())
    pin = dict(root=str(directory),sha256=file_pin(manifest)["sha256"],registration=file_pin(reg)["sha256"])
    return custody.Bank(pin,directory.parent,release),roles,ratio
