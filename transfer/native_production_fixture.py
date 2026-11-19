"""Independent, role-separated larger-field confirmation for retained observations."""

from itertools import product

import numpy as np

from evaluation.frozen_training_stages import file_pin, require
from loadData.observation_sampling import BandMetadata
from model.research_config import digest
from transfer import development_bank as custody
from transfer.spatial_coverage_protocol import counts, release_statistics, retained_positions

SCHEMA = "wp05_wp06_native_production_fields_v1"
CLASSES = 9
BANDS = tuple(100+3*i for i in range(27))
SOURCE_BANDS = {"source-a": tuple(i for i in range(27) if i not in (7, 8, 18, 19)),
                "source-b": tuple(range(27))}
TARGET_BANDS = tuple(i for i in range(27) if i not in (7, 19))
SEEDS = (64, 512, 2048)
SUPPORT = (211, 613)
MODES = ("local", "context", "erased")
VIEWS = ("full", "central_gap", "matched_dispersed")
SOURCE_STEPS = 144
STEPS = 216
CLOCKS = (0, 72, 144, 216)
RARE = CLASSES
GATES = dict(full_positive_min_aa=.80, minimum_rare_recall=.50, erased_gap_max_aa=.23,
             ordinary_delta_recall=-.02, ordinary_delta_aa=-.005, rare_added_errors=0,
             minimum_nonempty_q=2, reference_context_center_max_aa=.25)


def layout(kind, domain):
    if kind == "source-a":
        return 30, 32
    if kind == "source-b":
        return 34, 38
    return (40, 42) if domain == "in" else (44, 50)


def raw_tile(kind, mode, domain, ordinal, shape, seed):
    """Construct observed spectra independently of released label sampling."""
    h, w = shape
    noise = np.random.default_rng([seed, ordinal, 947])
    geometry = np.random.default_rng([seed, ordinal, 219])
    if domain == "out":
        ycuts = [0, h//3+2, 2*h//3-2, h]
        xcuts = [0, w//3-2, 2*w//3+3, w]
    else:
        ycuts, xcuts = np.linspace(0,h,4,dtype=int), np.linspace(0,w,4,dtype=int)
    label_map = np.zeros((h,w), dtype=np.int16)
    centers = {}
    book = np.zeros((CLASSES, len(BANDS)), dtype=np.float32)
    signal = list(range(9,18)) if mode == "erased" else [0,1,2,3,4,5,21,22,23]
    book[:,signal] = -.65 + 2.50*np.eye(CLASSES,dtype=np.float32)
    if mode != "erased":
        # The remaining outside lines carry a weaker shared class signature.
        book[:,[6,20,24,25,26]] = .20*np.random.default_rng(446).choice([-1,1], size=(CLASSES,5))
    for (iy,ix), label in zip(product(range(3), repeat=2), geometry.permutation(CLASSES), strict=True):
        y0,y1,x0,x1 = ycuts[iy],ycuts[iy+1],xcuts[ix],xcuts[ix+1]
        label_map[y0:y1,x0:x1] = label
        ys = list(range(2*((y0+3)//2), y1-3, 4))
        xs = list(range(2*((x0+3)//2), x1-3, 4))
        choices = [(y,x) for y,x in product(ys,xs) if y>=y0+2 and x>=x0+2]
        require(choices, "Field geometry cannot reserve a center")
        centers[int(label)] = choices
    values = book[label_map]
    if mode == "context":
        for choices in centers.values():
            for y,x in choices:
                values[y:y+2,x:x+2] = 0
    if domain == "out":
        values = 1.08*values + np.linspace(-.08,.08,len(BANDS),dtype=np.float32)
    values = values + noise.normal(0, .023, size=values.shape).astype(np.float32)
    return values, centers, label_map


def definition(kind, support_seed):
    source = kind in SOURCE_BANDS
    require(kind in (*SOURCE_BANDS,"target") and (support_seed in SUPPORT if not source else support_seed == SUPPORT[0]),
            "Unknown independent synthetic roster")
    scene = "paviaU" if kind == "source-a" else "xuzhou"
    axis = SOURCE_BANDS[kind] if source else TARGET_BANDS
    seed = {"source-a": 35261, "source-b": 67709, "target": 91333}[kind]
    master = BandMetadata(f"{SCHEMA}:{kind}", tuple(BANDS[i] for i in axis),
                         tuple(float(440+16*i) for i in axis), "nm", "verified", (True,)*len(axis),
                         (12.,)*len(axis), "nm")
    core = [("support","in",mode,i) for mode,i in product(MODES,range(2))]
    core += [("calibration","in",mode,0) for mode in MODES]
    if not source:
        core += [("evaluation",domain,mode,0) for domain,mode in product(("in","out"), MODES)]
    height = ((len(core)+2)//3)*60
    width = 180
    shape = (height,width,len(axis))
    roles = {key: [] for key in ("support","calibration","evaluation")}
    blocks, rows, arrays = {}, [], {}
    for index, (role, domain, mode, repetition) in enumerate(core, 1):
        h,w = layout(kind,domain)
        y0,x0 = (index-1)//3*60,(index-1)%3*60
        field, centers, labels = raw_tile(kind,mode,domain,index,(h,w),seed)
        rng = np.random.default_rng([support_seed if role == "support" else 52453, index, seed])
        samples = []
        for klass, locations in centers.items():
            # Prefer the neutral centers and use distinct original pixels within those cells.
            candidates = [(y+dy,x+dx) for y,x in locations for dy,dx in product(range(2),repeat=2)]
            n = (1 if klass+1 == RARE else 2) if role == "support" else (2 if klass+1 == RARE else 4)
            chosen = rng.permutation(len(candidates))[:n]
            require(len(chosen)==n, "Synthetic released class denominator incomplete")
            for at in chosen:
                y,x = candidates[at]
                require(labels[y,x] == klass, "A released point escaped its planted class")
                samples.append([(y0+y)*width+x0+x, klass+1])
        box = [y0,y0+h,x0,x0+w]
        bank_role = "devtrain" if role == "support" else "devval"
        selected = np.ascontiguousarray(field[...,axis])
        rows.append(dict(id=index, role=bank_role, partition="train" if role=="support" else "val",
                    box=box,parent_tile=index,parent_box=box,samples=sorted(samples),
                    origin=dict(generator=SCHEMA,array_sha256=custody.array_hash(selected)),
                    representation="raw_fp32",blocked=None))
        roles[role].append(index)
        blocks[str(index)] = dict(mode=mode, domain=domain, purpose=role, shape=[h,w], repetition=repetition)
        arrays[index] = selected
    rows.sort(key=custody._row_key)
    plan = digest([SCHEMA,kind,shape,blocks,roles])
    statistics = release_statistics(arrays, rows, plan)
    poisoned = {row["id"]: arrays[row["id"]] if row["role"]=="devtrain"
                else np.full_like(arrays[row["id"]],np.nan) for row in rows}
    require(statistics == release_statistics(poisoned, rows, plan), "Confirmation contaminated support statistics")
    custody_plan = dict(minimum_buffer=12, generator=SCHEMA, support_seed=support_seed, roles=digest(roles))
    provenance = dict(generator=SCHEMA, source_layout=kind, unlabeled_layout_seed=seed)
    release = custody.BankRelease(scene,"synthetic",shape,CLASSES,plan,master,statistics,provenance,custody_plan,tuple(rows))
    # Check the physically retained subset actually removes every designated erased line.
    kept = {master.band_ids[i] for i in retained_positions(master,"central_gap")}
    require(not kept.intersection(BANDS[i] for i in range(9,18)), "Erased truth leaked into a retained line")
    return release, arrays, roles, blocks


def checked_roles(release, roles):
    by_id = {row["id"]: row for row in release.rows}
    require(set(roles) == {"support","calibration","evaluation"}
            and sorted([i for items in roles.values() for i in items]) == sorted(by_id),
            "Spatial roles must partition the published Bank")
    population, minimum = {}, None
    for name, items in roles.items():
        rows = [by_id[i] for i in items]
        expected = "devtrain" if name=="support" else "devval"
        require(all(row["role"] == expected for row in rows), "A spatial role crosses the Bank partition")
        population[name] = counts(rows, release.classes)
    for first,second in (("support","calibration"),("support","evaluation"),("calibration","evaluation")):
        gaps = [custody._gap(by_id[i]["parent_box"],by_id[j]["parent_box"])
                for i,j in product(roles[first],roles[second])]
        if gaps:
            minimum = min(gaps) if minimum is None else min(minimum,min(gaps))
    require(minimum >= release.custody["minimum_buffer"] and all(min(row)>0 for row in population.values() if any(row)),
            "Independent fixture contains a shared buffer or missing class")
    return dict(per_class=population, minimum_buffer=minimum)


def publish(path, kind, support_seed):
    release, arrays, roles, blocks = definition(kind,support_seed)
    report = checked_roles(release,roles)
    if not (path/"bank.json").is_file():
        custody.publish(path,path.parent,release,lambda row:arrays[row["id"]],resume=path.exists())
    pin = dict(root=str(path),sha256=file_pin(path/"bank.json")["sha256"],
               registration=file_pin(path/"registration.json")["sha256"])
    bank = custody.Bank(pin,path.parent,release)
    return bank, dict(bank=pin,roles=roles,blocks=blocks,eligibility=report)
