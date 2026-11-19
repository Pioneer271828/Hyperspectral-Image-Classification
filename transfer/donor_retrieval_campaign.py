"""Run a preregistered WP06/WP08 campaign and preserve every sealed training arm."""

from itertools import product
from pathlib import Path
import os
from tempfile import TemporaryDirectory
from time import perf_counter
import resource

import torch

from evaluation.graph_edge_intervention import delete_support
from research_runtime import require_frozen_code
from scripts.run_frozen_band_region import _read, _write, check, pin
from scripts.run_frozen_mechanisms import case_lock
from transfer.donor_retrieval_graph import ARMS, DIRECTIONS, LAYOUTS, PREFIX, ROLES, SEEDS, UPDATES, exact_deletions
from transfer.donor_retrieval_training import BATCH, RATE, VALIDATION, Data, Run, scientific_state

PHASES = ("full", "prefix", "resume", "completed")
CASES = {f"{layout}-{direction}-{seed}":(seed,layout,direction)
         for layout,direction,seed in product(LAYOUTS,DIRECTIONS,SEEDS)}
SCHEMA = "wp06_wp08_donor_retrieval_v1"


def plan(code):
    return dict(schema=SCHEMA, producer=code["commit"], source_hash=code["source_hash"],
                cases=list(CASES), arms=list(ARMS), split_sizes={k:v[1] for k,v in ROLES.items()},
                steps=UPDATES, batch=BATCH, prefix=PREFIX, lr=RATE, validate=list(VALIDATION),
                method="selected sparse support with training-label-loss score-function update",
                fixed_parent="learned terminal scorer pulldown before confirmation",
                criteria=dict(oracle_balanced_accuracy=.85, learned_retrieval=.75, random_ba_delta=.10,
                              required_pairs=9, maximum_null_ba=.65,
                              degree_only_equivalent_per_graph=True),
                real_updates=0, raw_gt_loaded=False, query_accessed=False)


def registration(root, code, expected):
    record = pin(root/"registration.json",expected)
    check(_read(record)==plan(code),"Donor campaign registration differs")
    return record


def register(root, code):
    root.mkdir()
    return _write(root/"registration.json",plan(code))


def read_state(path):
    return torch.load(path,map_location="cpu",weights_only=True)


def verify_state(path, expected, digest):
    value = read_state(Path(pin(path,expected)["path"]))
    check(value["state_sha256"]==digest and scientific_state(value)==digest,"Frozen donor state differs")
    return value


def stage(path, data, name, scope, phase, parent):
    starts = dict(full=0,prefix=0,resume=PREFIX,completed=UPDATES)
    stop = PREFIX if phase=="prefix" else UPDATES
    start = starts[phase]
    run = Run(data,name,scope=scope,parent=parent)
    if start:
        previous = path/("prefix" if phase=="resume" else "resume")
        meta = _read(pin(previous/"receipt.json"))
        checkpoint = verify_state(previous/"state.pt",meta["checkpoint"],meta["state"])
        run.restore(checkpoint,expected_cursor=start)
    else:
        run.validate()
    while run.cursor<stop:
        run.update()
        if run.cursor in VALIDATION:
            run.validate()
    destination = path/phase
    with TemporaryDirectory(dir=path,prefix=f".{phase}-") as temp:
        directory = Path(temp)/"sealed"
        directory.mkdir()
        value = run.pack()
        torch.save(value,directory/"state.pt")
        artifact = _write(directory/"receipt.json",
                 dict(phase=phase, cursor=stop, performed=stop-start, scope=scope, roster=data.identities,
                      initial=run.initial, parent=parent, checkpoint=pin(directory/"state.pt")["sha256"],
                      state=value["state_sha256"], timing=value["cost"]))
        os.rename(directory,destination)
    return artifact["sha256"]


def all_phases(path, data, name, scope, parent):
    path.mkdir(exist_ok=True)
    receipts = {}
    for phase in PHASES:
        latest = path/phase/"receipt.json"
        if latest.exists():
            saved = pin(latest)
            meta = _read(saved)
            check(meta["scope"]==scope and meta["roster"]==data.identities and meta["parent"]==parent,
                  "Existing staged run belongs to a foreign campaign")
            verify_state(latest.with_name("state.pt"),meta["checkpoint"],meta["state"])
            receipts[phase]=saved["sha256"]
        else:
            receipts[phase]=stage(path,data,name,scope,phase,parent)
    finish = {}
    for phase in ("full","resume","completed"):
        meta = _read(pin(path/phase/"receipt.json",receipts[phase]))
        finish[phase]=verify_state(path/phase/"state.pt",meta["checkpoint"],meta["state"])
    check(len({item["state_sha256"] for item in finish.values()})==1,
          "Uninterrupted, resumed and completed graph states differ")
    return receipts, finish["full"]


def cut(graph, known, choice, matrix):
    return delete_support(matrix,[8*i+j for i,j in graph.pairs(choice) if i!=j])


def reverse(graph, known, choice, matrix):
    return graph.support.render(scores=graph.message_scores,support=sorted((j,i) for i,j in graph.pairs(choice)))


def candidate_report(data):
    designs, mats = {}, {}
    start = perf_counter()
    for graph,known in data.rows["confirmation"]:
        graphs, audit = exact_deletions(graph,known)
        mats[graph.id] = graphs
        designs[graph.id] = audit
    return mats, designs, perf_counter()-start


def run_one(root, code, scope, name):
    seed,layout,direction = CASES[name]
    path = root/name
    path.mkdir(exist_ok=True)
    with case_lock(path/".lock"):
        if (path/"receipt.json").exists():
            record = _read(pin(path/"receipt.json"))
            check(record["scope"]==scope and record["case"]==name,"Foreign completed donor receipt")
            return False
        started = perf_counter()
        data = Data(seed,layout,direction)
        parent = None
        arms, states = {}, {}
        # No confirmation is read by fitting or used to choose a frozen parent.
        for arm in ARMS:
            dependency = parent if arm.startswith("fixed_") else None
            receipts, end = all_phases(path/arm,data,arm,scope,dependency)
            states[arm] = end
            arms[arm] = dict(phases=receipts, state=end["state_sha256"], parent=dependency)
            if arm=="learned":
                parent = dict(state=end["state_sha256"],route=end["model"]["route"].tolist())
        graphs, designs, match_seconds = candidate_report(data)
        probabilities = {}
        for arm,end in states.items():
            runner = Run(data,arm,scope=scope,parent=arms[arm]["parent"])
            runner.model.load_state_dict(end["selected"]["model"],strict=True)
            reports = {"chosen":runner.evaluate("confirmation")}
            if arm in ("learned","oracle","fixed_final"):
                reports["cut"] = runner.evaluate("confirmation",intervention=cut)
                reports["reverse"] = runner.evaluate("confirmation",intervention=reverse)
            if arm in ("learned","oracle"):
                for variant in next(iter(graphs.values())):
                    reports[f"matched:{variant}"] = runner.evaluate("confirmation",
                        intervention=lambda graph, _, _choice, _matrix: graphs[graph.id][variant])
            runner.model.load_state_dict(end["model"],strict=True)
            reports["terminal"] = runner.evaluate("confirmation")
            arms[arm].update(initial=runner.initial, chosen=end["selected"]["step"], validation=end["validation"],
                             trajectory=end["trajectory"], endpoint_times=dict(runner.cost))
            probabilities[arm]=reports
        check(len({item["initial"] for item in arms.values()})==1,"Initialization differs across paired runs")
        require_frozen_code(code)
        _write(path/"receipt.json",dict(case=name,scope=scope, seed=seed, layout=layout, direction=direction,
               roster=data.identities, arms=arms, evaluation=probabilities, independent_matching=designs,
               candidate_costs={g.id:dict(g.cost) for rows in data.rows.values() for g,_ in rows},
               match_seconds=match_seconds, elapsed_seconds=perf_counter()-started,
               peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               real_updates=0, query_accessed=False, raw_gt_loaded=False))
    return True


def run(root,code,expected,selector):
    confirmed = registration(root,code,expected)["sha256"]
    cases = [name for name in CASES if selector=="all" or name==selector or name.rsplit("-",1)[0]==selector]
    check(bool(cases),"Unknown donor case")
    for name in cases:
        status = "completed" if run_one(root,code,confirmed,name) else "reused"
        print(f"{status} {name}",flush=True)
