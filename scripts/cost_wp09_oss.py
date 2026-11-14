"""Matched single-CPU cost for optional Tent and OpenOOD on authorized fixtures."""
from argparse import ArgumentParser
import json
from tempfile import TemporaryDirectory
from pathlib import Path
import resource
from statistics import median, quantiles
import subprocess
import sys
from time import perf_counter_ns, process_time_ns

import numpy as np
import torch

from evaluate_tta import _Session
from evaluation.open_set import OpenSetScorer
from evaluation.wp09_upstream import OpenOODReference, tent_reference
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json
from transfer.online_tta_run import load_checkpoint, predict, restore
from transfer.online_tta_stream import StreamRegistration
from transfer.wp09_tent import TentEntropySession


def measure(fn):
    started, cpu = perf_counter_ns(), process_time_ns()
    value = fn()
    return value, dict(wall_ms=(perf_counter_ns()-started)/1e6, cpu_ms=(process_time_ns()-cpu)/1e6)


def ab(arms, reset=lambda: None):
    order = list(arms)
    for _ in range(3):
        for run in arms.values():
            reset()
            run()
    samples = []
    for i in range(12):
        seq = order if i % 2 == 0 else order[::-1]
        row = {}
        for item in seq:
            reset()
            _, row[item] = measure(arms[item])
        samples.append(dict(order=seq, timing=row))
    summary = {}
    for key in arms:
        wall = [row["timing"][key]["wall_ms"] for row in samples]
        summary[key] = dict(p50=median(wall), p95=quantiles(wall,n=100,method="inclusive")[94])
    return dict(samples=samples, warmups=3, summary=summary)


def worker(mode, ref):
    if mode=="tent":
        result, duration = measure(lambda:tent_reference(ref))
        files = result[1]["files"]
    else:
        result, duration = measure(lambda:OpenOODReference(ref))
        files = result.evidence["files"]
    return dict(source_bytes=sum((Path(ref)/path).stat().st_size for path in files),
                attestation_and_import=duration, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    if len(sys.argv)==4 and sys.argv[1]=="--worker":
        print(json.dumps(worker(*sys.argv[2:])))
        return
    p=ArgumentParser(description=__doc__)
    for name in ("tent","openood","tent-fixture","ood-fixture","out"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--expected-commit",required=True)
    args=p.parse_args()
    code=code_snapshot()
    if code["commit"]!=args.expected_commit or torch.get_num_threads()!=1:
        raise ValueError("Exact clean single CPU source required")
    require_frozen_code(code)
    from transfer.registered_development import _pin
    loc=args.tent_fixture/"9"
    reg=StreamRegistration(_pin(loc/"affine-continual-original.json"))
    source=reg.source_model()
    settings={k:v for k,v in reg.settings.items() if k!="reset"}
    upstream,_=tent_reference(args.tent)
    sessions=dict(native=_Session(source,**settings),tent=TentEntropySession(source,entropy=upstream.softmax_entropy,**settings))
    frame=reg.frames[0]
    load_observation=lambda:reg.observation(frame["observation"])
    base=loc/"affine-native/frame-000001"
    event=json.loads((base/"event.json").read_text())
    checkpoint,t_load=measure(lambda:load_checkpoint(base/"checkpoint.pt",event["checkpoint"]))
    path=base/"checkpoint.pt"
    def reset():
        for s in sessions.values():
            restore(s,checkpoint)
    from transfer.online_tta_run import _checkpoint, _write
    destination = external_directory(args.out)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        external_write=Path(tmp)/"state.pt"
        def save(s):
            if external_write.exists():
                external_write.unlink()
            packet=_checkpoint(s,1,frame,reg.pin["sha256"])
            _write(external_write,lambda stream:torch.save(packet,stream))

        def full(s):
            observation=load_observation()
            predict(s,observation,9)
            s.adapt(observation)
            save(s)

        compare=ab({key:lambda s=s:full(s) for key,s in sessions.items()},reset)
        restore_cost=ab({key:lambda s=s:restore(s,load_checkpoint(path,event["checkpoint"])) for key,s in sessions.items()})
        separate_write=ab({key:lambda s=s:save(s) for key,s in sessions.items()})
    ref=OpenOODReference(args.openood)
    f=json.loads((args.ood_fixture/"openood/frozen/freeze.json").read_text())
    with np.load(args.ood_fixture/"openood/production/evaluation.npz",allow_pickle=False) as raw:
        logits={k:raw[k].copy() for k in ("logits_fused","logits_graph","logits_cnn")}
    pieces=[(OpenSetScorer(row["scorer"],row["scorer_sha256"]), logits[row["array_key"]])
            for row in f["rows"] if row["method"]!="prototype_squared_distance"]
    def native():
        return [s.score(x,input_kind="raw_logits") for s,x in pieces]
    def compared():
        return [ref.compare_logits(s,x,input_kind="raw_logits",column_ids=[2,7,19]) for s,x in pieces]
    raw=torch.from_numpy(logits["logits_fused"]).double()
    def entropy_native():
        p=raw.softmax(1)
        return -(p*p.clamp_min(1e-8).log()).sum(1)
    t_primitive=ab(dict(native=entropy_native,
                        tent=lambda:upstream.softmax_entropy(raw)))
    reference=ab(dict(native=native,openood=compared))
    from evaluation.open_set import _labels
    from evaluation.wp09_openood import independent
    from evaluation.registered_open_set import _load
    from transfer.registered_development import _read
    v, _, _, _, target, _ = _load(f["registration"])
    labels=_labels(json.loads(_read(v["labels"]))["labels"],len(target["sample_ids"]))
    columns=[c["id"] for c in v["known"]]
    def scoring(use_oracle):
        result=[]
        for row in f["rows"]:
            scorer=OpenSetScorer(row["scorer"],row["scorer_sha256"])
            x=target[row["array_key"]]
            if use_oracle and row["method"]!="prototype_squared_distance":
                ref.compare_logits(scorer,x,input_kind="raw_logits",column_ids=columns)
            result.append(independent(scorer,row,labels,x))
        return result
    full_scoring=ab(dict(native=lambda:scoring(False),openood=lambda:scoring(True)))
    cold={}
    for stage, path_ in (("tent",args.tent),("openood",args.openood)):
        rows=[]
        for _ in range(3):
            proc, elapsed=measure(lambda: subprocess.run([sys.executable,"-m","scripts.cost_wp09_oss","--worker",
                         stage,str(path_)],capture_output=True,text=True,check=True))
            rows.append(dict(process=elapsed,worker=json.loads(proc.stdout),stderr=proc.stderr))
        cold[stage]=rows
    require_frozen_code(code)
    write_json(external_directory(args.out),dict(commit=code["commit"], synthetic=True, optimization_fixture="9_class_affine",
        full_online_observation_predict_adapt_and_serialize=compare, checkpoint_read_and_restore=restore_cost,
        checkpoint_write=separate_write, observation_load=measure(load_observation)[1], serialized_checkpoint_bytes=path.stat().st_size,
        checkpoint_load=t_load, entropy_primitive=t_primitive, six_logits_scores_and_equivalence_check=reference,
        complete_score_and_metrics=full_scoring, worker_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cold_processes=cold, active_fixture_sgd=30, real_optimizer_updates=0, query=0))


if __name__=="__main__":
    main()
