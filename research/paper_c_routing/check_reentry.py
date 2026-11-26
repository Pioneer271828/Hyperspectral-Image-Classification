"""Trap neural work while testing actual frozen CLI completed and failed reentry."""
import argparse
from contextlib import redirect_stdout,redirect_stderr
import io
import json
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--producer-root',required=True);p.add_argument('--producer',required=True)
    p.add_argument('--dependency',required=True);p.add_argument('--runs',required=True)
    p.add_argument('--g3-root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--case',default='transfer-span02');p.add_argument('--seed',type=int)
    a=p.parse_args();root=Path(a.producer_root).resolve();runs=Path(a.runs).resolve()
    suite=runs/'paper-c-routing-20261126';events=[]
    script=root/'research/paper_c_routing/run.py'
    base=['--expected-commit',a.producer,'--dependency',a.dependency,'--runs',str(runs)]
    def invoke(name,case,*,args=None,expected=None):
        calls={'neural_forwards':0,'weight_deserializations':0};text=io.StringIO()
        def forbidden_forward(*args,**kw):
            calls['neural_forwards']+=1;raise AssertionError('Unexpected neural work in reentry')
        original_load=torch.load
        def load(*args,**kw):
            calls['weight_deserializations']+=1
            if name!='orphan_probability':raise AssertionError('Unexpected weight load before reentry guard')
            return original_load(*args,**kw)
        argv=[str(script),*base,'--out',str(case),'--profile','g3_pavia_xuzhou',
              '--g3-root',a.g3_root,'--range-view','--descriptor','approximate_span',*(args or [])]
        if a.seed is not None:argv+=['--g3-seed',str(a.seed)]
        error=None;started=perf_counter()
        with patch.object(sys,'argv',argv),patch.object(torch.nn.Module,'_call_impl',forbidden_forward),\
                patch.object(torch,'load',load),redirect_stdout(text),redirect_stderr(text):
            try:runpy.run_path(str(script),run_name='__main__')
            except Exception as exc:error=f'{type(exc).__name__}: {exc}'
        if calls['neural_forwards']:raise AssertionError('Replay spent neural calls')
        if expected is None:
            if error or '"completed_reentry": true' not in text.getvalue():raise AssertionError((name,error,text.getvalue()))
        elif error is None or expected not in error:raise AssertionError((name,error))
        events.append({'case':name,'expected_error':expected,'observed_error':error,**calls,
                       'elapsed_seconds':perf_counter()-started,'stdout':text.getvalue()})
    sys.path.insert(0,str(script.parent))
    original=suite/a.case
    invoke('completed',original,args=['--resume'])
    invoke('implicit_overwrite',original,expected='Explicit exact resume required')
    invoke('wrong_contract',original,args=['--resume','--descriptor','retained_fraction'],expected='Explicit exact resume required')
    invoke('wrong_code',original,args=['--resume','--expected-commit','0'*40],expected='Frozen code identity differs')
    with tempfile.TemporaryDirectory(prefix='reentry-negative-',dir=suite) as tmp:
        temporary=Path(tmp)
        corrupted=temporary/'tampered';shutil.copytree(original,corrupted)
        evidence=next(corrupted.glob('round-*.npz'));evidence.write_bytes(evidence.read_bytes()+b'tamper')
        invoke('tampered_output',corrupted,args=['--resume'],expected='Completed evidence changed')
        orphan=temporary/'orphan';orphan.mkdir()
        for name in ('contract.json','setup.json','round-0-light.npz'):shutil.copy2(original/name,orphan/name)
        invoke('orphan_probability',orphan,args=['--resume'],expected='Uncommitted probability artifact')
    value={'status':'PASSED','producer':a.producer,'tests':events,'optimizer_updates':0,
           'total_neural_forwards':sum(x['neural_forwards'] for x in events),
           'temporary_copies_removed':True,'originals_mutated':False}
    with Path(a.out).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps({'status':'PASSED','cases':len(events),'neural_forwards':0,'optimizer_updates':0}))


if __name__=='__main__':main()
