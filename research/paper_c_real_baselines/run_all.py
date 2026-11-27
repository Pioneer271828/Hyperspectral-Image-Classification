"""Run the frozen short package in isolated fitting stages, then development evaluation."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
from time import perf_counter

from c8_bridge import fresh,write,protocol,canonical,sha,require_learning_enabled
from evaluate import verified_stages


def main():
    require_learning_enabled()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected-commit',required=True);p.add_argument('--c7-root',required=True);p.add_argument('--inputs',required=True);p.add_argument('--plan',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=fresh(a.expected_commit,a.out);out.mkdir();began=perf_counter();cfg=protocol();env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');folder=Path(__file__).parent
    common=['--expected-commit',a.expected_commit,'--c7-root',a.c7_root,'--inputs',a.inputs,'--plan',a.plan]
    def execute(script,args,logname):
        remaining=1800-(perf_counter()-began)
        if remaining<=0:raise TimeoutError('Frozen short-run wall cap reached; no budget extension')
        with (out/logname).open('x') as log:subprocess.run([sys.executable,str(folder/script),*common,*args],check=True,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=remaining)
    for stage in ('source','target'):
        for source,target in cfg['directions']:
            args=['--stage',stage,'--direction',source,target,'--out',str(out/f'{stage}-{source}-to-{target}')]
            if stage=='target':args+=['--source-stage',str(out/f'source-{source}-to-{target}')]
            execute('train_stage.py',args,f'{stage}-{source}-to-{target}.log')
    stages=verified_stages(out,a.expected_commit);write(out/'fitting-complete.json',{'source':a.expected_commit,'protocol_sha256':sha(canonical(cfg)),'all_fitting_complete_before_any_development_array':True,'optimizer_updates':sum(m['optimizer_updates'] for _,m in stages),'linear_endpoints':48,'centroid_endpoints':4,'fitting_wall_seconds':perf_counter()-began})
    execute('evaluate.py',['--experiment',str(out),'--out',str(out/'evaluation01')],'evaluation01.log')
    write(out/'run-complete.json',{'source':a.expected_commit,'protocol_sha256':sha(canonical(cfg)),'producer_optimizer_updates':3840,'wall_seconds':perf_counter()-began,'complete':True,'no_long_training':True,'new_label_memberships':0,'query_reads':0});print(__import__('json').dumps({'complete':True,'producer_optimizer_updates':3840,'linear_endpoints':48,'centroid_endpoints':4,'wall_seconds':perf_counter()-began}))


if __name__=='__main__':main()
