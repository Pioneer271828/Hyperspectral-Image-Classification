"""Remote E1-C1 control/runner acceptance; only optional five-arm x5-update smoke."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_runtime import ROOT,code_snapshot,external_directory,hash_file,require_frozen_code,write_json
from transfer.container_runtime import training_container_kind

BASE='2d9b7b20b1c970a43a9f5829aaf239d66c34fe1b'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit',required=True);p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--development-root',type=Path);p.add_argument('--protocol-root',type=Path);p.add_argument('--data-root',type=Path)
    p.add_argument('--gpu');p.add_argument('--legacy-e1b-root',type=Path)
    a=p.parse_args();out=external_directory(a.out_dir);sources=[a.development_root,a.protocol_root,a.data_root]
    if any(sources) and not all(sources):raise ValueError('supply all real input roots together')
    if all(sources) and not a.gpu:raise ValueError('explicit GPU UUID required')
    for root in [ROOT,*[r for r in sources if r is not None]]:
        src=root.resolve(strict=True)
        if out.is_relative_to(src) or src.is_relative_to(out):raise ValueError('output overlaps source/inputs')
    out.mkdir(parents=True,exist_ok=False);state={'status':'running','command':sys.argv,'started_unix':time.time(),'checks':[]}
    def run(name,command,extra=None,timeout=300):
        begin=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            rc=subprocess.run(command,cwd=ROOT,env={**env,**(extra or {})},stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
        state['checks'].append({'name':name,'command':command,'environment':extra or {},'returncode':rc,'seconds':time.monotonic()-begin})
        write_json(out/'manifest.json',state);assert rc==0,name+' failed; logs retained'
    try:
        assert sys.platform=='linux' and __debug__ and training_container_kind()
        code=code_snapshot();assert code['commit']==a.expected_commit and not code['dirty']
        assert subprocess.check_output(['git','rev-parse','--is-shallow-repository'],text=True).strip()=='false'
        subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],check=True)
        import torch
        assert torch.cuda.is_available();torch.set_num_threads(1)
        state.update(code=code,python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,container=training_container_kind(),
            gpu=torch.cuda.get_device_name(0),visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),full_budget_launched=False)
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','E1_DEVICE':'cpu',
            'M1_DEVICE':'cpu','M4_DEVICE':'cpu','G3_DEVICE':'cpu','GRAPH_CACHE_DEVICE':'cpu','DYNAMIC_HEAD_DEVICE':'cpu'}
        def test(pattern):return [sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v']
        for name,pattern,extra in [('controls-cpu','test_coverage_controls.py',{}),('campaign-cpu','test_e1_ablation.py',{}),
            ('full-cpu','test_*.py',{}),('controls-cuda','test_coverage_controls.py',{'E1_DEVICE':'cuda'}),
            ('campaign-cuda','test_e1_ablation.py',{'E1_DEVICE':'cuda'}),('e1b-cuda','test_coverage_integration.py',{'E1_DEVICE':'cuda'}),
            ('e1a-cuda','test_observation_coverage.py',{'E1_DEVICE':'cuda'})]:run(name,test(pattern),extra)
        from scripts.validate_m2 import cuda_cache_lifecycle
        state['vca_cuda_cache_lifecycle']=cuda_cache_lifecycle()
        if a.legacy_e1b_root:
            from model.checkpoint_io import load_model_checkpoint
            from transfer.run_provenance import verify_run_artifacts
            before={str(p):hash_file(p) for p in a.legacy_e1b_root.iterdir() if p.is_file()}
            state['legacy_e1b']=verify_run_artifacts(a.legacy_e1b_root)
            old=load_model_checkpoint(a.legacy_e1b_root/'model.pt','cpu')
            assert old.backbone.coverage_modulation.condition=='continuous'
            assert before=={p:hash_file(p) for p in before}
        if all(sources):
            runner=str(ROOT/'scripts/run_e1_ablation.py');campaign=out/'real-preflight'
            run('real-prepare',[sys.executable,runner,'prepare','--expected-commit',a.expected_commit,'--out-dir',str(campaign),
                '--development-root',str(a.development_root),'--protocol-root',str(a.protocol_root),'--data-root',str(a.data_root),'--smoke'])
            command=[sys.executable,runner,'run','--expected-commit',a.expected_commit,'--plan',str(campaign/'plan.json'),'--gpus',a.gpu]
            run('real-five-arms',command,timeout=600)
            before={str(p):hash_file(p) for p in (campaign/'cases').rglob('*') if p.is_file()}
            run('real-verified-resume',command+['--resume'],timeout=600)
            assert before=={p:hash_file(p) for p in before}
            summary=json.loads((campaign/'paired-results.json').read_text())
            assert summary['target_updates']==25 and summary['decision']=='engineering_preflight_only' and not summary['test_evaluated']
            state['real_preflight']={'summary':summary,'case_files_unchanged_after_resume':len(before)}
            # Freeze the future75000-update plan; preparing it does not run it.
            full=out/'registered-full-budget'
            run('register-full-budget',[sys.executable,runner,'prepare','--expected-commit',a.expected_commit,'--out-dir',str(full),
                '--development-root',str(a.development_root),'--protocol-root',str(a.protocol_root),'--data-root',str(a.data_root)])
            value=json.loads((full/'plan.json').read_text());assert value['updates']==5000 and not (full/'cases').exists()
            state['registered_full_plan']={'path':str(full/'plan.json'),'plan_hash':value['plan_hash'],'target_updates':75000,'launched':False}
        import ast
        names=subprocess.check_output(['git','ls-files','-z'],text=True).strip('\0').split('\0')
        for n in names:
            if n.endswith('.py'):ast.parse((ROOT/n).read_text(encoding='utf-8-sig'),filename=n)
            if n.endswith('.json'):json.loads((ROOT/n).read_text(encoding='utf-8-sig'))
        state['static_files']={ext:sum(n.endswith(ext) for n in names) for ext in ('.py','.json')}
        subprocess.run(['git','diff','--check',BASE,'HEAD'],check=True);require_frozen_code(code);state['status']='passed'
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());raise
    finally:
        state['finished_unix']=time.time();write_json(out/'manifest.json',state)
    print(json.dumps({k:v for k,v in state.items() if k!='code'},indent=2))


if __name__=='__main__':main()
