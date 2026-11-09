"""Prepare or execute the registered E1 controls; artifacts never enter checkout."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_runtime import ROOT,code_snapshot,external_directory,write_json
from transfer.container_runtime import training_container_kind
from transfer.e1_ablation import (freeze_plan,prepare_policies,verify_plan,input_identity,
    read_scene,run_case,summarize,require)
from scripts.run_m4_ablation import execute


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','run','worker','summarize'))
    p.add_argument('--expected-commit',required=True);p.add_argument('--plan',type=Path)
    p.add_argument('--out-dir',type=Path);p.add_argument('--development-root',type=Path)
    p.add_argument('--protocol-root',type=Path);p.add_argument('--data-root',type=Path)
    p.add_argument('--gpus',nargs='+');p.add_argument('--seed',type=int)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();code=code_snapshot()
    require(sys.platform=='linux' and training_container_kind() and __debug__ and not code['dirty']
        and code['commit']==a.expected_commit,'clean exact SHA in training container required')
    if a.command=='prepare':
        require(all((a.out_dir,a.development_root,a.protocol_root,a.data_root)),'frozen input/output roots required')
        out=external_directory(a.out_dir)
        for src in (a.development_root,a.protocol_root,a.data_root):
            src=src.resolve(strict=True);require(not out.is_relative_to(src) and not src.is_relative_to(out),'output overlaps inputs')
        require(not out.exists(),'existing output; no overwrite')
        inputs=input_identity(a.development_root,a.protocol_root,a.data_root);scene,record=read_scene(inputs)
        plan=prepare_policies(freeze_plan(code,inputs,out,smoke=a.smoke),scene,record)
        write_json(out/'plan.json',plan);verify_plan(plan)
        print(json.dumps({'status':'frozen','plan_hash':plan['plan_hash'],'scope':plan['scope'],
            'runs':len(plan['modes'])*len(plan['seeds']),'updates_per_run':plan['updates'],
            'evaluation_bank_hash':plan['evaluation_bank_hash']}));return
    require(a.plan is not None,'--plan required');plan=json.loads(a.plan.read_text());verify_plan(plan)
    require(plan['code']['commit']==a.expected_commit and a.plan.resolve()==Path(plan['output'])/'plan.json','plan path/commit differs')
    if a.command=='worker':
        import torch
        require(a.seed in plan['seeds'] and torch.cuda.is_available() and torch.cuda.device_count()==1,'one visible CUDA GPU and registered seed required')
        torch.set_num_threads(1);scene,record=read_scene(plan['inputs']);operations=[]
        for mode in plan['modes']:
            r=run_case(plan,scene,record,a.seed,mode,resume=a.resume)
            operations.append({'mode':mode,'operation':r['operation'],'experiment_status':r['status'],'verified_artifact_dir':r['artifact_dir']})
            write_json(Path(plan['output'])/f'seed-{a.seed}-operations.json',operations)
        return
    if a.command=='run':
        require(a.gpus is not None,'explicit GPU UUIDs required')
        result=execute(plan,a.gpus,resume=a.resume,verifier=verify_plan,summarizer=summarize,worker_script=ROOT/'scripts/run_e1_ablation.py')
    else:
        result=summarize(plan);write_json(Path(plan['output'])/'paired-results.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
