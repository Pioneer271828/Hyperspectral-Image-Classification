"""Source-only data/model-domain preflight; no training or target payload access."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from source_data_context import load_source_context
from joint_training import load_plan
from joint_baseline import NAMES
from research_runtime import code_snapshot,require_frozen_code,external_directory,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('expected-commit','prepared-dir','data-root','out-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--held-out',choices=NAMES,required=True);p.add_argument('--sources',nargs='+',choices=NAMES)
    a=p.parse_args();code=code_snapshot()
    if sys.platform!='linux' or code['dirty'] or code['commit']!=a.expected_commit:
        raise RuntimeError('Remote clean expected published commit required')
    plan=load_plan('configs/experiments/base97_training.json')
    context=load_source_context(a.prepared_dir,a.data_root,manifest_sha256=plan['prepared_manifest_sha256'],
                                held_out=a.held_out,consumer=code,sources=a.sources)
    model=context.build_model()
    if set(model.heads)!=set(context.sources) or any(a.held_out in n.split('.') for n in model.state_dict()):
        raise ValueError('Held target entered model heads/domains')
    require_frozen_code(code);out=external_directory(a.out_dir);out.mkdir(exist_ok=False)
    write_json(out/'source-context.json',{'status':'passed','code':code,**context.record,
        'model_parameter_count':sum(p.numel() for p in model.parameters()),'model_inspection_device':'cpu',
        'training_budget_executed':False,'source_only_training_runner_implemented':False})
    print(json.dumps({'status':'passed','sources':list(context.sources),'held_out':a.held_out,
                      'target_payload_accessed':False,'optimizer_updates':0}),flush=True)


if __name__=='__main__':main()
