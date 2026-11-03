"""BN-ALIGN3 synthetic trajectory isolation gate; never reads real datasets."""
import argparse
from pathlib import Path
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'tests')]
import torch
from test_joint_baseline import tiny_items
from joint_baseline import load_config
from joint_calibrated_selection import calibrated_settings, validation_candidate
from joint_training import (_rng_state, _tree_fingerprint, load_plan, options, run_training)
from loadData.dataset_registry import file_hash
from model.research_config import ResearchConfig
from research_runtime import code_snapshot, external_directory, require_frozen_code, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit', required=True); p.add_argument('--out-dir', required=True)
    p.add_argument('--device', choices=['cpu','cuda'], required=True); a = p.parse_args()
    code = code_snapshot()
    if sys.platform != 'linux' or code['dirty'] or code['commit'] != a.expected_commit:
        raise RuntimeError('Remote clean exact published SHA required')
    out = external_directory(a.out_dir); out.mkdir(parents=True, exist_ok=False)
    try:
        baseline = load_config('configs/experiments/base97_joint.json')
        plan = load_plan('configs/experiments/base97_training.json'); records = []
        for seed in (64,512,2048):
            settings = options(plan,'engineering',seed,baseline); states = []
            items = tiny_items()
            for item in items.values(): item['zeros'] = torch.zeros(item['height']*item['width'])
            for arm, opts in [('raw',settings), ('calibrated',calibrated_settings(settings))]:
                folder = out/(str(seed)+'-'+arm)
                run_training(opts,items,folder,{'scope':'synthetic_only'},code,device=a.device)
                import json
                progress = json.loads((folder/'progress.json').read_text())
                states.append(torch.load(folder/progress['state_file'],map_location='cpu',weights_only=True))
            compared = {}
            for key in ('model','optimizer','rng','trace'):
                left,right = (_tree_fingerprint(s[key]) for s in states)
                if left != right: raise ValueError('Calibration changes raw training: '+key)
                compared[key] = left
            model = ResearchConfig.from_dict(settings['config']).build_model({n:i['class_num'] for n,i in items.items()}).to(a.device).eval()
            before_state = _tree_fingerprint(model.state_dict()); before_rng = _tree_fingerprint(_rng_state(torch.device(a.device)))
            validation_candidate(model,items,out/(str(seed)+'-probe.pt'))
            if (before_state != _tree_fingerprint(model.state_dict())
                or before_rng != _tree_fingerprint(_rng_state(torch.device(a.device)))):
                raise ValueError('Calibration changed source state or Python/NumPy/CPU/CUDA RNG')
            records.append({'seed':seed,'matched':compared})
        require_frozen_code(code)
        files = {str(f.relative_to(out)):file_hash(f) for f in out.rglob('*') if f.is_file()}
        write_json(out/'receipt.json',{'status':'passed','code':code,'device':a.device,'seeds':[64,512,2048],
            'raw_training_unchanged':True,'rng_unchanged':True,'real_optimizer_updates':0,
            'synthetic_updates':72,'records':records,'artifacts':files})
        print('REUSE_GATE_PASSED',a.device,flush=True)
    except Exception as error:
        write_json(out/'failure.json',{'error':str(error),'type':type(error).__name__,'code':code});raise


if __name__ == '__main__': main()
