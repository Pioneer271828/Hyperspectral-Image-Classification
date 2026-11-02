"""Read released support, sampler traces and training/source-val records only."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json


def exposure(plan, support, seed, steps):
    counts = {}
    width = plan['shape'][1]
    for tile in plan['tiles']:
        if tile['partition'] != 'train':
            continue
        y0,y1,x0,x1 = tile['box']
        values = [v for i,v in zip(support['indices'],support['labels'])
                  if y0 <= i//width < y1 and x0 <= i%width < x1]
        if values:
            counts[tile['id']] = [values.count(c) for c in support['classes']]
    if not counts:
        raise ValueError('No support-bearing training tiles')
    ids = list(counts)
    rng = np.random.default_rng(seed)
    sampled = [ids[int(rng.integers(len(ids)))] for _ in range(steps)]
    return {'support_tiles':len(ids), 'sampled_tiles':sampled,
            'classes':[{'class':c, 'support_tiles':sum(v[j]>0 for v in counts.values()),
                        'updates_present':sum(counts[t][j]>0 for t in sampled),
                        'label_exposures':sum(counts[t][j] for t in sampled)}
                       for j,c in enumerate(support['classes'])]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--campaign-root',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    code=code_snapshot()
    if sys.platform!='linux' or not Path('/.dockerenv').exists() or code['dirty'] or code['commit']!=args.expected_commit or sys.flags.optimize:
        parser.error('Require container and clean exact SHA with assertions')
    out=external_directory(args.out_dir)
    campaign_root=args.campaign_root.resolve(strict=True)
    if out==campaign_root or out in campaign_root.parents or campaign_root in out.parents:
        parser.error('Keep diagnostic outputs separate from campaign')
    inputs={}
    def read(path):
        inputs[str(path)]=hash_file(path)
        return json.loads(path.read_text())
    campaign=read(campaign_root/'campaign.json')
    if campaign['status']!='passed':
        parser.error('Only diagnose a completed campaign')
    records=[]
    for seed in campaign['seeds']:
        root=campaign_root/f'run-s{seed}'
        task=read(root/'frozen-task.json')
        protocol=campaign_root/f'protocol-s{seed}'
        support=read(protocol/'support-10.json')
        plan=read(protocol/'target-plan.json')
        source=read(root/'source/result.json')
        seen=exposure(plan,support,task['config']['target_sampler_seed'],task['config']['target_updates'])
        methods={}
        for method in task['config']['methods']:
            result=read(root/method/'result.json')
            assert all(seen['sampled_tiles'][e['step']-1]==e['tile_id'] for e in result['trace'])
            loss=[e['loss']['total'] for e in result['trace']]
            methods[method]={'first':loss[0], 'last':loss[-1],
                             'first_20_logged_mean':statistics.mean(loss[:20]),
                             'last_20_logged_mean':statistics.mean(loss[-20:])}
        history=source['history']
        records.append({'seed':seed,'exposure':seen['classes'],'support_tiles':seen['support_tiles'],
                        'training_loss':methods,'source_selected_step':source['selected_step'],
                        'source_best_val_nll':min(x['val']['nll'] for x in history),
                        'source_final_val_nll':history[-1]['val']['nll']})
    require_frozen_code(code)
    assert all(hash_file(path)==value for path,value in inputs.items())
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'diagnosis.json',{'status':'passed','code':code,'records':records,'inputs':inputs,
        'target_test_read':False,'new_labels_read':False,
        'limitations':'Logged losses use different sampled tiles; trends are not convergence or validation evidence.'})


if __name__=='__main__':
    main()
