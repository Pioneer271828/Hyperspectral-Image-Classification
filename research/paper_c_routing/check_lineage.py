"""Hash-only original G3 lineage and released-support/context exposure audit."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',required=True)
    p.add_argument('--development',required=True);p.add_argument('--inventory',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();profiles=json.loads((Path(__file__).parent/'g3-profiles.json').read_text())
    development=Path(a.development);dev=json.loads(development.read_text());rec=dev['record']
    inventory=json.loads(Path(a.inventory).read_text());cases=[]
    validation=dict(zip(rec['validation_support']['indices'],rec['validation_support']['labels']))
    width=dev['plan']['shape'][1]
    methods=('source_pretrained_full_finetune','source_pretrained_head_only','target_scratch_full')
    for seed,method in ((s,m) for s in profiles['profiles'] for m in methods):
        files=dict(profiles['profiles'][seed]);model_name='model.pt'
        if method!='source_pretrained_full_finetune':
            files.pop('model.pt');files.pop('model.pt.json')
            files.update(profiles['variants'][seed][method]);model_name=method+'/model.pt'
        root=Path(a.models)/f'g3-p2-s{seed}'
        pins={name:sha(root/name) for name in (*files,'support-10.json')}
        if pins!={**files,'support-10.json':profiles['support_sha256']}:raise ValueError('Original file identity differs')
        task=json.loads((root/'frozen-task.json').read_text());meta=json.loads((root/(model_name+'.json')).read_text())
        support=json.loads((root/'support-10.json').read_text());cfg=task['config'];training=meta['training_config']
        if (cfg['source_dataset']!='paviaU' or cfg['target_dataset']!='xuzhou' or cfg['model_seed']!=int(seed)
                or cfg['source_updates']!=10000 or cfg['target_updates']!=1000 or cfg['target_validation_labels']!=0
                or cfg['target_selection']!='fixed_final_step' or training['optimizer_steps']!=1000
                or training['method']!=method
                or training['task_hash']!=task['task_hash'] or training['support_hash']!=support['support_hash']
                or task['target_plan_hash']!=dev['identity']['plan_hash']
                or task['target_identity']['image_sha256']!=inventory['images']['xuzhou']['sha256']):
            raise ValueError('Original training/data ancestry differs')
        original=set(support['indices']);overlap=original&set(validation)
        context=[];class_parents={str(c):set() for c in range(1,10)}
        for tile in rec['roles']:
            if tile['role']!='devval':continue
            y0,y1,x0,x1=tile['box']
            inside=lambda idx:y0<=idx//width<y1 and x0<=idx%width<x1
            seen=sorted(i for i in original if inside(i))
            labels=[label for idx,label in validation.items() if inside(idx)]
            for label in labels:class_parents[str(label)].add(tile['id'])
            context.append({'tile':tile['id'],'original_plan_parent':tile['id'],
                            'original_training_points_inside':len(seen),'current_devval_points':len(labels)})
        cases.append({'seed':int(seed),'method':method,'pins':pins,'task_hash':task['task_hash'],'support_hash':support['support_hash'],
                      'source_updates':0 if method=='target_scratch_full' else 10000,'target_updates':1000,'original_target_K':10,'original_target_V':0,
                      'selected_target_optimizer_step':training['optimizer_steps'],
                      'checkpoint_statistics_hash':training['statistics_hash'],
                      'released_statistics_hash':dev['identity']['statistics_hash'],
                      'released_normalization_matches_checkpoint':training['statistics_hash']==dev['identity']['statistics_hash'],
                      'target_image_matches_previous_inventory':True,'target_plan_matches_development':True,
                      'original_support_labels':len(original),'released_validation_labels':len(validation),
                      'overlap_training_validation_point_ids':len(overlap),'context_exposure':context,
                      'validation_parent_count_by_class':{c:len(v) for c,v in class_parents.items()},
                      'independent_confirmation':False})
    value={'status':'PASSED','cases':cases,'development_sha256':sha(development),'prior_inventory_sha256':sha(a.inventory),
           'new_raw_gt_reads':0,'new_query_reads':0,'new_neural_forwards':0,'new_optimizer_updates':0,
           'interpretation':'Zero label-ID overlap cannot establish independence: original training-region normalization/context and historical development use remain exposed. No query opened.'}
    with Path(a.out).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps({'status':'PASSED','cases':len(cases),'label_overlaps':[x['overlap_training_validation_point_ids'] for x in cases]}))


if __name__=='__main__':main()
