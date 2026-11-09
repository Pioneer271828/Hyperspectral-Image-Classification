"""Descriptive label audit; one fixed label-blind train holdout, no training."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from loadData.spatial_tiles import tile_slice, validate_tile_plan
from research_runtime import code_snapshot, external_directory, hash_file, require_frozen_code, write_json
from train_cross_sensor import read_protocol


def audit(labels,plan,classes,support):
    validate_tile_plan(plan)
    labels=np.asarray(labels)
    if list(labels.shape)!=plan['shape'] or not np.isfinite(labels).all() or not np.equal(labels,np.floor(labels)).all() or not np.isin(labels,range(classes+1)).all():
        raise ValueError('Invalid label image')
    train=sorted(t['id'] for t in plan['tiles'] if t['partition']=='train')
    held=set(train[::5]) # Fixed before inspecting class counts, no fallback/search.
    regions=np.full(labels.shape,'guard',dtype='<U8')
    dev=np.full(labels.shape,'unused',dtype='<U8')
    for tile in plan['tiles']:
        regions[tile_slice(tile)]=tile['partition']
        if tile['partition']=='train':
            dev[tile_slice(tile)]='devval' if tile['id'] in held else 'devtrain'
    rows=[]
    for cls in range(1,classes+1):
        mask=labels==cls;y,x=np.nonzero(mask)
        row={'class':cls,'full':int(mask.sum()),
             'bbox_y0_y1_x0_x1':None if not len(y) else [int(y.min()),int(y.max()+1),int(x.min()),int(x.max()+1)],
             **{p:int((mask&(regions==p)).sum()) for p in ('train','val','test','guard')},
             **{p:int((mask&(dev==p)).sum()) for p in ('devtrain','devval')}}
        assert row['full']==sum(row[p] for p in ('train','val','test','guard'))
        assert row['train']==row['devtrain']+row['devval']
        rows.append(row)
    return {'rows':rows,'fixed_rule':'sorted original train IDs, positions 0,5,10,... held out',
            'heldout_tile_ids':sorted(held),'original_support_in_holdout':
                sum(dev.flat[i]=='devval' for i in support['indices']),
            'feasible_10_train_5_val_each':all(r['devtrain']>=10 and r['devval']>=5 for r in rows),
            'scope':'candidate counts only, not a validated training protocol'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol-root','data-root','out-dir'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-commit',required=True)
    args=p.parse_args();code=code_snapshot()
    if sys.platform!='linux' or not Path('/.dockerenv').exists() or sys.flags.optimize or code['dirty'] or code['commit']!=args.expected_commit:
        p.error('Require clean exact SHA in container')
    out=external_directory(args.out_dir)
    for path in (args.protocol_root.resolve(),args.data_root.resolve()):
        if out==path or out in path.parents or path in out.parents:p.error('Outputs overlap inputs')
    task,plans,support,files=read_protocol(args.protocol_root)
    identity=task['target_identity'];data=args.data_root.resolve(strict=True)
    label=(data/identity['label_path']).resolve(strict=True)
    assert label.is_relative_to(data) and hash_file(label)==identity['label_sha256']
    from scipy.io import loadmat
    from loadData.dataset_registry import DATASETS
    key=DATASETS[identity['name']][1]+'_gt'
    result=audit(loadmat(label,variable_names=[key])[key],plans['target'],identity['class_count'],support)
    require_frozen_code(code)
    assert hash_file(label)==identity['label_sha256'] and all(hash_file(k)==v for k,v in files.items())
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'coverage.json',{'status':'passed','code':code,'label_sha256':identity['label_sha256'],
        'disclosure':'Full target labels read for descriptive audit including test counts; no model selection or scoring',**result})
    print(json.dumps(result))


if __name__=='__main__':main()
