"""Prepare the first frozen G3 task without training or scoring a model."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    import copy
    import numpy as np
    from loadData.dataset_registry import inspect_dataset,file_hash
    from loadData.spatial_tiles import make_tile_plan
    from loadData.transfer_protocol import make_support,validate_support,freeze_task,validate_config,target_labels_for_tile
    from research_runtime import code_snapshot,external_directory,require_frozen_code,write_json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/protocols/paviau_to_xuzhou_g3.json')
    p.add_argument('--expected-commit',required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args();code=code_snapshot()
    if sys.platform!='linux' or not Path('/.dockerenv').exists() or sys.flags.optimize or code['dirty'] or code['commit']!=args.expected_commit:
        p.error('Use training container with clean exact SHA and assertions')
    config=json.loads(args.config.read_text(encoding='utf-8-sig'));validate_config(config)
    out=external_directory(args.out_dir);out.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','commit':code['commit'],'branch':code['branch'],'source_hash':code['source_hash'],
              'command':[sys.executable,*sys.argv],'config_file_sha256':file_hash(args.config)}
    write_json(out/'manifest.json',manifest)
    try:
        source,_=inspect_dataset(args.data_root,config['source_dataset']);target,labels=inspect_dataset(args.data_root,config['target_dataset'])
        plans={name:make_tile_plan(tuple(identity['shape'][:2]),config['tile_size'],config['gap'],config['split_seed'])
               for name,identity in [('source',source),('target',target)]}
        supports={}
        for budget in config['support_budgets_per_class']:
            record=make_support(labels,plans['target'],list(range(1,target['class_count']+1)),budget,config['support_seed'])
            validate_support(record,labels,plans['target']);supports[str(budget)]=record
            write_json(out/f'support-{budget}.json',record)
        assert set(supports['1']['indices'])<=set(supports['5']['indices'])<=set(supports['10']['indices'])
        readme=args.data_root/'Xuzhou/Xuzhou_Readme.txt'
        evidence={'target_sensor_readme_path':'Xuzhou/Xuzhou_Readme.txt','target_sensor_readme_sha256':file_hash(readme),
                  'target_sensor_readme_mentions_hyspex':'HYSPEX' in readme.read_text(errors='replace').upper(),
                  'source_sensor_family_registry':source['sensor_family'],'target_sensor_family_registry':target['sensor_family'],
                  'scope':'task-local bundled evidence; registry is not overwritten, not independent sensor calibration'}
        task=freeze_task(config,source,target,plans['source'],plans['target'],supports['10'],evidence)
        released=0
        for tile in plans['target']['tiles']:
            if tile['partition']=='train':
                released+=int((target_labels_for_tile(labels,plans['target'],supports['10'],tile['id'],phase='adapt')>0).sum())
        assert released==90
        poisoned=labels.copy()
        from loadData.spatial_tiles import tile_slice
        for tile in plans['target']['tiles']:
            if tile['partition']!='train':poisoned[tile_slice(tile)]=-999
        assert make_support(poisoned,plans['target'],list(range(1,target['class_count']+1)),10,config['support_seed'])==supports['10']
        test_tile=next(t for t in plans['target']['tiles'] if t['partition']=='test')
        try:target_labels_for_tile(labels,plans['target'],supports['10'],test_tile['id'],phase='adapt')
        except ValueError:pass
        else:raise AssertionError('Adaptation exposed test labels')
        for name,plan in plans.items():write_json(out/f'{name}-plan.json',plan)
        write_json(out/'task.json',task);write_json(out/'resolved-config.json',config)
        assert inspect_dataset(args.data_root,source['name'])[0]==source and inspect_dataset(args.data_root,target['name'])[0]==target
        require_frozen_code(code)
        manifest.update(status='passed',task_hash=task['task_hash'],target_support_labels=released,target_validation_labels=0,
                        budgets={k:len(v['indices']) for k,v in supports.items()},nested_supports=True,
                        heldout_label_values_do_not_affect_support=True,optimizer_steps=0,accuracy_evaluated=False)
        print(json.dumps(manifest),flush=True)
    except BaseException:
        import traceback
        manifest.update(status='failed',error=traceback.format_exc());raise
    finally:write_json(out/'manifest.json',manifest)

if __name__=='__main__':main()
