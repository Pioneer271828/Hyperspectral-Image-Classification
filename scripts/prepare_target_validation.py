"""Freeze a new target-validation protocol; no training or scoring."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_runtime import code_snapshot, external_directory, require_frozen_code, hash_file, write_json
from train_cross_sensor import read_protocol
from loadData.target_validation import make_validation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit',required=True)
    p.add_argument('--protocol-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args();code=code_snapshot()
    if sys.platform!='linux' or not Path('/.dockerenv').exists() or code['dirty'] or code['commit']!=args.expected_commit or sys.flags.optimize:
        p.error('Require clean exact SHA in container with assertions')
    out=external_directory(args.out_dir)
    for path in (args.protocol_root.resolve(),args.data_root.resolve()):
        if out==path or out in path.parents or path in out.parents:
            p.error('Keep outputs separate from inputs')
    out.mkdir(parents=True,exist_ok=False)
    manifest={'status':'running','code':code,'command':[sys.executable,*sys.argv],'optimizer_steps':0}
    write_json(out/'manifest.json',manifest)
    try:
        from scipy.io import loadmat
        from loadData.dataset_registry import DATASETS
        task,plans,support,files=read_protocol(args.protocol_root)
        identity=task['target_identity'];root=args.data_root.resolve(strict=True)
        label=(root/identity['label_path']).resolve(strict=True)
        assert label.is_relative_to(root) and hash_file(label)==identity['label_sha256']
        key=DATASETS[identity['name']][1]+'_gt'
        labels=loadmat(label,variable_names=[key])[key]
        record=make_validation(labels,plans['target'],support)
        assert record['training_labels']==90 and record['validation_labels']==45
        require_frozen_code(code)
        assert all(hash_file(path)==value for path,value in files.items())
        assert hash_file(label)==identity['label_sha256']
        write_json(out/'validation.json',record)
        write_json(out/'support.json',support)
        write_json(out/'target-plan.json',plans['target'])
        manifest.update(status='passed',parent_task_hash=task['task_hash'],validation_hash=record['validation_hash'],
            target_label_sha256=identity['label_sha256'],training_labels=90,validation_labels=45,total_labels=135,
            scope='protocol_only_no_training_or_test_evaluation',
            disclosure='MAT array decoded for protocol preparation; selector reads only val partition; never used in source training')
    except BaseException as error:
        manifest.update(status='failed',error=repr(error));raise
    finally:
        write_json(out/'manifest.json',manifest)


if __name__=='__main__':
    main()
