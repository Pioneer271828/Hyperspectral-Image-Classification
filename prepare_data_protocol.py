"""Inspect raw files and freeze random-pixel splits without loading image cubes."""
import argparse, json
from pathlib import Path
from loadData.dataset_registry import DATASETS, inspect_dataset
from loadData.fixed_splits import make_split, save_split, load_split
from research_runtime import external_directory, code_snapshot, write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--out-root',required=True)
    p.add_argument('--datasets',default=','.join(DATASETS))
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--train-num',type=int,default=30)
    p.add_argument('--val-num',type=int,default=10)
    a=p.parse_args()
    names=a.datasets.split(',')
    if len(set(names))!=len(names) or set(names)-set(DATASETS):p.error('Unknown/duplicate dataset names')
    code=code_snapshot()
    if code['dirty']:p.error('Commit and synchronize code first')
    out=external_directory(a.out_root)
    out.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'code':code,'data_root':str(Path(a.data_root).resolve()),'protocol':'fixed_random_pixel_v1','datasets':{},'status':'running'}
    try:
        for name in names:
            record,labels=inspect_dataset(a.data_root,name)
            split=make_split(labels,a.seed,a.train_num,a.val_num)
            save_split(out/(name+'.split.json'),split)
            restored=load_split(out/(name+'.split.json'),labels)
            record['split_counts']={k:len(v) for k,v in restored['indices'].items()}
            report['datasets'][name]=record
            print(name,record['shape'],record['split_counts'],flush=True)
        report['status']='passed'
    except BaseException as e:
        report.update(status='failed',error=str(e))
        raise
    finally:write_json(out/'registry.json',report)

if __name__=='__main__':main()
