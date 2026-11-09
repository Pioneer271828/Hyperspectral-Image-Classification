"""Audit a spatial protocol on raw registered data; no training loop."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from loadData.dataset_registry import DATASETS, inspect_dataset
from loadData.spatial_protocol import make_spatial_plan, fit_train_statistics, observe_partition, class_coverage
from research_runtime import code_snapshot, external_directory, write_json

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--out-root',required=True)
    parser.add_argument('--dataset',choices=list(DATASETS),default='paviaU')
    parser.add_argument('--axis',type=int,choices=(0,1),default=0)
    parser.add_argument('--gap',type=int,default=4)
    args=parser.parse_args()
    code=code_snapshot()
    if code['dirty']:parser.error('Commit and synchronize before auditing')
    out=external_directory(args.out_root)
    out.mkdir(parents=True,exist_ok=False)
    report={'schema_version':1,'code':code,'status':'running'}
    try:
        record,labels=inspect_dataset(args.data_root,args.dataset)
        stem,key,_,_=DATASETS[args.dataset]
        path=Path(args.data_root)/record['image_path']
        cube=loadmat(path,variable_names=[key])[key] if key else np.load(path,mmap_mode='r',allow_pickle=False)
        plan=make_spatial_plan(tuple(cube.shape[:2]),axis=args.axis,gap=args.gap)
        stats=fit_train_statistics(cube,plan)
        write_json(out/'plan.json',plan)
        write_json(out/'statistics.json',stats)
        # Round-trip and inspect one tiny crop in each independent partition.
        plan=json.loads((out/'plan.json').read_text())
        stats=json.loads((out/'statistics.json').read_text())
        crops={}
        for name,box in plan['boxes'].items():
            y0,y1,x0,x1=box
            obs=observe_partition(cube,plan,stats,name,(y0,min(y0+12,y1),x0,min(x0+12,x1)))
            crops[name]={'shape':list(obs.spectral.shape),'finite':bool(np.isfinite(obs.spectral).all())}
        report.update(status='passed',dataset=record,plan=plan,statistics_hash=stats['statistics_hash'],
                      coverage=class_coverage(labels,plan),crop_checks=crops,
                      scope='Partition preprocessing audit; no optimization or accuracy claim')
        print(args.dataset,report['coverage'],flush=True)
    except BaseException as error:
        report.update(status='failed',error=str(error))
        raise
    finally:write_json(out/'audit.json',report)

if __name__=='__main__':main()
