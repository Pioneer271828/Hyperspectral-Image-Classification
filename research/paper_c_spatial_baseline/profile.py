"""Synthetic shape costs, not real-data throughput or accuracy."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .protocol import METHODS, OPTIMIZER, SEEDS
from .tasks import Task, dependencies, clean, parameters
from .run import ROOT, execute, inspect, seed_all, new_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--native', required=True); p.add_argument('--author', required=True)
    p.add_argument('--out', required=True); p.add_argument('--expected-commit', required=True)
    a = p.parse_args(); clean(ROOT, a.expected_commit); torch.set_num_threads(1)
    adapter, author = dependencies(ROOT, a.native, a.author)
    out = Path(a.out); out.mkdir(exist_ok=False); rows = []
    rng = np.random.default_rng(90413)
    for scene, bands, classes in [('paviaU',103,9), ('xuzhou',436,9), ('whu-honghu',270,22)]:
        data = {'classes': classes, 'bands': bands, 'wavelengths': np.linspace(420,900,bands).astype('float32'), 'roles': {}}
        for role in ['train','devval']:
            data['roles'][role] = {'spectra': rng.uniform(.1,1.1,(classes,7,7,bands)).astype('float32'),
                                   'labels': np.arange(classes,dtype='int64')}
        for method in METHODS:
            c = {'roster':[0], 'sweeps':4, 'seed':64, 'plan':[0]*4, 'updates':4, 'selection_steps':[0,4],
                 'device':'cpu', 'query_reads':0, 'confirmation':False, 'optimizer':OPTIMIZER, 'protected':{},
                 'scope':'synthetic cost profile only', 'producer':a.expected_commit, 'scene':scene, 'method':method}
            destination = out/f'{scene}-{method}'
            execute(destination,c,lambda:Task(scene,data,method,adapter,author))
            stages, complete = inspect(destination,c)
            seed_all(64); task = Task(scene,data,method,adapter,author)
            row = {'scene':scene, 'bands':bands, 'classes':classes, 'method':method, 'trainable_parameters':parameters(task),
                   'maximum_state_bytes':max(r['checkpoint_bytes'] for r in stages), 'measured_synthetic_updates':4,
                   'train_step_median_seconds_excluding_warmup':float(np.median([sum(r['timings'][k] for k in
                       ['forward_seconds','backward_seconds','optimizer_seconds','checkpoint_seconds']) for r in stages[2:]])),
                   'selection_per_center_seconds':float(np.mean([r['timings']['selection_seconds']/classes for r in [stages[0],stages[-1]]]))}
            #12 copies/method over scenes? Each scene-method has3 seed runs,
            #40 sweeps *5 labeled centers/class, plus9 selections of5/class.
            row['planned_real_updates'] = 3*40*5*classes
            row['shape_only_train_seconds_projection'] = row['planned_real_updates']*row['train_step_median_seconds_excluding_warmup']
            row['shape_only_selection_seconds_projection'] = 3*9*5*classes*row['selection_per_center_seconds']
            row['measured_shape_state_projection_bytes'] = 3*(40*5*classes+1)*row['maximum_state_bytes']
            rows.append(row)
    value = {'schema':'c13_synthetic_shape_costs_v1', 'producer':a.expected_commit, 'rows':rows,
             'synthetic_updates':48, 'future_real_updates':sum(r['planned_real_updates'] for r in rows),
             'future_real_updates_executed':0, 'shape_seconds_projection':sum(r['shape_only_train_seconds_projection']+r['shape_only_selection_seconds_projection'] for r in rows),
             'measured_shape_state_projection_bytes':sum(r['measured_shape_state_projection_bytes'] for r in rows),
             'exclusions':['real input validation/normalization','larger real selection histories','file hashing/audit','source loading','backup','runtime contention'],
             'not_a_wall_time_or_storage_guarantee':True, 'query_reads':0}
    new_json(out/'profile.json',value); print(json.dumps({k:v for k,v in value.items() if k!='rows'}))


if __name__ == '__main__': main()
