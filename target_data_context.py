"""Pinned G4 target observations and sanitized support/validation labels only."""
import json
from pathlib import Path
import torch
from joint_baseline import _item_fingerprint,NAMES
from loadData.dataset_registry import DATASETS,file_hash
from loadData.target_label_protocol import phase_view
from model.research_config import digest
from target_bn_calibration import GRAPH_POLICY,OBS_KEYS,observation_id


def load_target_context(target,observation_path,record_path,record_sha256,family_path,family_sha256,*,per_class,support_seed):
    pins={str(record_path):record_sha256,str(family_path):family_sha256}
    if target not in NAMES or any(file_hash(p)!=h for p,h in pins.items()):
        raise ValueError('Pinned target metadata differs')
    record=json.loads(Path(record_path).read_text());family=json.loads(Path(family_path).read_text())
    if (record.get('policy')!=GRAPH_POLICY or record.get('target')!=target or record.get('graph_fit_labels')!=0
            or family.get('target')!=target or family.get('seed')!=support_seed):
        raise ValueError('Target graph/label roles differ')
    pins[str(observation_path)]=record['observation_sha256']
    if file_hash(observation_path)!=pins[str(observation_path)]:raise ValueError('Target observation changed')
    item=torch.load(observation_path,map_location='cpu',weights_only=True)
    if (_item_fingerprint(item)!=record['prepared_fields']
            or set(item)!=OBS_KEYS|{'name','height','width','bands','class_num'}
            or item['name']!=target or item['class_num']!=DATASETS[target][2]
            or family['shape']!=[item['height'],item['width']]
            or family['classes']!=list(range(1,item['class_num']+1))):
        raise ValueError('Target observation geometry/field contract differs')
    observation_id({k:item[k] for k in OBS_KEYS})
    releases={phase:phase_view(family,per_class,phase,expected_hash=family['family_hash'])
              for phase in ('adapt','validation')}
    for phase,key in (('adapt','train'),('validation','val')):
        release=releases[phase]
        item[key+'_indices']=torch.tensor(release['indices'],dtype=torch.long)
        item[key+'_labels']=torch.tensor(release['labels'],dtype=torch.long)-1
    provenance={'policy':'g4_sanitized_target_context_v1','target':target,'support_seed':support_seed,
        'support_per_class':per_class,'validation_per_class':5,'family_hash':family['family_hash'],
        'phase_hashes':{p:digest(v) for p,v in releases.items()},'graph_policy':GRAPH_POLICY,
        'observation_record_sha256':record_sha256,'observation_sha256':pins[str(observation_path)],
        'total_annotation_labels':item['class_num']*(per_class+5),
        'query_labels_accessed':False,'full_target_scene_transductive':True,'protected_files':pins}
    def guard():
        if any(file_hash(p)!=h for p,h in pins.items()):raise ValueError('Protected target input changed')
    guard()
    return {target:item},provenance,guard
