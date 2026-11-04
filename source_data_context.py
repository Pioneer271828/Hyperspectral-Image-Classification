"""Selective source payload access; cohort target metadata is explicitly visible."""
from dataclasses import dataclass,replace
import json
from pathlib import Path
import torch
from joint_baseline import NAMES,_item_fingerprint
from joint_training import audit_preparation_source
from loadData.dataset_registry import file_hash
from model.research_config import ResearchConfig,digest


@dataclass(frozen=True)
class SourceContext:
    sources: tuple
    held_out: str
    config: ResearchConfig
    items: dict
    record: dict

    @property
    def class_counts(self):return {n:self.items[n]['class_num'] for n in self.sources}

    def build_model(self):return self.config.build_model(self.class_counts)


def roles(held_out,sources=None):
    if held_out not in NAMES:raise ValueError('Unknown held target')
    sources=tuple(n for n in NAMES if n!=held_out) if sources is None else tuple(sources)
    if (not sources or len(set(sources))!=len(sources) or held_out in sources
            or any(n not in NAMES for n in sources)):
        raise ValueError('Sources must be distinct known datasets excluding held target')
    return tuple(n for n in NAMES if n in sources)


def bounded(root,name):
    path=Path(name)
    if not name or path.is_absolute() or '..' in path.parts:raise ValueError('Invalid relative artifact path')
    result=(root/path).resolve(strict=True)
    if root not in result.parents:raise ValueError('Artifact escapes its declared root')
    return result


def load_source_context(directory,data_root,*,manifest_sha256,held_out,consumer,sources=None):
    selected=roles(held_out,sources)
    directory=Path(directory).resolve(strict=True);root=Path(data_root).resolve(strict=True)
    manifest_path=directory/'manifest.json'
    if file_hash(manifest_path)!=manifest_sha256:raise ValueError('Pinned cohort manifest changed')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    baseline=ResearchConfig.from_dict(manifest['config'])
    if (manifest.get('protocol')!='acc1p_joint_prepared_v1' or manifest.get('schema_version')!=1
            or manifest.get('test_scoring') is not False or set(manifest['datasets'])!=set(NAMES)
            or manifest['config_hash']!=baseline.config_hash or tuple(baseline.data.datasets)!=NAMES
            or manifest.get('context')!='full_scene_transductive_train_label_lda_slic'):
        raise ValueError('Unaccepted cohort configuration/protocol')
    audit_preparation_source(manifest['code'],consumer)
    items={};protected={str(manifest_path):manifest_sha256};entries={}
    for name in selected:
        entry=manifest['datasets'][name]
        if entry['split_file']!=name+'-split.json' or entry['prepared_file']!=name+'-prepared.pt':
            raise ValueError('Unexpected source artifact name')
        for field in ('split','prepared'):
            path=bounded(directory,entry[field+'_file']);expected=entry[field+'_sha256']
            if file_hash(path)!=expected:raise ValueError('Source artifact changed: '+name)
            protected[str(path)]=expected
        for field in ('image','label'):
            path=bounded(root,entry['raw'][field+'_path']);expected=entry['raw'][field+'_sha256']
            if file_hash(path)!=expected:raise ValueError('Source raw identity changed: '+name)
            protected[str(path)]=expected
        split=json.loads((directory/entry['split_file']).read_text())
        if split['seed']!=baseline.training.seed+NAMES.index(name):
            raise ValueError('Original source split seed offset changed')
        item=torch.load(directory/entry['prepared_file'],map_location='cpu',weights_only=True)
        if (any(k.startswith('test_') for k in item) or _item_fingerprint(item)!=entry['prepared_fields']
                or item['class_num']!=entry['raw']['class_count'] or item.get('Q') is None or item.get('A') is None
                or entry['normalization']['scope']!='full_scene_global_minmax'):
            raise ValueError('Source item fingerprint/access/context mismatch')
        items[name]=item;entries[name]=entry
    if any(file_hash(p)!=h for p,h in protected.items()):raise ValueError('Source files changed during import')
    config=replace(baseline,name='source_context_'+'_'.join(selected),data=replace(baseline.data,datasets=selected))
    config.validate()
    source_id=digest({'policy':'selective_acc1p_source_v1','sources':selected,
                      'producer_source_hash':manifest['code']['source_hash'],'datasets':entries})
    record={'policy':'selective_acc1p_source_v1','sources':list(selected),'held_out':held_out,'source_identity':source_id,
            'cohort_manifest_sha256':manifest_sha256,'producer':manifest['code'],'config':config.to_dict(),
            'source_class_counts':{n:items[n]['class_num'] for n in selected},'protected_files':protected,
            'target_payload_accessed':False,'cohort_target_metadata_visible':True,
            'source_normalization_context':'each_selected_source_full_scene',
            'source_graph_context':'selected_source_train_label_lda_slic',
            'target_graph_reuse_authorized':False,'joint_checkpoint_reuse_authorized':False,
            'optimizer_updates':0,'test_scoring':False}
    return SourceContext(selected,held_out,config,items,record)
