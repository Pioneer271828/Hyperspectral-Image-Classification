"""Curate G4 target labels; no model, target graph, training or test scoring."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from joint_baseline import NAMES
from loadData.dataset_registry import inspect_dataset,file_hash
from loadData.target_label_protocol import make_target_family,phase_view,final_query,BUDGETS
from research_runtime import code_snapshot,require_frozen_code,external_directory,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('expected-commit','data-root','out-dir'):p.add_argument('--'+key,required=True)
    a=p.parse_args();source=code_snapshot()
    if sys.platform!='linux' or source['dirty'] or source['commit']!=a.expected_commit:
        raise RuntimeError('Remote clean expected published commit required')
    out=external_directory(a.out_dir);out.mkdir(exist_ok=False);rows=[];identities={}
    for name in NAMES:
        identity,labels=inspect_dataset(a.data_root,name);identities[name]=identity
        for seed in (64,512,2048):
            family=make_target_family(labels,name,seed);folder=out/(name+'-'+str(seed));folder.mkdir()
            write_json(folder/'curator-family.json',family)
            nested=[];query_hash=None
            for budget in BUDGETS:
                adapt=phase_view(family,budget,'adapt',expected_hash=family['family_hash'])
                val=phase_view(family,budget,'validation',expected_hash=family['family_hash'])
                assert len(adapt['indices'])==budget*identity['class_count'] and len(val['indices'])==5*identity['class_count']
                assert not set(adapt['indices'])&set(val['indices'])
                assert all(adapt['labels'].count(c)==budget and val['labels'].count(c)==5 for c in family['classes'])
                assert all(int(labels.reshape(-1)[i])==v for i,v in zip(adapt['indices'],adapt['labels']))
                assert all(int(labels.reshape(-1)[i])==v for i,v in zip(val['indices'],val['labels']))
                if query_hash is not None:assert adapt['query_indices_hash']==query_hash
                query_hash=adapt['query_indices_hash'];nested.append(set(adapt['indices']))
                write_json(folder/f'adapt-k{budget}.json',adapt);write_json(folder/f'validation-k{budget}.json',val)
                rows.append({'target':name,'support_seed':seed,'support_per_class':budget,
                    'training_labels':len(adapt['indices']),'validation_labels':len(val['indices']),
                    'total_annotation_labels':adapt['total_annotation_labels'],'query_pixels':len(family['query_indices']),
                    'query_counts':family['query_counts'],'query_indices_hash':query_hash,'family_hash':family['family_hash']})
            assert nested[0]<=nested[1]<=nested[2]
            try:phase_view(family,10,'final_query',expected_hash=family['family_hash'])
            except ValueError:pass
            else:raise AssertionError('Early query phase accepted')
            try:final_query(family,labels,expected_hash=family['family_hash'],checkpoint_sha256=None)
            except ValueError:pass
            else:raise AssertionError('Unbound query labels exposed')
        assert inspect_dataset(a.data_root,name)[0]==identity
    require_frozen_code(source)
    write_json(out/'receipt.json',{'status':'passed','source':source,'rows':rows,'raw_identities':identities,
        'curator_full_target_label_access':True,'run_views_have_query_labels':False,
        'target_graph_built':False,'optimizer_updates':0,'model_test_scoring':False})
    print('TARGET_LABEL_PROTOCOL_PASSED',len(rows),flush=True)


if __name__=='__main__':main()
