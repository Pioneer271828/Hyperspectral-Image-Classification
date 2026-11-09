"""G4-v1 transductive label curation and sanitized phase release, not a sandbox."""
import re
import numpy as np
from joint_baseline import NAMES
from loadData.fixed_splits import make_split,label_hash
from model.research_config import digest

CONTEXT={'input_context':'full_target_scene_transductive','spatial_isolation':False,
    'curator_full_label_access':True,'target_graph_policy':'label_free_inputs_only',
    'reuse_target_acc1p_lda_graph':False,'shared_semantics_claim':False}
BUDGETS=(1,5,10)


def make_target_family(labels,target,seed):
    if target not in NAMES:raise ValueError('Unknown target')
    fingerprint=label_hash(labels);flat=np.asarray(labels).reshape(-1)
    classes=np.unique(flat[flat>0]).astype(int).tolist()
    if not classes or classes!=list(range(1,len(classes)+1)):raise ValueError('Contiguous foreground classes required')
    split=make_split(labels,seed=seed,train_num=10,val_num=5)
    rng=np.random.default_rng(seed ^ 0x4754);pool=np.asarray(split['indices']['train'])
    ordering={str(c):rng.permutation(pool[flat[pool]==c]).tolist() for c in classes}
    val=sorted(split['indices']['val']);query=sorted(split['indices']['test'])
    result={'schema_version':1,'protocol':'g4_nested_target_labels_v1','target':target,
        'shape':list(np.shape(labels)),'label_hash':fingerprint,'seed':seed,'classes':classes,
        'max_support_per_class':10,'validation_per_class':5,'context':dict(CONTEXT),
        'support_order':ordering,'validation':{'indices':val,'labels':flat[val].astype(int).tolist()},
        'query_indices':query,'query_counts':{str(c):int(np.count_nonzero(flat[query]==c)) for c in classes}}
    result['family_hash']=digest(result)
    validate_family(result,result['family_hash'])
    return result


def validate_family(record,expected_hash):
    required={'schema_version','protocol','target','shape','label_hash','seed','classes','max_support_per_class',
        'validation_per_class','context','support_order','validation','query_indices','query_counts','family_hash'}
    if (set(record)!=required or not isinstance(expected_hash,str) or not re.fullmatch('[0-9a-f]{64}',expected_hash)
            or record['family_hash']!=expected_hash or digest({k:v for k,v in record.items() if k!='family_hash'})!=expected_hash
            or record['schema_version']!=1 or record['protocol']!='g4_nested_target_labels_v1'
            or record['target'] not in NAMES or record['context']!=CONTEXT
            or record['max_support_per_class']!=10 or record['validation_per_class']!=5):
        raise ValueError('Pinned target family identity/protocol mismatch')
    classes=record['classes'];shape=record['shape'];seed=record['seed']
    if (not classes or any(type(c)is not int for c in classes) or classes!=list(range(1,len(classes)+1))
            or len(shape)!=2 or any(type(v)is not int or v<=0 for v in shape)
            or type(seed)is not int or not 0<=seed<2**32
            or not re.fullmatch('[0-9a-f]{64}',record['label_hash'])
            or set(record['support_order'])!=set(map(str,classes))
            or set(record['query_counts'])!=set(map(str,classes))
            or set(record['validation'])!={'indices','labels'}):
        raise ValueError('Invalid target family shape/classes/seed')
    groups=[record['support_order'][str(c)] for c in classes]
    val=record['validation'];query=record['query_indices'];all_ids=[]
    for ids in (*groups,val['indices'],query):
        if any(type(i)is not int or not 0<=i<shape[0]*shape[1] for i in ids):raise ValueError('Invalid target index')
        all_ids.extend(ids)
    if (any(len(g)!=10 for g in groups) or len(val['indices'])!=5*len(classes)
            or val['indices']!=sorted(val['indices']) or query!=sorted(query)
            or len(all_ids)!=len(set(all_ids)) or len(val['labels'])!=len(val['indices'])
            or any(type(c)is not int or c not in classes for c in val['labels'])
            or any(val['labels'].count(c)!=5 for c in classes)
            or any(type(v)is not int or v<=0 for v in record['query_counts'].values())
            or sum(record['query_counts'].values())!=len(query)):
        raise ValueError('Target roles overlap or class budgets differ')


def phase_view(record,per_class,phase,*,expected_hash):
    validate_family(record,expected_hash)
    if type(per_class)is not int or per_class not in BUDGETS or phase not in ('adapt','validation'):
        raise ValueError('Only registered support/validation phase releases are allowed')
    if phase=='adapt':
        pairs=sorted((i,c) for c in record['classes'] for i in record['support_order'][str(c)][:per_class])
        ids=[i for i,c in pairs];labels=[c for i,c in pairs]
    else:
        ids=list(record['validation']['indices']);labels=list(record['validation']['labels'])
    return {'schema_version':1,'protocol':record['protocol'],'target':record['target'],
        'parent_hash':expected_hash,'shape':list(record['shape']),'phase':phase,
        'support_per_class':per_class,'validation_per_class':5,
        'total_annotation_labels':len(record['classes'])*(per_class+5),
        'indices':ids,'labels':labels,'query_indices_hash':digest(record['query_indices']),
        'context':dict(CONTEXT),'unreleased_max_pool_labels_visible':False,'query_labels_visible':False}


def final_query(record,labels,*,expected_hash,checkpoint_sha256):
    validate_family(record,expected_hash)
    if not isinstance(checkpoint_sha256,str) or not re.fullmatch('[0-9a-f]{64}',checkpoint_sha256):
        raise ValueError('Final query release requires frozen checkpoint SHA256')
    if list(np.shape(labels))!=record['shape'] or label_hash(labels)!=record['label_hash']:
        raise ValueError('Final target labels changed')
    ids=list(record['query_indices']);values=np.asarray(labels).reshape(-1)[ids].astype(int).tolist()
    return {'parent_hash':expected_hash,'checkpoint_sha256':checkpoint_sha256,'phase':'final_query',
        'indices':ids,'labels':values,'model_evaluated':False}
