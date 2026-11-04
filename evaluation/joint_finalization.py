"""Explicit recovery of completed training blocked by native probability roundoff.

No optimizer is constructed or stepped. Existing weights/state are never rewritten.
The original training entry and its stricter finalization rule remain unchanged.
"""
from collections import Counter
import json
import math
from pathlib import Path
import torch

from joint_training import (_completed, _run_lock, _tree_fingerprint, schedule_for,
                            validation_epochs, validation_snapshot)
from joint_calibrated_selection import enabled
from loadData.dataset_registry import file_hash
from model.checkpoint_io import load_model_checkpoint
from model.research_config import digest
from research_runtime import write_json, external_directory

POLICY = 'exact_state_same_classes_probability_atol_1e-5_v1'


def verify_probability_pair(old, new):
    """Absolute probability bound plus unchanged classification, never a new selection."""
    if (old.shape != new.shape or old.ndim != 2 or not old.numel()
            or old.dtype != new.dtype or old.dtype != torch.float32):
        raise ValueError('Matching nonempty FP32 probability matrices required')
    for value in (old,new):
        if (not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any()
                or not torch.allclose(value.sum(1),torch.ones_like(value[:,0]),rtol=0,atol=2e-6)):
            raise ValueError('Invalid probability matrix')
    delta = (new-old).abs()
    if not torch.equal(old.argmax(1),new.argmax(1)):
        raise ValueError('Selected classifications changed')
    if float(delta.max()) > 1e-5:
        raise ValueError('Probability drift exceeds declared absolute bound')
    return {'max_abs_error':float(delta.max()),
            'legacy_tolerance_violations':int((delta > 1e-6+1e-5*old.abs()).sum()),
            'classification_changes':0}


def finalize(directory, settings, items, provenance, training_source, verification_source,
             runtime, *, device='cpu', finish_guard=None):
    """Finalize a fully trained calibrated run under a separately recorded verification policy."""
    if not enabled(settings) or not training_source.get('files'):
        raise ValueError('Calibrated training and bound producer source required')
    if training_source.get('dirty') or verification_source.get('dirty'):
        raise ValueError('Clean training and verification sources required')
    if training_source['source_hash'] != digest(training_source['files']):
        raise ValueError('Training source fingerprint mismatch')
    if any(verification_source['files'].get(n)!=h for n,h in training_source['files'].items()):
        raise ValueError('Training implementation changed; additive verifier only')
    directory=external_directory(directory)
    schedule=schedule_for(settings)
    identity={'protocol':settings['protocol'],'settings_hash':digest(settings),'settings':settings,
              'execution_device':str(torch.device(device)),
              'consumer_source_hash':training_source['source_hash'],'prepared':provenance,
              'schedule_sha256':digest(schedule)}
    load=lambda name:json.loads((directory/name).read_text())
    with _run_lock(directory):
        if load('identity.json') != identity:
            raise ValueError('Training identity differs')
        if (directory/'result.json').exists():
            return _completed(directory,identity,settings,schedule)
        progress=load('progress.json')
        state_name=progress['state_file']
        if (progress['cursor']!=len(schedule) or Path(state_name).name!=state_name
                or file_hash(directory/state_name)!=progress['state_sha256']):
            raise ValueError('Incomplete or changed final training state')
        state=torch.load(directory/state_name,map_location='cpu',weights_only=True)
        if (state['cursor']!=len(schedule) or state['identity']!=identity
                or state['start_code']!=training_source
                or [r['dataset'] for r in state['trace']]!=schedule
                or [r['step'] for r in state['trace']]!=list(range(1,len(schedule)+1))):
            raise ValueError('Final state provenance/schedule differs')
        for key in ('model','optimizer','scheduler'):
            if _tree_fingerprint(state[key])!=state['state_fingerprints'][key]:
                raise ValueError('Training-state fingerprint differs: '+key)
        if [r['epoch'] for r in state['history']]!=validation_epochs(settings):
            raise ValueError('Validation history is incomplete')
        if any(not math.isfinite(r['macro_oa']) for r in state['history']):
            raise ValueError('Nonfinite selection history')
        best=max(state['history'],key=lambda r:r['macro_oa'])
        if state['best_step']!=best['step'] or state['best_score']!=best['macro_oa']:
            raise ValueError('Validation selection changed')
        protected={p.name:file_hash(p) for p in directory.iterdir() if p.is_file()
                   and (p.suffix=='.pt' or p.name.endswith('.pt.json') or p.name=='identity.json')}
        primary=runtime.load_selected(directory/'primary.pt',items,
                    directory/f'raw-validation-{state["best_step"]:06d}.pt',device=device)
        last=load_model_checkpoint(directory/'last.pt',device='cpu')
        if (_tree_fingerprint(primary.state_dict())!=_tree_fingerprint(state['best_model'])
                or _tree_fingerprint(last.state_dict())!=state['state_fingerprints']['model']):
            raise ValueError('Saved weights differ from authoritative training state')
        for model,step in ((primary,state['best_step']),(last,len(schedule))):
            meta=model.training_config
            if (meta['training_code']!=training_source or meta['provenance']!=provenance
                    or meta['research_config']!=settings['config']
                    or meta['checkpoint_optimizer_step']!=step):
                raise ValueError('Saved checkpoint metadata changed')
        del last
        saved=torch.load(directory/'selected-probabilities.pt',map_location='cpu',weights_only=True)
        if _tree_fingerprint(saved)!=_tree_fingerprint(state['best_probabilities']):
            raise ValueError('Saved selection probabilities changed')
        moved={n:{k:v.to(device) if torch.is_tensor(v) else v for k,v in i.items()} for n,i in items.items()}
        check,values=runtime.validation_snapshot(primary,moved)
        comparisons=[]
        for name in items:
            for phase in ('train','val'):
                if set(saved[name][phase])!=set(values[name][phase]):
                    raise ValueError('Head set changed')
                for head in saved[name][phase]:
                    comparisons.append({'dataset':name,'phase':phase,'head':head,
                        **verify_probability_pair(saved[name][phase][head],values[name][phase][head])})
        if any(file_hash(directory/n)!=h for n,h in protected.items()):
            raise ValueError('Read-only verification changed protected artifacts')
        if finish_guard:finish_guard()
        # Existing outputs must agree before completing an interrupted final write.
        for name,value in (('trace.json',state['trace']),('history.json',state['history'])):
            if (directory/name).exists() and load(name)!=value:
                raise ValueError('Existing training record differs: '+name)
        receipt={'policy':POLICY,'training_code':training_source,'verification_code':verification_source,
                 'updates_this_call':0,'weights_exact':True,'comparisons':comparisons,'protected_files':protected,
                 'test_scoring':False,'original_strict_rule_not_retroactively_passed':True}
        if (directory/'finalization-recovery.json').exists():
            raise ValueError('Existing recovery receipt; inspect interrupted publication')
        write_json(directory/'finalization-recovery.json',receipt)
        for name,value in (('trace.json',state['trace']),('history.json',state['history'])):
            if not (directory/name).exists():write_json(directory/name,value)
        write_json(directory/'progress.json',{**progress,'status':'completed'})
        artifacts={p.name:file_hash(p) for p in directory.iterdir() if p.is_file() and p.name not in ('.lock','result.json')}
        result={'status':'completed','identity':identity,'start_code':training_source,'settings':settings,
            'optimizer_steps':len(schedule),'schedule_sha256':digest(schedule),'per_dataset_updates':dict(Counter(schedule)),
            'selected_step':state['best_step'],'selected_validation':check,
            'selected_probability_max_abs_error':max(r['max_abs_error'] for r in comparisons),
            'last_model_fingerprint':state['state_fingerprints']['model'],'artifacts':artifacts,'test_scoring':False,
            'finalization_policy':POLICY,'verification_code':verification_source,
            'execution':{'device':str(device),'torch_version':torch.__version__,'cuda_version':torch.version.cuda}}
        with (directory/'result.json').open('x',encoding='utf-8') as stream:
            json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
        return {'operation':'finalized_without_training','updates_this_call':0,'result':result}
