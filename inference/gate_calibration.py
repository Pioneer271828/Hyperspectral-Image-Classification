"""Small preregistered gate grid with spatially separate empirical confirmation.

Consumes authenticated development predictions; does not release labels, run
models, fit weights or establish statistical non-inferiority/deployment speed.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from inference.tile_cascade import gate_score
from model.research_config import digest


SPARSE_PROTOCOL = 'dy-gate-protocol-sparse-v2'


def _require(ok,message):
    if not ok:raise ValueError(message)


def _hash(x):return isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x)
def _unit(x):return type(x) in (int,float) and math.isfinite(x) and 0<=x<=1


def observed_records_hash(rows):
    """Bind the observed probabilities separately from a sparse label commitment."""
    return digest([dict(id=r['block']['id'], labels=r['labels'].tolist(), fast=r['fast'].tolist(),
                        full=r['full'].tolist(), coverage_known=r['known'], coverage_fraction=r['coverage'])
                   for r in rows])


class GateProtocol:
    def __init__(self,spec,expected_hash):
        _require(isinstance(spec,dict) and digest(spec)==expected_hash,'External protocol hash mismatch')
        fields = {'schema','name','full_state','light_artifact','criterion','thresholds','min_coverage',
                  'classes','min_class_support','max_drop','blocks','data_hashes'}
        schema = spec.get('schema')
        if schema == SPARSE_PROTOCOL:
            fields |= {'admission_hash', 'calibration_request_hash'}
            _require(all(_hash(spec.get(k)) for k in ('admission_hash', 'calibration_request_hash')),
                     'Pinned admission and calibration requests required')
        _require(set(spec)==fields,'Exact protocol fields required')
        _require(schema in ('dy-gate-protocol-v1', SPARSE_PROTOCOL) and isinstance(spec['name'],str) and spec['name'],'Named protocol required')
        _require(_hash(spec['full_state']) and _hash(spec['light_artifact']),'Frozen model identities required')
        _require(type(spec['classes']) is int and spec['classes']>=2,'At least two classes required')
        _require(type(spec['min_class_support']) is int and spec['min_class_support']>=1,'Explicit positive support floor required')
        _require(_unit(spec['min_coverage']),'Invalid coverage floor')
        _require(spec['criterion'] in ('max_probability','margin','entropy'),'One registered criterion required')
        grid=spec['thresholds'];_require(isinstance(grid,list) and 1<=len(grid)<=8,'Fixed grid of one to eight thresholds required')
        _require(all(type(t) in (int,float) and math.isfinite(t) and t>=0 and (spec['criterion']=='entropy' or t<=1) for t in grid)
                 and len(set(grid))==len(grid),'Invalid or repeated thresholds')
        _require(isinstance(spec['max_drop'],dict) and set(spec['max_drop'])=={'oa','aa','recall'} and all(_unit(v) for v in spec['max_drop'].values()),'Explicit empirical class protection required')
        _require(isinstance(spec['data_hashes'],dict) and set(spec['data_hashes'])=={'calibration','confirmation'} and all(_hash(v) for v in spec['data_hashes'].values()),'Both development data pins required')
        blocks=spec['blocks'];_require(isinstance(blocks,list) and blocks,'Registered spatial contexts required');seen=set();roles=set()
        for b in blocks:
            _require(isinstance(b,dict) and set(b)=={'id','scene','condition','role','context','label_indices'},'Exact block metadata required')
            _require(all(isinstance(b[k],str) and b[k] for k in ('id','scene','condition')) and b['id'] not in seen,'Unique block identity required')
            seen.add(b['id']);roles.add(b['role'])
            _require(b['role'] in ('training','selection','calibration','confirmation'),'Test/query blocks are forbidden')
            box=b['context'];_require(isinstance(box,list) and len(box)==4 and all(type(v) is int and v>=0 for v in box) and box[0]<box[1] and box[2]<box[3],'Actual rectangular context required')
            ids=b['label_indices'];area=(box[1]-box[0])*(box[3]-box[2])
            _require(isinstance(ids,list) and all(type(i) is int and 0<=i<area for i in ids) and ids==sorted(set(ids)),'Unique sorted authorized label locations required')
        _require(roles=={'training','selection','calibration','confirmation'},'All four roles must be declared')
        for i,a in enumerate(blocks):
            for b in blocks[i+1:]:
                if a['scene']!=b['scene'] or (a['role']==b['role'] and a['condition']!=b['condition']):continue
                x,y=a['context'],b['context'];overlap=max(x[0],y[0])<min(x[1],y[1]) and max(x[2],y[2])<min(x[3],y[3])
                _require(not overlap,'Context reused across roles or duplicate condition blocks')
        self._spec=deepcopy(spec);self.protocol_hash=expected_hash

    @property
    def specification(self):return deepcopy(self._spec)

    def _check(self):_require(digest(self._spec)==self.protocol_hash,'Protocol changed')

    def records(self,records,role):
        self._check();_require(role in ('calibration','confirmation'),'Only permitted development roles accepted')
        _require(isinstance(records,list),'Materialized aligned records required')
        blocks=[b for b in self._spec['blocks'] if b['role']==role];byid={b['id']:b for b in blocks}
        _require(len(records)==len(blocks),'Complete registered phase required');seen=set();rows=[];classes=self._spec['classes']
        sparse = self._spec['schema'] == SPARSE_PROTOCOL
        for r in records:
            _require(isinstance(r,dict) and set(r)=={'id','labels','fast','full','coverage_known','coverage_fraction'},'Exact aligned input fields required')
            _require(r['id'] in byid and r['id'] not in seen,'Wrong role or repeated block');seen.add(r['id']);b=byid[r['id']]
            box=b['context'];area=(box[1]-box[0])*(box[3]-box[2]);indices=np.asarray(b['label_indices'],dtype=np.int64)
            labels=np.asarray(r['labels']);_require(labels.shape==(area,) and labels.dtype.kind in 'iuf','Label shape/type mismatch')
            y=labels[indices];_require(np.isfinite(y).all() and np.equal(y,np.floor(y)).all() and ((y>=1)&(y<=classes)).all(),'Unauthorized or invalid selected label')
            y=y.astype(np.int64);probs=[]
            for field in ('fast','full'):
                p=np.asarray(r[field]);_require(p.shape==(area,classes) and p.dtype==np.float32 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                    and np.allclose(p.sum(1),1,rtol=0,atol=1e-6),'Finite aligned full-context FP32 probabilities required')
                probs.append(p.copy())
            _require(type(r['coverage_known']) is bool and _unit(r['coverage_fraction']),'Explicit finite coverage required')
            rows.append(dict(block=b,labels=y,indices=indices,fast=probs[0],full=probs[1],known=r['coverage_known'],coverage=r['coverage_fraction']))
        rows.sort(key=lambda r: r['block']['id'])
        if sparse:
            pin = digest({r['block']['id']: r['labels'].tolist() for r in rows})
        else:
            pin = observed_records_hash(rows)
        _require(pin==self._spec['data_hashes'][role],'Frozen development data changed')
        return rows,pin


def _metrics(y,pred,classes):
    c=np.bincount((y-1)*classes+pred,minlength=classes**2).reshape(classes,classes);support=c.sum(1)
    recall=np.divide(np.diag(c),support,out=np.zeros(classes,dtype=float),where=support>0)
    return dict(count=int(c.sum()),support=support.tolist(),confusion=c.tolist(),recall=recall.tolist(),
                oa=float(np.trace(c)/c.sum()) if c.sum() else None,aa=float(recall.mean()) if (support>0).all() else None)


def _evaluate(protocol,rows,kind,threshold=None):
    s=protocol.specification;classes=s['classes'];groups={};ys=[];ps=[];fs=[];routes=[];accepted_labels=wrong_accepted=0
    for row in rows:
        known=row['known'] and row['coverage']>=s['min_coverage'];accept=False;score=None
        if kind!='full' and known:
            if kind=='fast':accept=True
            else:
                score=gate_score(torch.from_numpy(row['fast']),s['criterion']);accept=score<=threshold if s['criterion']=='entropy' else score>=threshold
        indices=row['indices'];f=row['full'][indices].argmax(1);p=row['fast' if accept else 'full'][indices].argmax(1);y=row['labels']
        ys.extend(y);ps.extend(p);fs.extend(f);key=(row['block']['scene'],row['block']['condition'])
        groups.setdefault(key,[]).append((y,p,f));routes.append(dict(id=row['block']['id'],fast=accept,score=score))
        if accept:accepted_labels+=len(y);wrong_accepted+=int((p!=y-1).sum())
    a=lambda x:np.asarray(x,dtype=np.int64)
    result=dict(kind=kind,threshold=threshold,routes=routes,accepted_tiles=sum(r['fast'] for r in routes),total_tiles=len(rows),
        accepted_labeled_points=accepted_labels,wrong_accepted_points=wrong_accepted,metrics=_metrics(a(ys),a(ps),classes),full=_metrics(a(ys),a(fs),classes),groups=[])
    for (scene,condition),parts in sorted(groups.items()):
        y,p,f=[np.concatenate([v[i] for v in parts]) for i in range(3)]
        result['groups'].append(dict(scene=scene,condition=condition,metrics=_metrics(y,p,classes),full=_metrics(y,f,classes)))
    def eligible(item):
        m,b=item['metrics'],item['full'];drop=s['max_drop']
        return (min(m['support'])>=s['min_class_support'] and b['oa']-m['oa']<=drop['oa']
                and b['aa']-m['aa']<=drop['aa'] and all(x-y<=drop['recall'] for x,y in zip(b['recall'],m['recall'])))
    result['passes_empirical_protection']=eligible(result) and all(eligible(g) for g in result['groups'])
    return result


def calibrate(protocol,records):
    rows,pin=protocol.records(records,'calibration')
    candidates=[_evaluate(protocol,rows,'full')]+[_evaluate(protocol,rows,'adaptive',t) for t in protocol.specification['thresholds']]
    eligible=[i for i,c in enumerate(candidates) if c['passes_empirical_protection']]
    # First registered candidate wins equal-coverage ties; full remains the safe zero-coverage option.
    selected=max(eligible,key=lambda i:candidates[i]['accepted_tiles']) if eligible else 0
    return dict(format='dy-gate-freeze-v1',protocol_hash=protocol.protocol_hash,calibration_hash=pin,candidates=candidates,
                fast_endpoint=_evaluate(protocol,rows,'fast'),selected=selected,selection='max_accepted_tiles_then_registered_order',
                statistical_noninferiority=False,deployment_speed_verified=False)


def _validate_freeze(protocol,frozen,expected_freeze_hash):
    protocol._check();_require(digest(frozen)==expected_freeze_hash,'External frozen policy hash mismatch')
    _require(isinstance(frozen,dict) and set(frozen)=={'format','protocol_hash','calibration_hash','candidates','fast_endpoint','selected','selection','statistical_noninferiority','deployment_speed_verified'}
        and frozen['format']=='dy-gate-freeze-v1' and frozen['protocol_hash']==protocol.protocol_hash
        and frozen['calibration_hash']==protocol.specification['data_hashes']['calibration']
        and frozen['selection']=='max_accepted_tiles_then_registered_order' and frozen['statistical_noninferiority'] is False
        and frozen['deployment_speed_verified'] is False,'Wrong frozen calibration contract')
    candidates=frozen['candidates'];_require(len(candidates)==1+len(protocol.specification['thresholds']),'Candidate grid changed')
    _require(candidates[0]['kind']=='full' and candidates[0]['threshold'] is None,'Missing full endpoint')
    for c,t in zip(candidates[1:],protocol.specification['thresholds']):_require(c['kind']=='adaptive' and c['threshold']==t,'Registered grid changed')
    eligible=[i for i,c in enumerate(candidates) if c['passes_empirical_protection']]
    selected=max(eligible,key=lambda i:candidates[i]['accepted_tiles']) if eligible else 0
    _require(type(frozen['selected']) is int and frozen['selected']==selected,'Frozen candidate was reselected')


def confirm(protocol,frozen,records,*,expected_freeze_hash):
    """Confirm only the preselected candidate; failure never searches alternatives."""
    _validate_freeze(protocol,frozen,expected_freeze_hash);rows,pin=protocol.records(records,'confirmation');c=frozen['candidates'][frozen['selected']]
    tested=_evaluate(protocol,rows,c['kind'],c['threshold']);use_fast=c['kind']=='adaptive' and c['accepted_tiles']>0 and tested['passes_empirical_protection']
    result = dict(format='dy-gate-confirmation-v1',protocol_hash=protocol.protocol_hash,freeze_hash=expected_freeze_hash,confirmation_hash=pin,
                selected=frozen['selected'],tested=tested,status='empirical_pass' if use_fast else 'keep_full',
                inference_kwargs=dict(mode='adaptive',criterion=protocol.specification['criterion'],threshold=c['threshold'],min_coverage=protocol.specification['min_coverage']) if use_fast else dict(mode='full'),
                full_endpoint=_evaluate(protocol,rows,'full'),fast_endpoint=_evaluate(protocol,rows,'fast'),
                statistical_noninferiority=False,deployment_speed_verified=False,optimizer_updates=0)
    if protocol.specification['schema'] == SPARSE_PROTOCOL:
        result['observed_records_hash'] = observed_records_hash(rows)
    return result


def save_freeze(protocol,frozen,path):
    """New JSON only; caller archives returned file hash separately from policy hash."""
    _validate_freeze(protocol,frozen,digest(frozen));raw=(json.dumps(frozen,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()
    with Path(path).open('xb') as f:f.write(raw)
    return hashlib.sha256(raw).hexdigest()


def load_freeze(protocol,path,*,expected_file_sha256,expected_freeze_hash):
    raw=Path(path).read_bytes();_require(hashlib.sha256(raw).hexdigest()==expected_file_sha256,'Frozen file hash mismatch')
    frozen=json.loads(raw);_validate_freeze(protocol,frozen,expected_freeze_hash);return frozen
