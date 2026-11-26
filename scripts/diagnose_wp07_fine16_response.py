"""Preregistered same-weight finite differences of the isolated Q/graph/broadcast path."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, os.environ['WP07_GRAPH_SOURCE'])
import torch
from research_runtime import code_snapshot, require_frozen_code, tensor_hash
from scripts.run_wp07_b_pair import observation_only_resources
from scripts import run_native_released_points as worker
from transfer import wp07_fine16_training as training, wp07_graph_range as graph
from transfer.native_scale_campaign import read, packet
from transfer import registered_development as dev
from transfer.g3_training import state_hash
from transfer.xuzhou_multiscale_audit import _exact


def probe(first, second, x, support, dataset, entry):
    q, near, wide = (entry[k].to(device=x.device, dtype=x.dtype).coalesce() for k in ('q','near','wide'))
    count = near.shape[0]
    graph.require(x.shape[0] == 2*count, 'Expected original paired graph input')
    Q = q.to_dense()
    mass = Q.sum(0).clamp_min(1e-6)
    rows = []
    for source in (0, Q.shape[0]//2, Q.shape[0]-1):
        r = Q[source]/mass
        possible = []
        for a in (near, wide):
            # Binary arithmetic on nonnegative supports, no epsilon admission edges.
            a = torch.sparse_coo_tensor(a.indices(), (a.values()>0).float(), a.shape).coalesce()
            z = (r>0).float()[:,None]
            z = torch.sparse.mm(a, torch.sparse.mm(a,z))
            possible.append(torch.sparse.mm(q,z).flatten()>0)
        direction = torch.zeros_like(x)
        direction[:,0] = r.repeat(2)
        for epsilon in (1e-2, 1e-3):
            plus = second(first(x+epsilon*direction, support, dataset), support, dataset)
            minus = second(first(x-epsilon*direction, support, dataset), support, dataset)
            for path, a, b, bound in zip(('near','wide'), plus.split(count), minus.split(count), possible, strict=True):
                delta = torch.sparse.mm(q, a).double()-torch.sparse.mm(q, b).double()
                nonzero = (delta != 0).any(1)
                outside = int((nonzero & ~bound).sum())
                graph.require(outside == 0, 'Response escaped the frozen graph-path structural bound')
                derivative = delta/(2*epsilon)
                rows.append({'source_pixel_feature_index':source,'source_channel':0,'epsilon':epsilon,'path':path,
                    'native_cells':Q.shape[0],'potential_support_cells':int(bound.sum()),
                    'strict_nonzero_response_cells':int(nonzero.sum()),'outside_support_nonzero_cells':outside,
                    'central_difference_rms':float(derivative.square().mean().sqrt()),
                    'central_difference_absmax':float(derivative.abs().max()),
                    'raw_plus_minus_rms':float(delta.square().mean().sqrt())})
    return rows


def fixture():
    q = torch.eye(5).to_sparse().coalesce()
    a = torch.tensor([[.5,.5,0,0,0],[1/3,1/3,1/3,0,0],[0,1/3,1/3,1/3,0],[0,0,1/3,1/3,1/3],[0,0,0,.5,.5]]).to_sparse().coalesce()
    b = (torch.ones(5,5)/5).to_sparse().coalesce()
    support = torch.block_diag(a.to_dense(),b.to_dense()).to_sparse().coalesce()
    def stage(x,s,d):
        return torch.sparse.mm(s,x)
    data = probe(stage,stage,torch.zeros(10,2),support,None,{'q':q,'near':a,'wide':b})
    graph.require(len(data)==12 and all(r['strict_nonzero_response_cells']==r['potential_support_cells'] for r in data), 'Linear fixture response differs')
    return {'status':'passed','rows':len(data),'outside_support_nonzero_cells':0}


def main():
    graph.require_checks()
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','base','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-commit',required=True);p.add_argument('--gpu',required=True)
    args=p.parse_args();torch.set_num_threads(1);code=code_snapshot();require_frozen_code(code)
    graph.require(code['commit']==args.expected_commit and os.environ.get('CUDA_VISIBLE_DEVICES')==args.gpu,'Wrong frozen worker or physical GPU')
    graph.require(args.root in args.out.parents and not args.out.exists(),'Use new external diagnostic output')
    self_test=fixture();all_rows=[];models=[]
    with observation_only_resources(),worker.arithmetic(args.root,args.gpu) as execution,torch.no_grad():
        for scene in ('paviaU','xuzhou'):
            plan=read(dev._pin(args.base/scene/'pair/plan.json'),args.root)
            cache=packet(plan['cache'],args.root);receipt=read(plan['preflight'],args.root)
            for case in (c for c in plan['cases'] if c['arm']=='content'):
                dest=Path(case['destination']);spec=read(dev._pin(dest/'registration.json'),args.root)
                state=packet(read(dev._pin(dest/'completed.json'),args.root)['checkpoint'],args.root)
                provider=training.BankProvider(args.root,receipt,case['seed'])
                run=training.RangeRun(spec,provider,cache['entries'],code['commit'],execution)
                run.restore(state,spec['steps']);before=run.pack()
                for endpoint,weights in [('selected',state['selected']['state']),('last',state['model'])]:
                    run.model.load_state_dict(weights,strict=True);run.model.eval();fingerprint=state_hash(run.model)
                    rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state()
                    for event in spec['validation']:
                        actual,_,_=run._materialize('devval',event);captured={};backbone=run.model.backbone
                        graph.require('graph_stage1' not in vars(backbone) and 'cnn_stage1' not in vars(backbone),'Observer collision')
                        first,cnn=backbone.graph_stage1,backbone.cnn_stage1
                        def first_capture(*values,**kw):
                            captured['args']=values;return first(*values,**kw)
                        def cnn_capture(*values,**kw):
                            captured['cnn_input']=tensor_hash(values[0]);return cnn(*values,**kw)
                        backbone.graph_stage1,backbone.cnn_stage1=first_capture,cnn_capture
                        try:run._forward(actual,'devval',event['condition'])
                        finally:
                            del backbone.graph_stage1;del backbone.cnn_stage1
                        x,support,dataset=captured['args'];input_hash=tensor_hash(x)
                        result=probe(first,backbone.graph_stage2,x,support,dataset,run._current_entry)
                        graph.require(tensor_hash(x)==input_hash,'Graph input mutated')
                        all_rows.extend({'scene':scene,'seed':case['seed'],'endpoint':endpoint,'checkpoint_state':fingerprint,
                            'tile':event['tile'],'band':event['band'],'cnn_input':captured['cnn_input'],
                            'graph_input':input_hash,**r} for r in result)
                    graph.require(state_hash(run.model)==fingerprint and torch.equal(rng,torch.get_rng_state())
                                  and torch.equal(cuda_rng,torch.cuda.get_rng_state()),'Model or RNG changed')
                    models.append({'scene':scene,'seed':case['seed'],'endpoint':endpoint,'state':fingerprint,'views':len(spec['validation'])})
                run.restore(state,spec['steps']);graph.require(_exact(before,run.pack()),'Optimizer/model packet changed')
    graph.require(len(all_rows)==6624,'Response denominator differs')
    args.out.mkdir()
    with (args.out/'responses.csv').open('x',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    result={'status':'passed','producer':code['commit'],'program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'fixture':self_test,'models':models,'rows':len(all_rows),'optimizer_updates':0,'labels_scored':0,
        'outside_support_nonzero_cells':0,'epsilon':[1e-2,1e-3],
        'scope':'isolated graph-input pixel-feature direction, two frozen graph stages, broadcast; float32 finite differences with float64 norms; not full-network raw-input receptive field'}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
