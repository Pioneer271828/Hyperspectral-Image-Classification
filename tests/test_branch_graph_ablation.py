"""Independent routes, matched controls and real training-state continuity."""
import copy,json,os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from branch_graph_training import AblationRuntime,ablation_settings,forward,loss_terms,parameter_manifest
from model.branch_graph_ablation import (VARIANTS,BranchGraphAblation,controlled_items,ordinary_graph,
                                        load_ablation,load_ablation_selected)
from model.VCASAHGCN import MultiDatasetVCASAHGCN
from model.checkpoint_io import save_model_checkpoint,load_model_checkpoint
from joint_training import run_training,_tree_fingerprint
from test_joint_baseline import tiny_items
from test_joint_training import settings,SOURCE,PROVENANCE,checkpoints,hashes
from model.research_config import ResearchConfig
from research_runtime import tensor_hash

torch.set_num_threads(1)
DEVICE=os.environ.get('BG_ABL_DEVICE','cpu')
KWARGS=dict(latent_channels=8,adapter_embed_dim=4,norm_type='domain_bn')


def data():
    result=tiny_items()
    for n,i in result.items():
        i['zeros']=torch.zeros(i['height']*i['width'])
        result[n]={k:v.to(DEVICE) if torch.is_tensor(v) else v for k,v in i.items()}
    return result


def model(variant,items):
    torch.manual_seed(64)
    return BranchGraphAblation({n:i['class_num'] for n,i in items.items()},variant,**KWARGS).to(DEVICE)


class BranchGraphTests(unittest.TestCase):
    def test_legacy_forward_loss_and_adam_step_match_original(self):
        items=data();name=next(iter(items));item=items[name];variant=VARIANTS[0]
        candidate=model(variant,items)
        torch.manual_seed(64);reference=MultiDatasetVCASAHGCN({n:i['class_num'] for n,i in items.items()},**KWARGS).to(DEVICE)
        candidate.eval();reference.eval()
        args=(name,item['net_input'],item['Q'],item['A'])
        kwargs=dict(wavelengths=item['wavelengths'],fwhm=item['fwhm'])
        with torch.no_grad():
            a=candidate(*args,**kwargs);b=reference(*args,**kwargs)
            for h,q in zip(('fused','graph','cnn'),b):torch.testing.assert_close(a[h],q,rtol=0,atol=0)
        config=ResearchConfig.from_dict(settings()['config'])
        from train_vca_multidataset import supervised_loss
        def opt(m):return torch.optim.Adam([{'params':m.backbone.parameters(),'lr':5e-5},
            {'params':list(m.heads.parameters())+list(m.hgnn_heads.parameters())+list(m.cnn_heads.parameters()),'lr':5e-4}],weight_decay=.001)
        aopt,bopt=opt(candidate),opt(reference)
        candidate.train();reference.train();aopt.zero_grad();bopt.zero_grad()
        a=loss_terms(forward(candidate,name,item),item,config)['total']
        b=supervised_loss(reference(*args,**kwargs),item,'train',class_balance=True)
        torch.testing.assert_close(a,b,rtol=0,atol=0);a.backward();b.backward();aopt.step();bopt.step()
        for key,value in reference.state_dict().items():torch.testing.assert_close(candidate.state_dict()[key],value,rtol=0,atol=0)
        self.assertEqual(_tree_fingerprint(aopt.state_dict()),_tree_fingerprint(bopt.state_dict()))

    def test_retained_initialization_and_actual_parameter_removal(self):
        items=data();reference=model(VARIANTS[0],items);base=reference.state_dict()
        manifests={}
        for variant in VARIANTS:
            m=model(variant,items);manifests[variant]=parameter_manifest(m)
            for k,p in m.named_parameters():torch.testing.assert_close(p,base[k],rtol=0,atol=0)
            if variant=='cnn_fused_only':self.assertIsNone(m.backbone.HGNN_Branch)
            if variant=='graph_fused_hypergraph':self.assertEqual(len(m.backbone.CNN_Branch),0)
            if m.ablation['loss']=='primary_only':self.assertEqual(len(m.hgnn_heads)+len(m.cnn_heads),0)
        self.assertEqual(manifests[VARIANTS[0]],manifests[VARIANTS[4]])
        self.assertLess(manifests['cnn_fused_only']['total'],manifests['dual_fused_hypergraph']['total'])
        self.assertLess(manifests['graph_fused_hypergraph']['total'],manifests['dual_fused_hypergraph']['total'])

    def test_cnn_does_not_read_graph_or_execute_graph_route(self):
        items=data();name=next(iter(items));i=items[name];m=model('cnn_fused_only',items).eval()
        with patch.object(m.backbone,'graph_stage1',side_effect=AssertionError('inactive graph')):
            with torch.no_grad():
                a=forward(m,name,i)['fused'];b=m(name,i['net_input'],None,None,wavelengths=i['wavelengths'],fwhm=i['fwhm'])['fused']
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_graph_does_not_execute_spatial_cnn_and_primary_loss_is_unscaled(self):
        items=data();name=next(iter(items));i=items[name];m=model('graph_fused_hypergraph',items)
        with patch.object(m.backbone,'cnn_stage1',side_effect=AssertionError('inactive cnn')):
            output=forward(m,name,i)
        config=ResearchConfig.from_dict(settings()['config']);terms=loss_terms(output,i,config)
        self.assertEqual(set(output),{'fused'});self.assertEqual(set(terms),{'fused_supervised','total'})
        self.assertIs(terms['total'],terms['fused_supervised']);terms['total'].backward()
        self.assertTrue(any(p.grad is not None for p in m.backbone.HGNN_Branch.parameters()))
        self.assertFalse(any(n.startswith('backbone.CNN_Branch.') for n,_ in m.named_parameters()))

    def test_ordinary_dense_sparse_matches_independent_matrix_and_ties(self):
        q=torch.eye(6);a=ordinary_graph(q,1,6).to_dense().numpy()
        w=np.ones((6,6));w[0,5]=w[5,0]=0;d=w.sum(1)**-.5
        np.testing.assert_allclose(a,d[:,None]*w*d[None,:],rtol=1e-7,atol=1e-7)
        torch.testing.assert_close(ordinary_graph(q.to_sparse(),1,6).to_dense(),torch.from_numpy(a),rtol=0,atol=0)
        torch.testing.assert_close(ordinary_graph(torch.ones(4,1),2,2).to_dense(),torch.ones(1,1),rtol=0,atol=0)
        for bad in (-q,q*float('nan'),torch.cat((q,torch.zeros(6,1)),1),q*2):
            with self.assertRaises(ValueError):ordinary_graph(bad,1,6)

    def test_graph_controls_preserve_sources_and_only_graph_path_changes(self):
        items=data();before={n:(tensor_hash(i['Q']),tensor_hash(i['A'])) for n,i in items.items()};name=next(iter(items))
        reference=model(VARIANTS[0],items).eval()
        with torch.no_grad():baseline=forward(reference,name,items[name])
        for variant in (VARIANTS[4],VARIANTS[5]):
            controlled,binding=controlled_items(items,variant);m=model(variant,items).eval()
            with torch.no_grad():out=forward(m,name,controlled[name])
            torch.testing.assert_close(out['cnn'],baseline['cnn'],rtol=0,atol=0)
            self.assertGreater(float((out['graph']-baseline['graph']).abs().max()),1e-8)
            self.assertNotEqual(binding[name]['source_A_sha256'],binding[name]['effective_A_sha256'])
            if variant.endswith('identity'):torch.testing.assert_close(controlled[name]['A'],torch.eye(items[name]['A'].shape[0],device=DEVICE),rtol=0,atol=0)
        self.assertEqual(before,{n:(tensor_hash(i['Q']),tensor_hash(i['A'])) for n,i in items.items()})

    def test_checkpoint_signature_policy_and_legacy_loader_rejection(self):
        items=data()
        with tempfile.TemporaryDirectory() as temp:
            for variant in VARIANTS:
                path=Path(temp)/(variant+'.pt');m=model(variant,items).eval();save_model_checkpoint(m,path)
                restored=load_ablation(path,device=DEVICE,expected_variant=variant)
                self.assertEqual(_tree_fingerprint(m.state_dict()),_tree_fingerprint(restored.state_dict()))
                with self.assertRaises(RuntimeError):load_model_checkpoint(path,device=DEVICE)
                other=next(v for v in VARIANTS if v!=variant)
                with self.assertRaises(ValueError):load_ablation(path,device=DEVICE,expected_variant=other)
            path=Path(temp)/(VARIANTS[0]+'.pt');state=torch.load(path,weights_only=True);state['_branch_graph_signature'][1]=4;torch.save(state,path)
            with self.assertRaisesRegex(ValueError,'signature'):load_ablation(path,device=DEVICE)

    def test_all_variants_train_calibrate_reload_and_completed_resume(self):
        original=data()
        with tempfile.TemporaryDirectory() as temp:
            for variant in VARIANTS:
                items,binding=controlled_items(original,variant);opts=ablation_settings(settings(),variant)
                provenance={**PROVENANCE,'ablation_graph_binding':binding};runtime=AblationRuntime(variant);path=Path(temp)/variant
                result=run_training(opts,items,path,provenance,SOURCE,device=DEVICE,runtime_adapter=runtime)
                state=checkpoints(path);self.assertEqual(result['result']['optimizer_steps'],12)
                heads={'fused','graph','cnn'} if variant.startswith('dual_legacy_') else {'fused'}
                for phases in state['best_probabilities'].values():
                    for q in phases.values():self.assertEqual(set(q),heads)
                self.assertTrue(all(sum(t['changed_elements'].get(group,0) for t in state['trace'])>0 for group in state['trace'][0]['changed_elements']))
                before=hashes(path)
                resumed=run_training(opts,items,path,provenance,SOURCE,device=DEVICE,runtime_adapter=runtime,resume=True)
                self.assertEqual(resumed['operation'],'verified_completed');self.assertEqual(resumed['updates_this_call'],0);self.assertEqual(before,hashes(path))
                with self.assertRaises(ValueError):run_training(opts,items,path,provenance,SOURCE,device=DEVICE,runtime_adapter=AblationRuntime(next(v for v in VARIANTS if v!=variant)),resume=True)
                if variant==VARIANTS[4]:
                    changed=copy.deepcopy(items);changed[next(iter(changed))]['A']=changed[next(iter(changed))]['A']*.5
                    with self.assertRaisesRegex(ValueError,'Q/A binding'):
                        load_ablation_selected(path/'primary.pt',changed,path/f'raw-validation-{state["best_step"]:06d}.pt',device=DEVICE)

    def test_mid_epoch_resume_exact_and_missing_runtime_rejected(self):
        variant='graph_fused_hypergraph';items,binding=controlled_items(data(),variant);opts=ablation_settings(settings(),variant)
        provenance={**PROVENANCE,'ablation_graph_binding':binding};runtime=AblationRuntime(variant)
        with tempfile.TemporaryDirectory() as temp:
            ref=Path(temp)/'ref';path=Path(temp)/'resume'
            run_training(opts,items,ref,provenance,SOURCE,device=DEVICE,runtime_adapter=runtime)
            run_training(opts,items,path,provenance,SOURCE,device=DEVICE,runtime_adapter=runtime,stop_after=5)
            result=run_training(opts,items,path,provenance,SOURCE,device=DEVICE,runtime_adapter=runtime,resume=True)
            self.assertEqual(result['updates_this_call'],7)
            a,b=checkpoints(ref),checkpoints(path)
            for key in ('model','optimizer','scheduler','rng','trace','history','best_model','best_probabilities','best_calibration'):
                self.assertEqual(_tree_fingerprint(a[key]),_tree_fingerprint(b[key]),key)
            with self.assertRaises(ValueError):run_training(opts,items,Path(temp)/'bad',provenance,SOURCE,device=DEVICE)


if __name__=='__main__':unittest.main()
