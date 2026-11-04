import copy,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from joint_baseline import forward,_item_fingerprint
from joint_training import run_training,_tree_fingerprint
from loadData.dataset_registry import file_hash
from loadData.target_label_protocol import make_target_family,phase_view
from model import utils
from model.research_config import ResearchConfig
from research_runtime import write_json
from target_bn_calibration import OBS_KEYS,GRAPH_POLICY,backbone_id
from target_initialization import initialize_target,METHODS
from target_data_context import load_target_context
from target_training import (TargetRuntime,target_settings,selected_outputs,loss_terms,features,HEAD_ONLY)
import test_target_initialization as initialization_tests
from test_joint_training import SOURCE,PROVENANCE,checkpoints,hashes
from test_source_training import SourceRuntime,source_settings
from test_calibrated_selection import items as source_items

DEVICE=os.environ.get('G4G_DEVICE','cpu');torch.set_num_threads(1)


def fixture(root):
    side=18;bands=7;classes=9;pixels=side*side;regions=6
    rng=torch.Generator().manual_seed(125)
    item={'name':'xuzhou','height':side,'width':side,'bands':bands,'class_num':classes,'superpixel_scale':50,
        'Q':torch.nn.functional.one_hot(torch.arange(pixels)%regions,regions).float().to_sparse().coalesce(),
        'A':torch.eye(regions)+torch.ones(regions,regions)/regions,
        'net_input':torch.rand(side,side,bands+2,generator=rng),'wavelengths':torch.linspace(450,950,bands),'fwhm':None}
    obs=root/'observation.pt';torch.save(item,obs)
    record=root/'record.json';write_json(record,{'policy':GRAPH_POLICY,'target':'xuzhou','graph_fit_labels':0,
        'observation_sha256':file_hash(obs),'prepared_fields':_item_fingerprint(item)})
    labels=(torch.arange(pixels)%classes+1).reshape(side,side).numpy()
    family=root/'family.json';write_json(family,make_target_family(labels,'xuzhou',64))
    return load_target_context('xuzhou',obs,record,file_hash(record),family,file_hash(family),per_class=10,support_seed=64), (obs,record,family)


class TargetTrainingTests(unittest.TestCase):
    def test_context_phase_budget_pins_and_no_query_or_unreleased_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(data,provenance,guard),paths=fixture(root);item=data['xuzhou']
            self.assertEqual(len(item['train_labels']),90);self.assertEqual(len(item['val_labels']),45)
            self.assertFalse(any('query' in k or 'test' in k for k in item));guard()
            with self.assertRaises(ValueError):load_target_context('paviaU',paths[0],paths[1],file_hash(paths[1]),paths[2],file_hash(paths[2]),per_class=10,support_seed=64)
            with paths[0].open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):guard()
        for key,value in [('epochs',1000),('support_per_class',9),('test_scoring',True)]:
            settings=target_settings('xuzhou',METHODS[0]);settings[key]=value
            with self.assertRaises(ValueError):TargetRuntime(settings,{},'/unused')

    def test_selected_rows_independent_fp64_full_masked_loss_and_gradient_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            (data,_,_),_=fixture(Path(tmp));item=data['xuzhou']
            item={k:v.double() if torch.is_tensor(v) and v.is_floating_point() else v for k,v in item.items()}
            item['A']=item['A'].to_sparse().coalesce()
            model,_=initialize_target('xuzhou',METHODS[0],model_seed=64,head_seed=1064);model.double().train()
            other=copy.deepcopy(model);config=ResearchConfig.from_dict(target_settings('xuzhou',METHODS[0])['config'])
            all_outputs=forward(model,'xuzhou',item);ids=item['train_indices'];labels=item['train_labels'];n=18*18
            onehot=torch.zeros(n,9,dtype=torch.double);onehot[ids,labels]=1
            mask=torch.zeros_like(onehot);mask[ids]=1
            old=sum(utils.compute_loss(v,onehot,mask,class_balance=True) for v in all_outputs[:3])
            new=loss_terms(selected_outputs(other,'xuzhou',item,'train'),labels,config)['total']
            torch.testing.assert_close(old,new,rtol=1e-12,atol=1e-12);old.backward();new.backward()
            for (name,a),(other_name,b) in zip(model.named_parameters(),other.named_parameters()):
                self.assertEqual(name,other_name);torch.testing.assert_close(a.grad,b.grad,rtol=1e-9,atol=1e-11)

    def test_all_methods_scope_exact_cpu_resume_selection_and_readonly_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(data,provenance,guard),_=fixture(root)
            source=initialization_tests.TargetInitializationTests().source(root)
            spec=dict(source_path=str(source[0]),source_weight_sha256=source[1],source_sidecar_sha256=source[2])
            for method in METHODS:
                opts=target_settings('xuzhou',method,device=DEVICE);kwargs={} if method==METHODS[0] else spec
                full=root/(method+'-full');split=root/(method+'-split')
                runtime=TargetRuntime(opts,data,full,source_spec=kwargs)
                result=run_training(opts,data,full,provenance,SOURCE,device=DEVICE,runtime_adapter=runtime)
                a=TargetRuntime(opts,data,split,source_spec=kwargs)
                run_training(opts,data,split,provenance,SOURCE,device=DEVICE,runtime_adapter=a,stop_after=5)
                b=TargetRuntime(opts,data,split,source_spec=kwargs)
                resumed=run_training(opts,data,split,provenance,SOURCE,device=DEVICE,runtime_adapter=b,resume=True)
                self.assertEqual(resumed['updates_this_call'],7)
                for key in ('model','optimizer','scheduler','rng','trace','history','best_model','best_probabilities'):
                    # Tiny native CUDA fixture also checks exact continuation; this
                    # is not a general full-scene GPU bitwise guarantee.
                    self.assertEqual(_tree_fingerprint(checkpoints(full)[key]),_tree_fingerprint(checkpoints(split)[key]),(method,key))
                events=json.loads((full/'history.json').read_text());self.assertEqual([e['step'] for e in events],[1,4,7,10,12])
                self.assertTrue(all(e['selection_score']==-e['val_loss'] for e in events))
                self.assertEqual(result['result']['selected_step'],min(events,key=lambda e:e['val_loss'])['step'])
                self.assertGreater(events[0]['val_loss'],1)
                saved=hashes(split)
                verified=run_training(opts,data,split,provenance,SOURCE,device=DEVICE,runtime_adapter=TargetRuntime(opts,data,split,source_spec=kwargs),resume=True)
                self.assertEqual(verified['updates_this_call'],0);self.assertEqual(saved,hashes(split))
                last=checkpoints(full)
                if method==HEAD_ONLY:
                    self.assertTrue(all(t['gradient_norms']['graph']==0 and t['gradient_norms']['adapter']==0 and t['graph_changed_elements']==0 for t in last['trace']))
                    self.assertTrue(all(t['feature_bank_sha256']==file_hash(full/'feature-bank.json') for t in last['trace']))
                else:self.assertTrue(all(t['gradient_norms']['graph']>0 and t['graph_changed_elements']>0 for t in last['trace']))
                with (split/'primary.pt').open('ab') as stream:stream.write(b'changed')
                with self.assertRaises(ValueError):run_training(opts,data,split,provenance,SOURCE,device=DEVICE,runtime_adapter=TargetRuntime(opts,data,split,source_spec=kwargs),resume=True)
            guard()

    def test_frozen_cache_equivalence_tamper_and_selected_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(data,provenance,_),_=fixture(root);source=initialization_tests.TargetInitializationTests().source(root)
            spec=dict(source_path=str(source[0]),source_weight_sha256=source[1],source_sidecar_sha256=source[2])
            opts=target_settings('xuzhou',HEAD_ONLY,device=DEVICE);run=root/'run';run.mkdir()
            runtime=TargetRuntime(opts,data,run,source_spec=spec);config=ResearchConfig.from_dict(opts['config'])
            model=runtime.build_model(config,{'xuzhou':9});optimizer=runtime.optimizer(model,config)
            self.assertEqual({id(p) for g in optimizer.param_groups for p in g['params']},{id(p) for p in model.parameters() if p.requires_grad})
            bank=runtime._bank(model,data['xuzhou'],cursor=0)
            with torch.no_grad():
                cached=selected_outputs(model,'xuzhou',data['xuzhou'],'train',bank['train'])
                uncached=selected_outputs(model,'xuzhou',data['xuzhou'],'train')
            for a,b in zip(cached,uncached):torch.testing.assert_close(a,b,rtol=0,atol=0)
            before=backbone_id(model);runtime.train_step_at(0,model,optimizer,'xuzhou',data['xuzhou'],config)
            self.assertEqual(before,backbone_id(model))
            with (run/'feature-bank.pt').open('ab') as stream:stream.write(b'changed')
            with self.assertRaises(ValueError):runtime._bank(model,data['xuzhou'])
            # Fresh run has persisted source/calibration and selected head binding.
            path=root/'complete';rt=TargetRuntime(opts,data,path,source_spec=spec)
            result=run_training(opts,data,path,provenance,SOURCE,device=DEVICE,runtime_adapter=rt)
            step=result['result']['selected_step'];raw=path/f'raw-validation-{step:06d}.pt'
            selected=rt.load_selected(path/'primary.pt',data,raw,device=DEVICE)
            self.assertTrue(all(not p.requires_grad for p in selected.backbone.parameters()))
            bad=copy.deepcopy(data);bad['xuzhou']['A']*=2
            with self.assertRaises(ValueError):rt.load_selected(path/'primary.pt',bad,raw,device=DEVICE)

    def test_optional_hooks_preserve_existing_source_and_reject_metadata_collision(self):
        class ExplicitSource(SourceRuntime):
            def optimizer(self,model,config):
                heads=list(model.heads.parameters())+list(model.hgnn_heads.parameters())+list(model.cnn_heads.parameters())
                return torch.optim.Adam([{'params':model.backbone.parameters(),'lr':config.training.lr},
                    {'params':heads,'lr':config.training.head_lr}],weight_decay=config.training.weight_decay)
            def checkpoint_metadata(self,model):return {}
        with tempfile.TemporaryDirectory() as tmp:
            opts=source_settings('xuzhou');data={n:v for n,v in source_items().items() if n in opts['source_tasks']}
            for label,runtime in [('old',SourceRuntime()),('hooks',ExplicitSource())]:
                run_training(opts,data,Path(tmp)/label,PROVENANCE,SOURCE,runtime_adapter=runtime)
            for key in ('model','optimizer','scheduler','rng','trace','history','best_model','best_probabilities'):
                self.assertEqual(_tree_fingerprint(checkpoints(Path(tmp)/'old')[key]),_tree_fingerprint(checkpoints(Path(tmp)/'hooks')[key]))
            runtime=ExplicitSource();runtime.checkpoint_metadata=lambda model:{'joint_runtime':{}}
            with self.assertRaisesRegex(ValueError,'metadata collides'):
                run_training(opts,data,Path(tmp)/'collision',PROVENANCE,SOURCE,runtime_adapter=runtime)


if __name__=='__main__':unittest.main()
