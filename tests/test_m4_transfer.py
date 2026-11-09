import copy,json,os,tempfile,unittest
from pathlib import Path
import torch
from joint_training import run_training,_tree_fingerprint
from source_training import source_settings,SourceRuntime
from target_training import target_settings,TargetRuntime,HEAD_ONLY
from target_initialization import initialize_target,domain_keys,state_id,METHODS
from m4_transfer import (MODES,source_options,target_options,M4SourceRuntime,M4TargetRuntime,
    initialize_m4_target,graph_binding)
from loadData.dataset_registry import file_hash
from model.research_config import ResearchConfig
from model.checkpoint_io import load_model_checkpoint
from research_runtime import write_json
from test_calibrated_selection import items as source_items
from test_target_training import fixture
from test_joint_training import SOURCE,checkpoints,hashes

DEVICE=os.environ.get('M4G4_DEVICE','cpu');torch.set_num_threads(1)


def source_case(root,mode):
    opts=source_options('xuzhou',mode);items={n:i for n,i in source_items().items() if n in opts['source_tasks']}
    provenance={'policy':'selective_acc1p_source_v1','sources':opts['source_tasks'],'held_out':'xuzhou',
        'target_payload_accessed':False,'source_identity':'a'*64}
    if mode!='none':provenance.update(m4_mode=mode,m4_graph_binding=graph_binding(items))
    directory=root/('source-'+mode);runtime=M4SourceRuntime()
    result=run_training(opts,items,directory,provenance,SOURCE,device=DEVICE,runtime_adapter=runtime)
    spec={'source_path':str(directory/'primary.pt'),'source_weight_sha256':file_hash(directory/'primary.pt'),
        'source_sidecar_sha256':file_hash(directory/'primary.pt.json')}
    return spec,directory,items,provenance,result


class M4TransferTests(unittest.TestCase):
    def test_none_source_and_target_exact_legacy_settings_state_and_continuation(self):
        self.assertEqual(source_options('xuzhou','none'),source_settings('xuzhou'))
        self.assertEqual(target_options('xuzhou',METHODS[0],'none',device=DEVICE),target_settings('xuzhou',METHODS[0],device=DEVICE))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);spec,src,source_data,prov,_=source_case(root,'none');opts=source_settings('xuzhou')
            run_training(opts,source_data,root/'old-source',prov,SOURCE,device=DEVICE,runtime_adapter=SourceRuntime())
            (data,target_prov,_),_=fixture(root)
            for label,klass in [('old',TargetRuntime),('new',M4TargetRuntime)]:
                options=target_settings('xuzhou',METHODS[1],device=DEVICE);directory=root/label
                run_training(options,data,directory,target_prov,SOURCE,device=DEVICE,runtime_adapter=klass(options,data,directory,source_spec=spec))
            if DEVICE=='cpu':
                for a,b in [(src,root/'old-source'),(root/'old',root/'new')]:
                    for key in ('model','optimizer','scheduler','rng','trace','history','best_model','best_probabilities'):
                        self.assertEqual(_tree_fingerprint(checkpoints(a)[key]),_tree_fingerprint(checkpoints(b)[key]),key)
            model,a=initialize_target('xuzhou',METHODS[1],model_seed=64,head_seed=1064,device=DEVICE,**spec)
            other,b=initialize_m4_target('xuzhou',METHODS[1],'none',model_seed=64,head_seed=1064,device=DEVICE,**spec)
            self.assertEqual(state_id(model),state_id(other));self.assertEqual(a,b)

    def test_common_initialization_rng_source_copy_and_fresh_domains_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            base,_=initialize_target('xuzhou',METHODS[0],model_seed=64,head_seed=1064,device=DEVICE)
            for mode in MODES[1:]:
                spec,folder,_,_,_=source_case(root,mode);source=load_model_checkpoint(folder/'primary.pt',device=DEVICE)
                before=state_id(source);rng=torch.get_rng_state().clone()
                cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if DEVICE=='cuda' else None
                scratch,_=initialize_m4_target('xuzhou',METHODS[0],mode,model_seed=64,head_seed=1064,device=DEVICE)
                model,receipt=initialize_m4_target('xuzhou',METHODS[1],mode,model_seed=64,head_seed=1064,device=DEVICE,**spec)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                if cuda is not None:self.assertTrue(all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())))
                for name,value in scratch.state_dict().items():
                    if 'region_residual' not in name:torch.testing.assert_close(value,base.state_dict()[name],rtol=0,atol=0)
                for name,value in model.backbone.state_dict().items():
                    expected=scratch.backbone.state_dict()[name] if name in domain_keys(model.backbone) else source.backbone.state_dict()[name]
                    torch.testing.assert_close(value,expected,rtol=0,atol=0)
                for family in ('heads','hgnn_heads','cnn_heads'):
                    for name,value in getattr(model,family).state_dict().items():torch.testing.assert_close(value,getattr(scratch,family).state_dict()[name],rtol=0,atol=0)
                self.assertEqual(state_id(source),before);self.assertFalse(receipt['source_heads_copied'])
                self.assertEqual(receipt['m4_mode'],mode);self.assertEqual(file_hash(spec['source_path']),spec['source_weight_sha256'])

    def test_active_source_gradients_graph_mode_and_pair_provenance_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            specs={}
            for mode in MODES[1:]:
                spec,folder,data,_,result=source_case(root,mode);specs[mode]=spec
                trace=json.loads((folder/'trace.json').read_text())
                self.assertTrue(all(r['m4_parameter_gradient_norm']>0 and r['m4_changed_elements']>0 for r in trace))
                if mode=='x_residual':self.assertGreater(trace[-1]['m4_hint_weight_norm'],0)
                bad=copy.deepcopy(data);bad[next(iter(bad))]['A']*=2
                with self.assertRaisesRegex(ValueError,'graph/mode binding'):
                    M4SourceRuntime().load_selected(folder/'primary.pt',bad,folder/f'raw-validation-{result["result"]["selected_step"]:06d}.pt',device=DEVICE)
            with self.assertRaises(ValueError):initialize_m4_target('xuzhou',METHODS[1],'x_residual',model_seed=64,head_seed=1064,device=DEVICE,**specs['capacity_control'])
            spec=specs['x_residual'];path=Path(spec['source_path']);raw=path.read_bytes();sidecar=Path(str(path)+'.json');meta=sidecar.read_bytes()
            for kind in ('semantic','target_access','double','nonfinite'):
                bad=root/('bad-'+kind+'.pt');bad.write_bytes(raw);bp=Path(str(bad)+'.json');bp.write_bytes(meta)
                if kind in ('semantic','target_access'):
                    value=json.loads(meta)
                    if kind=='semantic':value['backbone_kwargs']['region_residual_mode']='x_region'
                    else:value['training_config']['provenance']['target_payload_accessed']=True
                    write_json(bp,value)
                else:
                    state=torch.load(bad,weights_only=True)
                    if kind=='double':state={n:v.double() if v.is_floating_point() else v for n,v in state.items()}
                    else:next(v for v in state.values() if v.is_floating_point()).fill_(float('nan'))
                    torch.save(state,bad)
                kwargs=dict(source_path=str(bad),source_weight_sha256=file_hash(bad),source_sidecar_sha256=file_hash(bp))
                with self.assertRaises(ValueError):initialize_m4_target('xuzhou',METHODS[1],'x_residual',model_seed=64,head_seed=1064,device=DEVICE,**kwargs)
            self.assertEqual(path.read_bytes(),raw);self.assertEqual(sidecar.read_bytes(),meta)

    def test_active_target_resume_parameters_and_readonly_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(data,provenance,_),_=fixture(root)
            for mode in MODES[1:]:
                spec,_,_,_,_=source_case(root,mode);opts=target_options('xuzhou',METHODS[1],mode,device=DEVICE)
                full=root/(mode+'-full');split=root/(mode+'-split')
                result=run_training(opts,data,full,provenance,SOURCE,device=DEVICE,runtime_adapter=M4TargetRuntime(opts,data,full,source_spec=spec))
                run_training(opts,data,split,provenance,SOURCE,device=DEVICE,stop_after=5,runtime_adapter=M4TargetRuntime(opts,data,split,source_spec=spec))
                interrupted=checkpoints(split)
                resumed=run_training(opts,data,split,provenance,SOURCE,device=DEVICE,resume=True,runtime_adapter=M4TargetRuntime(opts,data,split,source_spec=spec))
                self.assertEqual(resumed['restored_state_fingerprints'],interrupted['state_fingerprints'])
                if DEVICE=='cpu':
                    for key in ('model','optimizer','scheduler','rng','trace','history','best_model','best_probabilities'):
                        self.assertEqual(_tree_fingerprint(checkpoints(full)[key]),_tree_fingerprint(checkpoints(split)[key]),(mode,key))
                trace=json.loads((full/'trace.json').read_text());self.assertTrue(all(r['m4_parameter_gradient_norm']>0 for r in trace))
                self.assertGreater(sum(r['m4_changed_elements'] for r in trace),0)
                before=hashes(split)
                verified=run_training(opts,data,split,provenance,SOURCE,device=DEVICE,resume=True,runtime_adapter=M4TargetRuntime(opts,data,split,source_spec=spec))
                self.assertEqual(verified['updates_this_call'],0);self.assertEqual(before,hashes(split))
                bad=copy.deepcopy(opts);bad['config']['model']['region_residual_mode']='none'
                with self.assertRaises(ValueError):M4TargetRuntime(bad,data,root/'bad',source_spec=spec)

    def test_m4_headonly_calibration_optimizer_scope_and_frozen_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(data,provenance,_),_=fixture(root);mode='x_residual';spec,_,_,_,_=source_case(root,mode)
            opts=target_options('xuzhou',HEAD_ONLY,mode,device=DEVICE);out=root/'head'
            result=run_training(opts,data,out,provenance,SOURCE,device=DEVICE,runtime_adapter=M4TargetRuntime(opts,data,out,source_spec=spec))
            trace=json.loads((out/'trace.json').read_text());self.assertTrue(all(r['m4_changed_elements']==0 and r['m4_parameter_gradient_norm']==0 for r in trace))
            rt=M4TargetRuntime(opts,data,out,source_spec=spec);step=result['result']['selected_step']
            selected=rt.load_selected(out/'primary.pt',data,out/f'raw-validation-{step:06d}.pt',device=DEVICE)
            self.assertTrue(all(not p.requires_grad for p in selected.backbone.parameters()))
            self.assertFalse(any(m.training for m in selected.backbone.modules()))
            self.assertFalse(selected.training_config['target_initialization']['head_only_requires_target_bn_calibration'])


if __name__=='__main__':unittest.main()
