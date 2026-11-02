"""M4-B1 default compatibility, actual branch wiring and training restoration."""
import contextlib
import copy
from functools import lru_cache
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from model.VCASAHGCN import MultiDatasetVCASAHGCN, VCASAHGCN
from model.checkpoint_io import save_model_checkpoint, load_model_checkpoint
from model.graph_state import GraphState
from model.region_pixel_alignment import pool_regions
from model.region_residual_decomposition import RegionResidualDecomposition
from model.research_config import ResearchConfig
from train_spatial import SpatialConfig, execute
from train_vca_multidataset import build_parser
from research_runtime import code_snapshot, tensor_hash
from transfer.g3_training import build_model, build_target, state_hash
from transfer.development_training import prepare_scene, adapt
from transfer.run_provenance import verify_run_artifacts


ROOT = Path(__file__).resolve().parents[1]
BASE = 'ca78fff9c8f6d032de4dc8985603e533485035fa'


@lru_cache(maxsize=1)
def parent_module():
    module = types.ModuleType('m4b_parent')
    source = subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE + ':model/VCASAHGCN.py'], text=True)
    exec(compile(source, 'm4b_parent.py', 'exec'), module.__dict__)
    return module


class RegionResidualIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get('M4_DEVICE', 'cpu')

    def model(self, mode='none', **kwargs):
        return MultiDatasetVCASAHGCN(
            {'a': 3}, latent_channels=8, adapter_embed_dim=8, norm_type='group_norm',
            graph_mode=kwargs.pop('graph_mode', 'learned'), region_residual_mode=mode,
            **kwargs).to(self.device)

    def inputs(self):
        x = torch.linspace(-1, 2, 6 * 7 * 8, device=self.device).reshape(6, 7, 8)
        return x, torch.linspace(450, 950, 6, device=self.device)

    def static(self):
        q = torch.nn.functional.one_hot(torch.arange(42, device=self.device) % 6, 6).float()
        return q, torch.eye(6, device=self.device)

    def test_default_matches_actual_parent_initialization_outputs_gradients_and_buffers(self):
        options = dict(latent_channels=8, adapter_embed_dim=8, norm_type='domain_bn', graph_mode='learned')
        for interaction in ('none', 'gated'):
            torch.manual_seed(64)
            old = parent_module().MultiDatasetVCASAHGCN({'a': 3}, interaction_mode=interaction, **options).to(self.device)
            torch.manual_seed(64)
            new = MultiDatasetVCASAHGCN({'a': 3}, interaction_mode=interaction, **options).to(self.device)
            self.assertIsNone(new.backbone.region_residual)
            self.assertEqual(set(old.state_dict()), set(new.state_dict()))
            for key, value in old.state_dict().items():
                torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0, msg=key)
            x, wave = self.inputs(); a=x.clone().requires_grad_(); b=x.clone().requires_grad_()
            expected=old('a', a, wavelengths=wave, superpixel_scale=9, return_graph_info=True)
            actual=new('a', b, wavelengths=wave, superpixel_scale=9, return_graph_info=True)
            self.assertEqual(set(expected[3]), set(actual[3]))
            for left, right in zip(expected[:3], actual[:3]):
                torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
            for outputs in (expected, actual):
                sum(-p[:,0].clamp_min(1e-8).log().mean() for p in outputs[:3]).backward()
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6)
            for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
                self.assertEqual(p.grad is None, q.grad is None, name)
                if p.grad is not None:
                    torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6, msg=name)
            for name, value in old.named_buffers():
                torch.testing.assert_close(value, dict(new.named_buffers())[name], rtol=0, atol=0)

    def test_enabled_modes_preserve_existing_weight_and_head_rng_initialization(self):
        torch.manual_seed(512); reference=self.model(); rng=torch.random.get_rng_state().clone()
        for mode in RegionResidualDecomposition.MODES:
            torch.manual_seed(512); model=self.model(mode)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0, msg=name)
        with self.assertRaises(ValueError): self.model('invalid')

    def test_single_q_original_region_pool_and_effective_cnn_input_in_both_paths(self):
        x, wave=self.inputs()
        for interaction in ('none', 'gated'):
            for mode in RegionResidualDecomposition.MODES[1:]:
                model=self.model(mode, interaction_mode=interaction).eval(); backbone=model.backbone
                layer=backbone.region_residual
                with torch.no_grad():
                    if layer.projection is not None: layer.projection.weight[:,64:].copy_(torch.eye(64,device=self.device)*0.25)
                    if layer.control is not None: layer.control[2].bias.fill_(0.2)
                captured={}
                def capture(_, args, out): captured.update(x=args[0].detach().clone(), q=args[1], out=out)
                handle=layer.register_forward_hook(capture)
                try:
                    with patch.object(backbone.learnable_graph,'build_graph',wraps=backbone.learnable_graph.build_graph) as graph, \
                         patch.object(backbone,'forward_stages',wraps=backbone.forward_stages) as stages, \
                         patch('model.region_residual_decomposition.pool_regions',wraps=pool_regions) as pool, \
                         patch('model.VCASAHGCN.pool_regions',side_effect=AssertionError('duplicate legacy pool')):
                        outputs=model('a',x,wavelengths=wave,superpixel_scale=9,return_graph_info=True,return_diagnostics=True)
                        self.assertEqual(graph.call_count,1); self.assertEqual(pool.call_count,1)
                        cnn, regions, q, _=stages.call_args.args[:4]
                        self.assertIs(q,captured['q']); self.assertIs(pool.call_args.args[0],q)
                        torch.testing.assert_close(regions,pool_regions(q,captured['x']))
                        self.assertGreater(float((cnn.reshape(-1,64)-captured['x']).abs().max()),1e-6)
                        self.assertEqual(outputs[3]['region_residual']['mode'],mode)
                        self.assertTrue(all(isinstance(v,(str,int,float)) for v in outputs[3]['region_residual'].values()))
                        self.assertIsNone(captured['out'].residual)
                        self.assertIsNone(captured['out'].region_at_pixels)
                finally: handle.remove()

    def test_all_modes_static_learned_and_injected_graphs_keep_logits_and_regularizers(self):
        x,wave=self.inputs(); q,a=self.static()
        for mode in RegionResidualDecomposition.MODES:
            for graph in ('static','learned','injected'):
                model=self.model(mode,graph_mode='static' if graph=='static' else 'learned').eval()
                kwargs=dict(wavelengths=wave,superpixel_scale=9,return_graph_info=True,return_logits=True)
                if graph=='static': kwargs.update(Q=q,A=a)
                if graph=='injected':
                    loss=torch.tensor(.2,device=self.device,requires_grad=True)
                    state=GraphState(q.to_sparse().coalesce(),a.to_sparse().coalesce(),torch.zeros(6,2,device=self.device),
                                     (6,7),(2,3),regularizers={'compactness_loss':loss})
                    kwargs['graph_state']=state
                out=model('a',x,**kwargs)
                self.assertTrue(all(p.shape==(42,3) and torch.isfinite(p).all() for p in out[:3]))
                if graph=='injected': self.assertIs(out[3]['compactness_loss'],loss)
                elif graph=='learned': self.assertTrue(out[3]['compactness_loss'].requires_grad)
                if graph=='learned' and mode=='none': self.assertNotIn('region_residual',out[3])

    def test_cnn_only_loss_reaches_m4_projection_and_learned_assignment(self):
        model=self.model('x_residual'); layer=model.backbone.region_residual
        with torch.no_grad(): layer.projection.weight[:,64:].copy_(torch.eye(64,device=self.device)*.3)
        x,wave=self.inputs(); out=model('a',x,wavelengths=wave,superpixel_scale=9,return_logits=True)
        out[2][:,0].square().mean().backward()
        for p in (layer.projection.weight,model.backbone.learnable_graph.pixel_query.weight):
            self.assertIsNotNone(p.grad); self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().max()),0)
        self.assertTrue(all(p.grad is None for p in model.backbone.HGNN_Branch.parameters()))

    def test_shared_config_cli_and_spatial_config_roundtrip_and_mode_hash(self):
        hashes=[]
        for mode in RegionResidualDecomposition.MODES:
            args=build_parser().parse_args(['--region-residual-mode',mode,'--device','cpu'])
            config=ResearchConfig.from_legacy(args)
            self.assertEqual(config.model.region_residual_mode,mode)
            restored=ResearchConfig.from_legacy(config.trainer_args('/tmp/model.pt','/tmp/metrics.json'))
            self.assertEqual(restored.to_dict(),config.to_dict()); hashes.append(config.config_hash)
            spatial=SpatialConfig({'model':{'graph_mode':'learned','region_residual_mode':mode},'training':{'steps_per_epoch':2}})
            self.assertEqual(SpatialConfig(spatial.to_dict()).to_dict(),spatial.to_dict())
            self.assertEqual(config.model_kwargs()['region_residual_mode'],mode)
        self.assertEqual(len(set(hashes)),5)
        self.assertEqual(ResearchConfig.from_dict({}).model.region_residual_mode,'none')
        with self.assertRaises(ValueError): ResearchConfig.from_dict({'model':{'region_residual_mode':'bad'}})
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            build_parser().parse_args(['--region-residual-mode','bad'])

    def test_all_mode_checkpoint_sidecars_restore_actual_architecture_and_outputs(self):
        x,wave=self.inputs()
        with tempfile.TemporaryDirectory() as directory:
            for mode in RegionResidualDecomposition.MODES:
                model=self.model(mode).eval()
                with torch.no_grad():
                    if model.backbone.region_residual is not None:
                        for p in model.backbone.region_residual.parameters(): p.add_(.01)
                path=Path(directory)/(mode+'.pt'); save_model_checkpoint(model,path)
                sidecar=json.loads(Path(str(path)+'.json').read_text())
                self.assertEqual(sidecar['backbone_kwargs']['region_residual_mode'],mode)
                restored=load_model_checkpoint(path,self.device)
                self.assertEqual(restored.backbone.region_residual_mode,mode)
                self.assertEqual(restored.export_config(),model.export_config())
                with torch.no_grad():
                    a=model('a',x,wavelengths=wave,superpixel_scale=9)
                    b=restored('a',x,wavelengths=wave,superpixel_scale=9)
                for left,right in zip(a,b): torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)

    def test_old_sidecar_and_explicit_raw_checkpoint_restore_default_none(self):
        model=parent_module().MultiDatasetVCASAHGCN({'a':3},latent_channels=8,adapter_embed_dim=8,
              norm_type='group_norm',graph_mode='learned').to(self.device).eval()
        x,wave=self.inputs()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.pt'; save_model_checkpoint(model,path)
            self.assertNotIn('region_residual_mode',json.loads(Path(str(path)+'.json').read_text())['backbone_kwargs'])
            old_bytes=Path(str(path)+'.json').read_bytes()
            restored=load_model_checkpoint(path,self.device)
            self.assertEqual(restored.backbone.region_residual_mode,'none')
            self.assertEqual(old_bytes,Path(str(path)+'.json').read_bytes())
            raw=Path(directory)/'raw.pt'; torch.save(model.state_dict(),raw)
            raw_model=load_model_checkpoint(raw,self.device,config=model.export_config()['backbone_kwargs'])
            for candidate in (restored,raw_model):
                self.assertEqual(set(candidate.state_dict()),set(model.state_dict()))
                with torch.no_grad():
                    for a,b in zip(model('a',x,wavelengths=wave,superpixel_scale=9),candidate('a',x,wavelengths=wave,superpixel_scale=9)):
                        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            enabled=self.model('x_residual'); torch.save(enabled.state_dict(),raw)
            with self.assertRaises(RuntimeError):
                load_model_checkpoint(raw,self.device,config=model.export_config()['backbone_kwargs'])

    def test_single_dataset_compatibility_wrapper_forwards_m4_mode(self):
        q,a=self.static(); x,wave=self.inputs()
        model=VCASAHGCN(6,7,8,3,q,a,.8,wavelengths=wave,latent_channels=8,adapter_embed_dim=8,
                       norm_type='group_norm',region_residual_mode='residual_only').to(self.device)
        outputs=model(x)
        self.assertEqual(model.model.backbone.region_residual_mode,'residual_only')
        self.assertTrue(all(p.shape==(42,3) for p in outputs))

    def test_g3_transfers_matching_mode_and_rejects_same_shape_semantic_mismatch(self):
        from test_transfer_training import fixture
        config,base,(scene,_,target,_,_)=fixture()
        base['model']['region_residual_mode']='x_residual'
        source,_=build_model(base,scene.dataset,9,64,self.device)
        model,_=build_target(base,target,'source_pretrained_full_finetune',source,model_seed=64,head_seed=66,device=self.device)
        self.assertEqual(state_hash(model.backbone),state_hash(source.backbone))
        for mode in ('x_region','none','residual_only'):
            other=copy.deepcopy(base); other['model']['region_residual_mode']=mode
            with self.assertRaisesRegex(ValueError,'region_residual_mode must match'):
                build_target(other,target,'source_pretrained_full_finetune',source,model_seed=64,head_seed=66,device=self.device)

    def test_existing_spatial_runner_optimizes_projection_reloads_and_resumes(self):
        from test_spatial_training import fixture
        config,cube,labels=fixture(); value=config.to_dict(); value['device']=self.device
        value['model']['region_residual_mode']='x_residual'; config=SpatialConfig(value)
        retained={}
        def initializer(model):
            retained['model']=model
            retained['weight']=model.backbone.region_residual.projection.weight.detach().clone()
        loader=lambda:(cube,labels,{'cube':tensor_hash(cube),'labels':tensor_hash(labels)},None)
        with tempfile.TemporaryDirectory() as directory:
            code=code_snapshot(); result=execute(config,directory,code,loader,model_initializer=initializer)
            self.assertEqual(result['status'],'completed')
            model=retained['model']; current=model.backbone.region_residual.projection.weight
            self.assertGreater(float((current-retained['weight']).abs().max()),0)
            path=Path(directory)/result['run_id']/'attempts'/result['attempt']/'model.pt'
            restored=load_model_checkpoint(path,self.device)
            self.assertEqual(restored.backbone.region_residual_mode,'x_residual')
            self.assertEqual(restored.training_config['spatial_config']['model']['region_residual_mode'],'x_residual')
            with patch.object(torch.optim,'Adam',side_effect=AssertionError('must skip completed run')):
                resumed=execute(config,directory,code,loader,True)
            self.assertEqual(resumed['status'],'skipped')

    def test_existing_development_runner_saves_mode_optimizer_budget_and_selection(self):
        from test_development_training import DevelopmentTrainingTests
        from test_transfer_training import fixture
        plan,record,cube,bands=DevelopmentTrainingTests().fixture()
        scene=prepare_scene(cube,bands,plan,record); _,base,_=fixture()
        base['model']['region_residual_mode']='x_residual'
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'development'
            result=adapt(scene,record,base,folder,steps=2,val_every=1,seed=64,device=self.device)
            self.assertEqual(result['resolved_config']['model']['region_residual_mode'],'x_residual')
            self.assertEqual(result['runtime_config']['optimizer_steps'],2)
            self.assertEqual(verify_run_artifacts(folder)['status'],'verified')
            self.assertFalse(result['test_evaluated'])
            restored=load_model_checkpoint(folder/'model.pt',self.device)
            self.assertEqual(restored.backbone.region_residual_mode,'x_residual')
            self.assertEqual(result['selected_step'],min(result['validation'],key=lambda r:r['nll'])['step'])


if __name__ == '__main__':
    unittest.main()
