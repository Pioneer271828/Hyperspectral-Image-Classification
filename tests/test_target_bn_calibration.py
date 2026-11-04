import copy,os,tempfile,types,unittest
from pathlib import Path
import torch
import torch.nn.functional as F
from target_initialization import initialize_target,state_id
from target_bn_calibration import calibrate_target,verify_target_binding,backbone_id,POLICY,CONTEXT,GRAPH_POLICY,OBS_KEYS
from model.checkpoint_io import save_model_checkpoint,load_model_checkpoint
from joint_baseline import forward
from test_calibrated_selection import items

DEVICE=os.environ.get('G4F_DEVICE','cpu');torch.set_num_threads(1)


class TargetBNTests(unittest.TestCase):
    def fixture(self,*,double=False):
        model,_=initialize_target('xuzhou','target_scratch_full',model_seed=64,head_seed=1064,device=DEVICE)
        model.eval();item=items()['xuzhou'];observation={k:item[k] for k in OBS_KEYS}
        if double:
            model.double();observation={k:v.double() if torch.is_tensor(v) else v for k,v in observation.items()}
            observation['A']=observation['A'].to_sparse().coalesce()
        observation={k:v.to(DEVICE) if torch.is_tensor(v) else v for k,v in observation.items()}
        return model,observation

    def calibrate(self,model,obs):
        return calibrate_target(model,obs,mode=POLICY,context=CONTEXT,graph_policy=GRAPH_POLICY,source_state_sha256=state_id(model))

    def test_none_exact_noop_and_only_target_moments_change(self):
        model,obs=self.fixture();self.assertIs(calibrate_target(model)[0],model);before=state_id(model)
        cpu=torch.get_rng_state().clone();cuda=[x.clone() for x in torch.cuda.get_rng_state_all()] if DEVICE=='cuda' else None
        result,receipt=self.calibrate(model,obs);self.assertEqual(state_id(model),before)
        self.assertEqual(len(receipt['layers']),5);self.assertEqual(len(receipt['calibrated_buffers']),10)
        self.assertEqual({r['population_units'] for r in receipt['layers']},{'pixels','regions'})
        for n,v in result.state_dict().items():
            if n not in receipt['calibrated_buffers']:torch.testing.assert_close(v,model.state_dict()[n],rtol=0,atol=0)
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
        if cuda is not None:self.assertTrue(all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())))
        self.assertFalse(any(m.training for m in result.modules()));self.assertEqual(receipt['optimizer_updates'],0)

    def test_independent_functional_population_oracle_fp64_sparse_graph(self):
        # CPUFP64 is the numerical oracle; native GPU tests exercise other paths.
        model,obs=self.fixture(double=True);reference=copy.deepcopy(model)
        for module in reference.modules():
            if isinstance(module,(torch.nn.BatchNorm1d,torch.nn.BatchNorm2d)):
                def independent(self,x):return F.batch_norm(x,None,None,self.weight,self.bias,True,0.,self.eps)
                module.forward=types.MethodType(independent,module)
        with torch.no_grad():expected=forward(reference,'xuzhou',obs)
        result,_=self.calibrate(model,obs)
        with torch.no_grad():actual=forward(result,'xuzhou',obs)
        for a,b in zip(actual[:3],expected[:3]):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-11)

    def test_checkpoint_binding_head_updates_and_graph_tamper(self):
        model,obs=self.fixture();result,receipt=self.calibrate(model,obs)
        verify_target_binding(result,obs,context=CONTEXT,graph_policy=GRAPH_POLICY,source_state_sha256=state_id(model))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'calibrated.pt';save_model_checkpoint(result,path);loaded=load_model_checkpoint(path,device=DEVICE)
            self.assertEqual(state_id(result),state_id(loaded));verify_target_binding(loaded,obs,context=CONTEXT,graph_policy=GRAPH_POLICY,source_state_sha256=state_id(model))
            with torch.no_grad():loaded.heads['xuzhou'].weight.add_(.01)
            verify_target_binding(loaded,obs,context=CONTEXT,graph_policy=GRAPH_POLICY,source_state_sha256=state_id(model),backbone_only=True)
            with self.assertRaises(ValueError):verify_target_binding(loaded,obs,context=CONTEXT,graph_policy=GRAPH_POLICY,source_state_sha256=state_id(model))
        bad=copy.deepcopy(obs);bad['A']=bad['A']*2
        with self.assertRaises(ValueError):verify_target_binding(result,bad,context=CONTEXT,graph_policy=GRAPH_POLICY,source_state_sha256=state_id(model))

    def test_labels_wrong_context_state_mode_and_nonfinite_fail_without_mutation(self):
        model,obs=self.fixture();before=state_id(model)
        for data in ({**obs,'labels':torch.zeros(1)}, {**obs,'net_input':obs['net_input']*float('nan')}):
            with self.assertRaises(ValueError):self.calibrate(model,data)
            self.assertEqual(state_id(model),before)
        with self.assertRaises(ValueError):calibrate_target(model,obs,mode=POLICY,context='DG',graph_policy=GRAPH_POLICY,source_state_sha256=before)
        with self.assertRaises(ValueError):calibrate_target(model,obs,mode=POLICY,context=CONTEXT,graph_policy=GRAPH_POLICY,source_state_sha256='a'*64)
        model.train()
        with self.assertRaises(ValueError):self.calibrate(model,obs)


if __name__=='__main__':unittest.main()
