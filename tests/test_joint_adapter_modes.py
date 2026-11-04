import copy,os,tempfile,unittest
from pathlib import Path
import torch
from joint_adapter_modes import MODES,adapter_settings,AdapterModeRuntime
from joint_training import run_training,train_step,_tree_fingerprint
from scripts.train_joint_m4 import m4_settings,graph_binding
from model.research_config import ResearchConfig
from test_joint_training import settings,SOURCE,PROVENANCE,hashes,checkpoints
from test_calibrated_selection import items
from test_joint_supervision import optimizer

DEVICE=os.environ.get('M2J1_DEVICE','cpu')
torch.set_num_threads(1)


def build(mode,data):
    opts=adapter_settings(settings(),mode);runtime=AdapterModeRuntime(mode)
    torch.manual_seed(64)
    m=runtime.build_model(ResearchConfig.from_dict(opts['config']),{n:i['class_num'] for n,i in data.items()}).to(DEVICE)
    return m,opts,runtime


class AdapterJointTests(unittest.TestCase):
    def data(self):return {n:{k:v.to(DEVICE) if torch.is_tensor(v) else v for k,v in i.items()} for n,i in items().items()}

    def test_mode_and_configuration_guards(self):
        for mode in MODES:
            cfg=adapter_settings(settings(),mode);AdapterModeRuntime(mode).validate(cfg)
            bad=copy.deepcopy(cfg);bad['config']['model']['region_residual_mode']='x_residual'
            with self.assertRaises(ValueError):AdapterModeRuntime(mode).validate(bad)
            bad=copy.deepcopy(cfg);bad['adapter_comparison']['attention_parameters_active']=mode=='uniform'
            with self.assertRaises(ValueError):AdapterModeRuntime(mode).validate(bad)
        with self.assertRaises(ValueError):adapter_settings(settings(),'invalid')
        cfg=adapter_settings(settings(),'wavelength');cfg.pop('adapter_comparison')
        self.assertEqual(cfg,m4_settings(settings(),'none'))

    def test_all_modes_same_initial_state_and_wavelength_legacy_step(self):
        data=self.data();models=[build(mode,data) for mode in MODES]
        expected=models[0][0].state_dict()
        for m,_,_ in models[1:]:
            for k,v in m.state_dict().items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
        m,opts,runtime=models[0];old=copy.deepcopy(m);a,b=optimizer(m),optimizer(old);name=next(iter(data));cfg=ResearchConfig.from_dict(opts['config'])
        torch.manual_seed(12);ra=runtime.train_step(m,a,name,data[name],cfg)
        torch.manual_seed(12);rb=train_step(old,b,name,data[name],cfg)
        self.assertEqual(ra,rb);self.assertEqual(_tree_fingerprint(m.state_dict()),_tree_fingerprint(old.state_dict()))
        self.assertEqual(_tree_fingerprint(a.state_dict()),_tree_fingerprint(b.state_dict()))

    def test_actual_attention_modes_and_uniform_inactive_parameters(self):
        data=self.data();name=next(iter(data));outputs=[]
        for mode in MODES:
            m,opts,runtime=build(mode,data);adapter=m.backbone.adapter
            x=data[name]['net_input'][...,:-2];w=torch.linspace(440,940,x.shape[-1],device=DEVICE)
            y=adapter(x,w,None);outputs.append(y.detach());before={n:p.clone() for n,p in adapter.named_parameters()}
            y.square().sum().backward();opt=torch.optim.Adam(adapter.parameters(),lr=.001,weight_decay=.001);opt.step()
            self.assertTrue(any(not torch.equal(p,before[n]) for n,p in adapter.named_parameters() if n.startswith('value_proj.')))
            if mode=='uniform':
                for n,p in adapter.named_parameters():
                    if not n.startswith('value_proj.'):
                        self.assertIsNone(p.grad);self.assertNotIn(p,opt.state);torch.testing.assert_close(p,before[n],rtol=0,atol=0)
        self.assertFalse(torch.equal(outputs[0],outputs[1]));self.assertFalse(torch.equal(outputs[1],outputs[2]))

    def test_all_mode_serialization_and_exact_uniform_continuation(self):
        data=self.data();prov={**PROVENANCE,'m4_graph_binding':graph_binding(data)}
        with tempfile.TemporaryDirectory() as temp:
            for mode in MODES:
                _,opts,runtime=build(mode,data);folder=Path(temp)/mode
                result=run_training(opts,data,folder,prov,SOURCE,device=DEVICE,runtime_adapter=runtime)
                step=result['result']['selected_step'];raw=folder/f'raw-validation-{step:06d}.pt'
                runtime.load_selected(folder/'primary.pt',data,raw,device=DEVICE)
                other=MODES[(MODES.index(mode)+1)%3]
                with self.assertRaises(ValueError):AdapterModeRuntime(other).load_selected(folder/'primary.pt',data,raw,device=DEVICE)
                before=hashes(folder)
                self.assertEqual(run_training(opts,data,folder,prov,SOURCE,device=DEVICE,runtime_adapter=runtime,resume=True)['updates_this_call'],0)
                self.assertEqual(before,hashes(folder))
            _,opts,runtime=build('uniform',data);split=Path(temp)/'split'
            run_training(opts,data,split,prov,SOURCE,device=DEVICE,runtime_adapter=runtime,stop_after=5)
            run_training(opts,data,split,prov,SOURCE,device=DEVICE,runtime_adapter=runtime,resume=True)
            for key in ('model','optimizer','scheduler'):
                self.assertEqual(_tree_fingerprint(checkpoints(Path(temp)/'uniform')[key]),_tree_fingerprint(checkpoints(split)[key]))


if __name__=='__main__':unittest.main()
