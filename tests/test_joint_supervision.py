import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
import torch

from joint_supervision import SupervisionRuntime, supervision_settings
from joint_training import run_training, train_step, _tree_fingerprint
from model.research_config import ResearchConfig
from scripts.train_joint_m4 import m4_settings, graph_binding
from branch_graph_training import AblationRuntime, ablation_settings
from test_joint_training import SOURCE, PROVENANCE, settings, hashes, checkpoints
from test_calibrated_selection import items

torch.set_num_threads(1)
DEVICE=os.environ.get('M4S1_DEVICE','cpu')


def prepared():
    return {n:{k:v.to(DEVICE) if torch.is_tensor(v) else v for k,v in i.items()} for n,i in items().items()}


def model(runtime, opts, data):
    torch.manual_seed(64)
    return runtime.build_model(ResearchConfig.from_dict(opts['config']),{n:i['class_num'] for n,i in data.items()}).to(DEVICE)


def optimizer(m):
    return torch.optim.Adam([{'params':m.backbone.parameters(),'lr':5e-5},
        {'params':list(m.heads.parameters())+list(m.hgnn_heads.parameters())+list(m.cnn_heads.parameters()),'lr':5e-4}],weight_decay=.001)


class JointSupervisionTests(unittest.TestCase):
    def test_settings_legacy_unchanged_and_policy_rejections(self):
        base=settings();before=copy.deepcopy(base)
        for mode in ('none','x_residual'):
            candidate=supervision_settings(base,mode,'legacy_three_heads')
            candidate.pop('supervision_policy')
            self.assertEqual(candidate,m4_settings(base,mode))
        self.assertEqual(base,before)
        for mode,supervision in [('bad','primary_only'),('none','bad')]:
            with self.assertRaises(ValueError):supervision_settings(base,mode,supervision)
        opts=supervision_settings(base,'x_residual','primary_only')
        runtime=SupervisionRuntime('x_residual','primary_only')
        runtime.validate(opts)
        for field in ('supervision_policy','validation_policy'):
            with self.assertRaises(ValueError):runtime.validate({**opts,field:{}})
        bad=copy.deepcopy(opts);bad['config']['loss']['cnn_loss_weight']=1
        with self.assertRaises(ValueError):runtime.validate(bad)

    def test_legacy_step_exactly_matches_original_m4(self):
        data=prepared();n=next(iter(data))
        for mode in ('none','x_residual'):
            runtime=SupervisionRuntime(mode,'legacy_three_heads')
            opts=supervision_settings(settings(),mode,'legacy_three_heads');cfg=ResearchConfig.from_dict(opts['config'])
            a=model(runtime,opts,data);b=copy.deepcopy(a);oa,ob=optimizer(a),optimizer(b)
            torch.manual_seed(123);x=runtime.train_step(a,oa,n,data[n],cfg)
            torch.manual_seed(123);y=train_step(b,ob,n,data[n],cfg)
            self.assertEqual(x['loss'],y['loss'])
            for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
            self.assertEqual(_tree_fingerprint(oa.state_dict()),_tree_fingerprint(ob.state_dict()))

    def test_primary_updates_backbone_residual_but_not_auxiliary(self):
        data=prepared();n=next(iter(data));runtime=SupervisionRuntime('x_residual','primary_only')
        opts=supervision_settings(settings(),'x_residual','primary_only');cfg=ResearchConfig.from_dict(opts['config'])
        m=model(runtime,opts,data);before={k:v.clone() for k,v in m.named_parameters()};opt=optimizer(m)
        row=runtime.train_step(m,opt,n,data[n],cfg)
        self.assertGreater(row['residual_gradient_norm'],0);self.assertGreater(row['residual_changed_elements'],0)
        for k,p in m.named_parameters():
            if k.startswith(('cnn_heads.','hgnn_heads.')):
                self.assertIsNone(p.grad);self.assertNotIn(p,opt.state)
                torch.testing.assert_close(p,before[k],rtol=0,atol=0)
        for prefix in ('backbone.CNN_Branch.','backbone.HGNN_Branch.','heads.'+n+'.'):
            self.assertTrue(any(not torch.equal(p,before[k]) for k,p in m.named_parameters() if k.startswith(prefix)))
        metrics,probs=runtime.validation_snapshot(m,data)
        for name in data:
            for phase in ('train','val'):
                self.assertEqual(set(probs[name][phase]),{'fused'})
                self.assertEqual(set(metrics['metrics'][name][phase]),{'fused'})

    def test_primary_no_residual_matches_independent_head_removal_baseline(self):
        data=prepared();n=next(iter(data))
        runtime=SupervisionRuntime('none','primary_only');opts=supervision_settings(settings(),'none','primary_only')
        a=model(runtime,opts,data)
        old=AblationRuntime('dual_fused_hypergraph');old_opts=ablation_settings(settings(),'dual_fused_hypergraph')
        b=model(old,old_opts,data);oa,ob=optimizer(a),optimizer(b)
        torch.manual_seed(33);ra=runtime.train_step(a,oa,n,data[n],ResearchConfig.from_dict(opts['config']))
        torch.manual_seed(33);rb=old.train_step(b,ob,n,data[n],ResearchConfig.from_dict(old_opts['config']))
        self.assertEqual(ra['loss']['total'],rb['loss']['total'])
        for k,p in b.named_parameters():torch.testing.assert_close(p,dict(a.named_parameters())[k],rtol=0,atol=0)

    def test_exact_continuation_immutable_completion_and_policy_binding(self):
        data=prepared();runtime=SupervisionRuntime('x_residual','primary_only')
        opts=supervision_settings(settings(),'x_residual','primary_only')
        prov={**PROVENANCE,'m4_graph_binding':graph_binding(data)}
        with tempfile.TemporaryDirectory() as temp:
            full,split=Path(temp)/'full',Path(temp)/'split'
            run_training(opts,data,full,prov,SOURCE,device=DEVICE,runtime_adapter=runtime)
            run_training(opts,data,split,prov,SOURCE,device=DEVICE,runtime_adapter=runtime,stop_after=5)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                other=supervision_settings(settings(),'x_residual','legacy_three_heads')
                run_training(other,data,split,prov,SOURCE,device=DEVICE,resume=True,runtime_adapter=SupervisionRuntime('x_residual','legacy_three_heads'))
            answer=run_training(opts,data,split,prov,SOURCE,device=DEVICE,resume=True,runtime_adapter=runtime)
            self.assertEqual(answer['updates_this_call'],7)
            for key in ('model','optimizer','scheduler'):
                self.assertEqual(_tree_fingerprint(checkpoints(full)[key]),_tree_fingerprint(checkpoints(split)[key]))
            saved=hashes(split)
            self.assertEqual(run_training(opts,data,split,prov,SOURCE,device=DEVICE,resume=True,runtime_adapter=runtime)['updates_this_call'],0)
            self.assertEqual(saved,hashes(split))
            step=answer['result']['selected_step'];path=split/'primary.pt';raw=split/f'raw-validation-{step:06d}.pt'
            runtime.load_selected(path,data,raw,device=DEVICE)
            with self.assertRaisesRegex(ValueError,'policy mismatch'):
                SupervisionRuntime('x_residual','legacy_three_heads').load_selected(path,data,raw,device=DEVICE)
            bad=copy.deepcopy(data);bad['paviaU']['A']=bad['paviaU']['A']*2
            with self.assertRaisesRegex(ValueError,'graph binding mismatch'):runtime.load_selected(path,bad,raw,device=DEVICE)
            sidecar=Path(str(path)+'.json');metadata=json.loads(sidecar.read_text())
            metadata['training_config']['research_config']['loss']['cnn_loss_weight']=1
            sidecar.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'configuration mismatch'):runtime.load_selected(path,data,raw,device=DEVICE)


if __name__=='__main__':unittest.main()
