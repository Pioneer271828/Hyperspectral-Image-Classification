import copy,json,os,tempfile,unittest,subprocess,sys
from pathlib import Path
import torch
from joint_baseline import NAMES,load_config
from loadData.dataset_registry import DATASETS,file_hash
from model.checkpoint_io import save_model_checkpoint,load_model_checkpoint
from source_training import source_settings
from target_initialization import initialize_target,domain_keys,state_id,METHODS

DEVICE=os.environ.get('G4E_DEVICE','cpu')
torch.set_num_threads(1)


class TargetInitializationTests(unittest.TestCase):
    def source(self,root,*,joint=False,bad_protocol=False,double=False):
        target='xuzhou';settings=source_settings(target);tasks=NAMES if joint else tuple(settings['source_tasks'])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(7)
            source=load_config('configs/experiments/base97_joint.json').build_model({n:DATASETS[n][2] for n in tasks})
        with torch.no_grad():
            for k,t in source.backbone.state_dict().items():
                if k in domain_keys(source.backbone) and t.is_floating_point():t.fill_(3)
            for family in (source.heads,source.hgnn_heads,source.cnn_heads):
                for p in family.parameters():p.fill_(9)
        if bad_protocol:settings['protocol']='base97_honghu_joint_training_v1'
        source.training_config={'joint_runtime':settings,'research_config':settings['config'],
            'checkpoint_optimizer_step':8,'provenance':{'policy':'selective_acc1p_source_v1','sources':settings['source_tasks'],
                'held_out':target,'target_payload_accessed':False,'source_identity':'a'*64}}
        if double:source.double()
        path=Path(root)/'source.pt';save_model_checkpoint(source,path)
        return path,file_hash(path),file_hash(str(path)+'.json'),source

    def create(self,method,source=None):
        kwargs={} if source is None else dict(source_path=source[0],source_weight_sha256=source[1],source_sidecar_sha256=source[2])
        return initialize_target('xuzhou',method,model_seed=64,head_seed=1064,device=DEVICE,**kwargs)

    def test_exact_shared_copy_fresh_heads_domains_freeze_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=self.source(tmp);before=state_id(source[3]);scratch,_=self.create(METHODS[0])
            for method in METHODS[1:]:
                target,record=self.create(method,source);self.assertEqual(tuple(target.heads),('xuzhou',))
                self.assertEqual(record['head_only_requires_target_bn_calibration'],method==METHODS[2])
                for family in ('heads','hgnn_heads','cnn_heads'):
                    for k,t in getattr(target,family).state_dict().items():torch.testing.assert_close(t,getattr(scratch,family).state_dict()[k],rtol=0,atol=0)
                for k,t in target.backbone.state_dict().items():
                    expected=scratch.backbone.state_dict()[k] if k in domain_keys(target.backbone) else source[3].backbone.state_dict()[k]
                    torch.testing.assert_close(t.cpu(),expected.cpu(),rtol=0,atol=0)
                self.assertTrue(all(p.requires_grad==(method!=METHODS[2]) for p in target.backbone.parameters()))
                self.assertTrue(all(p.requires_grad for p in target.heads.parameters()))
                self.assertEqual(target.backbone.training,method!=METHODS[2]);self.assertFalse(record['source_heads_copied'])
            self.assertEqual(state_id(source[3]),before);self.assertEqual(file_hash(source[0]),source[1])

    def test_pair_pins_joint_source_scope_and_dtype_rejected(self):
        for kwargs in ({'joint':True},{'bad_protocol':True},{'double':True}):
            with tempfile.TemporaryDirectory() as tmp:
                source=self.source(tmp,**kwargs)
                with self.assertRaises(ValueError):self.create(METHODS[1],source)
        with tempfile.TemporaryDirectory() as tmp:
            source=self.source(tmp)
            with self.assertRaises(ValueError):self.create(METHODS[0],source)
            with self.assertRaises(ValueError):self.create(METHODS[1],(source[0],'b'*64,source[2]))
            with open(str(source[0])+'.json','a') as f:f.write(' ')
            with self.assertRaises(ValueError):self.create(METHODS[1],source)
        with self.assertRaises(ValueError):self.create(METHODS[1])
        with self.assertRaises(ValueError):initialize_target('xuzhou',METHODS[0],model_seed=True,head_seed=64)

    def test_roundtrip_and_caller_rng_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=self.source(tmp);cpu=torch.get_rng_state().clone()
            cuda=[s.clone() for s in torch.cuda.get_rng_state_all()] if DEVICE=='cuda' else None
            target,record=self.create(METHODS[1],source)
            self.assertTrue(torch.equal(cpu,torch.get_rng_state()))
            if cuda is not None:self.assertTrue(all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all())))
            path=Path(tmp)/'target.pt';save_model_checkpoint(target,path);loaded=load_model_checkpoint(path,device=DEVICE)
            self.assertEqual(state_id(target),state_id(loaded));self.assertEqual(loaded.training_config['target_initialization'],record)

    def test_source_provenance_and_architecture_mismatch_and_inspection_entry(self):
        for change in ('provenance','architecture'):
            with tempfile.TemporaryDirectory() as tmp:
                source=self.source(tmp);sidecar=Path(str(source[0])+'.json');meta=json.loads(sidecar.read_text())
                if change=='provenance':meta['training_config']['provenance']['target_payload_accessed']=True
                else:meta['backbone_kwargs']['adapter_attention_mode']='index'
                sidecar.write_text(json.dumps(meta))
                with self.assertRaises(ValueError):self.create(METHODS[1],(source[0],source[1],file_hash(sidecar)))
        result=subprocess.run([sys.executable,'scripts/inspect_target_initialization.py','--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--source-sidecar-sha256',result.stdout)
        target,record=initialize_target('xuzhou',METHODS[0],model_seed=512,head_seed=1512,device=DEVICE)
        self.assertEqual(record['target_initial_config']['training']['seed'],512)
        self.assertEqual(record['initial_device'],str(next(target.parameters()).device))
        self.assertEqual(record['target_initial_config']['device'],DEVICE)
        self.assertFalse(record['target_training_budget_registered'])


if __name__=='__main__':unittest.main()
