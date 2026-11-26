"""Synthetic actual-model cross-band backbone transfer and unchanged fresh heads."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from tasks import NativeTask,OPTIMIZER,prior_module,synthetic_data
from transactions import execute,paired_plan,rng_state,seed_all
from transfer_run import TransferTask
from test_transactions import equal,state


@unittest.skipUnless(os.environ.get('PAPER_C_NATIVE') and os.environ.get('PAPER_C_AUTHOR'),'explicit pinned dependencies required')
class OriginalNativeTransfer(unittest.TestCase):
    def test_cross_band_cross_class_init_resume_and_light_endpoint(self):
        sys.path.insert(0,os.environ['PAPER_C_NATIVE']);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        root=Path(__file__).resolve().parents[2];adapter=prior_module(root,'adapter')
        source=synthetic_data(adapter,103,9);target=synthetic_data(adapter,270,22)
        def contract(scene):
            return {'roster':[0],'sweeps':3,'seed':64,'plan':paired_plan([0],3,64),'updates':3,
                'selection_steps':[0,1,2,3],'device':'cpu','query_reads':0,'confirmation':False,
                'optimizer':OPTIMIZER,'protected':{},'scene':scene,'method':'native'}
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);src=tmp/'src';base=tmp/'base'
            execute(src,contract('paviaU'),lambda:NativeTask(source,'paviaU',9))
            execute(base,contract('whu-honghu'),lambda:NativeTask(target,'whu-honghu',22),stop_after=1)
            source_file=src/'stages'/'000003'/'state.pt';base0=base/'stages'/'000000'/'state.pt'
            seed_all(999);trans=TransferTask(target,'whu-honghu',22,source_file,base0)
            srcstate=state(src,3);basestate=state(base,0);actual=trans.model.state_dict()
            for key,value in actual.items():
                expected=srcstate['model'][key] if key.startswith('backbone.') else basestate['model'][key]
                self.assertTrue(torch.equal(value,expected),key)
            self.assertFalse(trans.optimizer.state);self.assertTrue(equal(rng_state(),basestate['rng']))
            full=tmp/'full';split=tmp/'split';c=contract('whu-honghu')
            factory=lambda:TransferTask(target,'whu-honghu',22,source_file,base0)
            execute(full,c,factory);execute(split,c,factory,stop_after=1);execute(split,c,factory,resume=True)
            for step in range(4):self.assertTrue(equal(state(full,step),state(split,step)))
            from inference.light_cnn import LightCNN
            trans.model.load_state_dict(state(full,3)['model']);trans.model.eval()
            trans.model.backbone.adapter.cache_attention=False;trans.model.backbone.adapter.clear_attention_cache()
            row=target['devval'][0]
            with torch.no_grad():expected=trans._forward(row)[2]
            light=LightCNN(trans.model)
            x=torch.tensor(__import__('numpy').concatenate((row['observation'].spectral,row['observation'].coordinates),-1))
            wave=torch.tensor(row['wavelengths'])
            with patch.object(trans.model.backbone.learnable_graph,'forward',side_effect=AssertionError('cheap path built graph')):
                with torch.no_grad():actual=light('whu-honghu',x,wavelengths=wave)
            self.assertTrue(torch.equal(expected,actual))


if __name__=='__main__':unittest.main()
