"""Pinned actual native/DSNet resume integration, synthetic input only."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

from transactions import execute,paired_plan
from tasks import NativeTask,DSNetTask,OPTIMIZER,prior_module,synthetic_data
from test_transactions import equal,state
from audit import audit


@unittest.skipUnless(os.environ.get('PAPER_C_NATIVE') and os.environ.get('PAPER_C_AUTHOR'),'explicit pinned external dependencies required')
class OriginalModelRecovery(unittest.TestCase):
    def test_all_three_shapes_both_original_models(self):
        sys.path.insert(0,os.environ['PAPER_C_NATIVE'])
        root=Path(__file__).resolve().parents[2];adapter=prior_module(root,'adapter');author=adapter.load_author_model(os.environ['PAPER_C_AUTHOR'])
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory() as tmp:
            for scene,bands,classes in (('paviaU',103,9),('xuzhou',436,9),('whu-honghu',270,22)):
                data=synthetic_data(adapter,bands,classes)
                for method in ('native','dsnet'):
                    with self.subTest(scene=scene,method=method):
                        c={'roster':[0],'sweeps':3,'seed':64,'plan':paired_plan([0],3,64),'updates':3,
                           'selection_steps':[0,1,2,3],'device':'cpu','query_reads':0,'confirmation':False,
                           'optimizer':OPTIMIZER,'protected':{},'scene':scene,'method':method}
                        factory=(lambda:NativeTask(data,scene,classes)) if method=='native' else (lambda:DSNetTask(data,scene,classes,bands,adapter,author))
                        full=Path(tmp)/(scene+method+'-full');split=Path(tmp)/(scene+method+'-split')
                        execute(full,c,factory);execute(split,c,factory,stop_after=1)
                        r=execute(split,c,factory,resume=True);self.assertEqual(r['new_updates'],2)
                        for step in range(4):self.assertTrue(equal(state(full,step),state(split,step)),f'{scene}/{method}/{step}')
                        audited=audit(split);self.assertEqual(audited['committed_optimizer_updates'],3)
                        execute(split,c,lambda:self.fail('completed task constructed'),resume=True)


if __name__=='__main__':unittest.main()
