import copy
import unittest
from scripts.run_sampling_rebuild import verify_source_record


class ParallelSourceTests(unittest.TestCase):
    def test_only_completed_matching_source_selection_is_reused(self):
        campaign={'status':'passed','plan':{'source_only':True},'code':{'commit':'expected'},'seeds':[2056]}
        task={'config':{'model_seed':2056}}
        result={'status':'completed','optimizer_steps':10000,'source_test_evaluated':False,
                'selected_step':200,'history':[{'step':100,'val':{'nll':2}}, {'step':200,'val':{'nll':1}}]}
        verify_source_record(campaign,task,result,'expected')
        for name in ('running','wrong_sha','wrong_seed','wrong_budget','wrong_step','test_access','not_source_only'):
            c=copy.deepcopy(campaign);t=copy.deepcopy(task);r=copy.deepcopy(result)
            if name=='running':c['status']='running'
            elif name=='wrong_sha':c['code']['commit']='other'
            elif name=='wrong_seed':t['config']['model_seed']=2046
            elif name=='wrong_budget':r['optimizer_steps']=9999
            elif name=='wrong_step':r['selected_step']=100
            elif name=='test_access':r['source_test_evaluated']=True
            else:c['plan']['source_only']=False
            with self.subTest(case=name),self.assertRaises(AssertionError):verify_source_record(c,t,r,'expected')
