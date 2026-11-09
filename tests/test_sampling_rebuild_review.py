import copy
import unittest
from scripts.run_sampling_rebuild import review_training


class SamplingRebuildReviewTests(unittest.TestCase):
    def fixture(self):
        return {'status':'completed','steps':5000,
                'trace':[{'step':i,'tile_id':0,'loss':{'total':1/i},'anchor_class':1} for i in range(1,5001)],
                'validation':[{'step':i,'nll':1 if i in (100,200) else 2,'oa':.5} for i in range(100,5001,100)],
                'selected_step':100,'sampler':{'classes':[1,2],'tile_ids':[0],'tile_class_counts':[[2,1]]},
                'initial_backbone_hash':'a','final_backbone_hash':'b',
                'initial_head_hash':'c','final_head_hash':'d'}

    def test_earliest_selection_participation_and_failed_trace(self):
        result=self.fixture();review=review_training(result)
        self.assertEqual(review['selected']['step'],100)
        self.assertEqual(review['class_participation_updates'],[5000,5000])
        self.assertEqual(review['class_label_exposures'],[10000,5000])
        self.assertEqual(review['anchor_counts'],[5000,0])
        self.assertTrue(review['backbone_changed'] and review['heads_changed'])
        self.assertLess(review['last_100_train_loss'],review['first_100_train_loss'])
        for change in ('bad_selection','missing_update','nonfinite_loss'):
            bad=copy.deepcopy(result)
            if change=='bad_selection':bad['selected_step']=200
            elif change=='missing_update':bad['trace'].pop()
            else:bad['trace'][0]['loss']['total']=float('nan')
            with self.subTest(change=change),self.assertRaises(AssertionError):review_training(bad)
