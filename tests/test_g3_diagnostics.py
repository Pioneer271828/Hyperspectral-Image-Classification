import unittest
from scripts.diagnose_g3_training import exposure


class ExposureTests(unittest.TestCase):
    def test_tile_sampling_preserves_trace_and_distinguishes_class_presence(self):
        plan={'shape':[2,4],'tiles':[
            {'id':0,'partition':'train','box':[0,2,0,2]},
            {'id':1,'partition':'train','box':[0,2,2,4]},
            {'id':2,'partition':'test','box':[0,2,0,4]}]}
        support={'indices':[0,1,2,3],'labels':[1,2,1,1],'classes':[1,2]}
        result=exposure(plan,support,42,100)
        a,b=result['classes']
        self.assertEqual(a['updates_present'],100)
        self.assertEqual(b['updates_present'],result['sampled_tiles'].count(0))
        self.assertEqual(a['label_exposures'],100+result['sampled_tiles'].count(1))
        self.assertNotIn(2,result['sampled_tiles'])
        self.assertEqual(result,exposure(plan,support,42,100))

    def test_no_released_training_labels_rejected(self):
        with self.assertRaises(ValueError):
            exposure({'shape':[2,4],'tiles':[]},{'indices':[],'labels':[],'classes':[1]},1,10)
