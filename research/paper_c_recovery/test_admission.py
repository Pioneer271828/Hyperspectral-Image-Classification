"""Reject tempting source/head substitutions before real direction execution."""
import copy
import unittest
from inventory_models import qualify


class DirectionAdmission(unittest.TestCase):
    def setUp(self):
        self.row={'source_datasets':['paviaU'],'target_dataset':'xuzhou','class_counts':{'xuzhou':9},
            'graph_mode':'learned','norm_type':'group_norm','weight_present':True,'method':'source_pretrained_full_finetune'}

    def test_matching_metadata_remains_candidate_not_automatic_permission(self):
        self.assertEqual(qualify(self.row,'paviaU','xuzhou'),[])
        # qualify is only a candidate filter; inventory always leaves execution unadmitted.

    def test_two_source_and_equal_cardinality_head_rejected(self):
        row=copy.deepcopy(self.row);row['source_datasets']=['paviaU','whu-honghu']
        self.assertIn('not_verified_single_source_ancestry',qualify(row,'paviaU','xuzhou'))
        row=copy.deepcopy(self.row);row['class_counts']={'paviaU':9}
        self.assertIn('no_matching_target_semantic_head',qualify(row,'paviaU','xuzhou'))

    def test_scratch_static_domain_bn_or_missing_weight_rejected(self):
        for field,value,reason in (
            ('method','target_scratch_full','not_a_verified_original_target_method'),
            ('graph_mode','static','not_local_learned_graph'),
            ('norm_type','domain_bn','not_verified_group_norm_lazy_profile'),
            ('weight_present',False,'weight_bytes_missing')):
            with self.subTest(field=field):
                row=copy.deepcopy(self.row);row[field]=value
                self.assertIn(reason,qualify(row,'paviaU','xuzhou'))


if __name__=='__main__':unittest.main()
