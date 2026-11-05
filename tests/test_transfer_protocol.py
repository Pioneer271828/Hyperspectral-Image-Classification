"""G3 label budget, spatial isolation and held-out-label noninterference."""
import copy
import json
from pathlib import Path
import unittest
import numpy as np
from loadData.spatial_tiles import make_tile_plan,tile_slice
from loadData.transfer_protocol import make_support,validate_support,target_labels_for_tile,freeze_task,validate_config


def fixture():
    plan=make_tile_plan((96,96),12,1)
    labels=np.tile(np.arange(1,4),96*32).reshape(96,96)
    return plan,labels


class TransferProtocolTests(unittest.TestCase):
    def test_nested_deterministic_support_and_exact_budget(self):
        plan,labels=fixture();before=np.random.get_state()
        records=[make_support(labels,plan,[1,2,3],k,2027) for k in (1,5,10)]
        self.assertTrue(set(records[0]['indices'])<=set(records[1]['indices'])<=set(records[2]['indices']))
        self.assertEqual([len(r['indices']) for r in records],[3,15,30])
        for record in records:
            validate_support(record,labels,plan)
            self.assertEqual(record,make_support(labels,plan,[1,2,3],record['per_class'],2027))
        np.testing.assert_array_equal(before[1],np.random.get_state()[1])
        self.assertEqual(before[2:],np.random.get_state()[2:])
        released=sum(int((target_labels_for_tile(labels,plan,records[-1],t['id'],phase='adapt')>0).sum()) for t in plan['tiles'] if t['partition']=='train')
        self.assertEqual(released,30)

    def test_heldout_and_unselected_labels_do_not_enter_adaptation(self):
        plan,labels=fixture();support=make_support(labels,plan,[1,2,3],10)
        poisoned=labels.astype(float)
        for tile in plan['tiles']:
            if tile['partition']!='train':poisoned[tile_slice(tile)]=float('nan')
        self.assertEqual(support,make_support(poisoned,plan,[1,2,3],10))
        only_selected=np.full(labels.shape,-999.)
        only_selected.reshape(-1)[support['indices']]=support['labels']
        for tile in plan['tiles']:
            if tile['partition']=='train':
                a=target_labels_for_tile(labels,plan,support,tile['id'],phase='adapt')
                b=target_labels_for_tile(only_selected,plan,support,tile['id'],phase='adapt')
                np.testing.assert_array_equal(a,b)
            else:
                with self.assertRaises(ValueError):target_labels_for_tile(labels,plan,support,tile['id'],phase='adapt')
                with self.assertRaises(ValueError):target_labels_for_tile(labels,plan,support,tile['id'],phase='validation')
                with self.assertRaises(ValueError):target_labels_for_tile(labels,plan,support,tile['id'],phase='final_test')
        test=next(t for t in plan['tiles'] if t['partition']=='test')
        np.testing.assert_array_equal(target_labels_for_tile(labels,plan,support,test['id'],phase='final_test',final_checkpoint_sha256='f'*64),labels[tile_slice(test)])

    def test_tamper_insufficiency_and_changed_selected_label_rejected(self):
        plan,labels=fixture();support=make_support(labels,plan,[1,2,3],10)
        corrupt=copy.deepcopy(support);corrupt['indices'][0]+=1
        with self.assertRaises(ValueError):validate_support(corrupt,labels,plan)
        tile=next(t for t in plan['tiles'] if t['partition']=='train')
        with self.assertRaises(ValueError):target_labels_for_tile(labels,plan,corrupt,tile['id'],phase='adapt')
        with self.assertRaisesRegex(ValueError,'Insufficient'):make_support(labels,plan,[1,2,3,4],10)
        changed=labels.copy();changed.reshape(-1)[support['indices']]=0
        with self.assertRaises(ValueError):validate_support(support,changed,plan)
        for bad in (0,True,-1):
            with self.assertRaises(ValueError):make_support(labels,plan,[1,2,3],bad)

    def test_frozen_task_identity_and_access_claims(self):
        config=json.loads(Path('configs/protocols/paviau_to_xuzhou_g3.json').read_text(encoding='utf-8-sig'))
        validate_config(config)
        source_plan=make_tile_plan((96,96),config['tile_size'],config['gap'],config['split_seed'])
        labels=np.tile(np.arange(1,4),96*32).reshape(96,96)
        support=make_support(labels,source_plan,[1,2,3],10,config['support_seed'])
        source={'name':'paviaU','shape':[96,96,103],'class_count':3}
        target={'name':'xuzhou','shape':[96,96,436],'class_count':3}
        evidence={'target_sensor_readme_sha256':'a'*64,'target_sensor_readme_mentions_hyspex':True}
        task=freeze_task(config,source,target,source_plan,source_plan,support,evidence)
        self.assertEqual(task['target_total_training_selection_labels'],30)
        self.assertFalse(task['claims']['zero_shot']);self.assertFalse(task['claims']['shared_semantic_classes'])
        self.assertEqual(task,freeze_task(config,source,target,source_plan,source_plan,support,evidence))
        for bad in ({'target_validation_labels':1},{'target_selection':'best_target_oa'},{'task_type':'G2_zero_shot'},{'extra':1}):
            with self.assertRaises(ValueError):validate_config(config|bad)
        with self.assertRaises(ValueError):freeze_task(config,source,target,source_plan,source_plan,support,{})

if __name__=='__main__':unittest.main()
