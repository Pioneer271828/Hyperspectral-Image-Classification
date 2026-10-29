import copy, tempfile, unittest
from pathlib import Path
import numpy as np
from scipy.io import savemat
from loadData.fixed_splits import make_split, save_split, load_split, validate_split
from loadData.dataset_registry import inspect_dataset

class DataTests(unittest.TestCase):
    def setUp(self):self.labels=np.array([0]+[1]*20+[2]*20).reshape(1,41)
    def test_reproducible_budget_and_global_rng(self):
        np.random.seed(9); before=np.random.get_state()
        a=make_split(self.labels,7,3,2)
        self.assertEqual(a,make_split(self.labels,7,3,2))
        self.assertNotEqual(a['indices'],make_split(self.labels,8,3,2)['indices'])
        self.assertTrue(np.array_equal(before[1],np.random.get_state()[1]))
        self.assertEqual([len(a['indices'][k]) for k in ('train','val','test')],[6,4,30])
    def test_reject_overlap_unlabeled_missing_and_changed_labels(self):
        base=make_split(self.labels,7,3,2)
        for mode in ('overlap','zero','missing','changed'):
            a=copy.deepcopy(base); labels=self.labels.copy()
            if mode=='overlap':a['indices']['test'].append(a['indices']['train'][0])
            if mode=='zero':a['indices']['test'].append(0)
            if mode=='missing':a['indices']['test'].pop()
            if mode=='changed':labels[0,1]=2
            with self.assertRaises(ValueError):validate_split(a,labels)
    def test_no_overwrite_and_reload(self):
        a=make_split(self.labels,7,3,2)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'split.json';save_split(path,a)
            self.assertEqual(load_split(path,self.labels),a)
            with self.assertRaises(FileExistsError):save_split(path,a)
    def test_invalid_budget(self):
        for args in ((-1,3,2),(1,0,2),(1,15,10)):
            with self.assertRaises(ValueError):make_split(self.labels,*args)
    def test_registry_real_mat_fixture_and_invalid_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'PaviaU').mkdir()
            image=np.zeros((9,10,3));labels=np.repeat(np.arange(1,10),10).reshape(9,10)
            savemat(root/'PaviaU/PaviaU.mat',{'paviaU':image})
            savemat(root/'PaviaU/PaviaU_gt.mat',{'paviaU_gt':labels})
            record,_=inspect_dataset(root,'paviaU')
            self.assertEqual(record['shape'],[9,10,3])
            self.assertEqual(record['class_count'],9)
            labels[0,0]=-1;savemat(root/'PaviaU/PaviaU_gt.mat',{'paviaU_gt':labels})
            with self.assertRaises(ValueError):inspect_dataset(root,'paviaU')

if __name__=='__main__':unittest.main()
