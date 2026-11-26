import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

from policy import admit, plan, metrics


class PolicyTests(unittest.TestCase):
    def fixture(self):
        return [SimpleNamespace(key=f"{p}-{v}",parent=str(p),band=str(v),coverage=1/v,known=True,
                  x=SimpleNamespace(shape=(8,8,12//v+2))) for p in range(3) for v in (1,2,4)]

    def test_direction_role_and_scratch_guard(self):
        table=Path(__file__).resolve().parents[1]/"paper1_c/ordered-pair-feasibility.csv"
        ancestry={"source":"paviaU","target":"xuzhou","kind":"verified_source_pretrain_target_adaptation"}
        for role in ("confirmation","calibration","test","query"):
            with self.assertRaises(ValueError):admit(table,"paviaU","xuzhou",scope="cross_device_development",role=role,ancestry=ancestry)
        result=admit(table,"paviaU","xuzhou",scope="cross_device_development",role="devval",ancestry=ancestry)
        self.assertFalse(result["confirmation"])
        for source,target in (("whu-longkou","xuzhou"),("paviaU","paviaU"),("missing","xuzhou")):
            with self.assertRaises(ValueError):admit(table,source,target,scope="cross_device_development",role="devval",ancestry=ancestry)
        with self.assertRaises(ValueError):admit(table,"paviaU","xuzhou",scope="cross_device_development",role="devval",ancestry={"source":"xuzhou","target":"xuzhou","kind":"target_scratch"})

    def test_quota_permutation_random_and_label_absence(self):
        rows=self.fixture();m={r.key:(i+1)/10 for i,r in enumerate(rows)}
        before=copy.deepcopy(m)
        reference,c=plan(rows,m,"content_coverage")
        self.assertEqual(len(reference),5)
        actual,permuted=plan(rows,m,"permuted_coverage")
        self.assertEqual(len(actual),5)
        for parent in ("0","1","2"):
            keys=[r.key for r in rows if r.parent==parent]
            self.assertEqual(sorted(c[k] for k in keys),sorted(permuted[k] for k in keys))
        random,_=plan(rows,m,"matched_random",reference=reference)
        for band in ("1","2","4"):
            keys={r.key for r in rows if r.band==band}
            self.assertEqual(len(keys&set(reference)),len(keys&set(random)))
        self.assertEqual(m,before)
        self.assertEqual(plan(rows,m,"content",full_fraction=0)[0],[])
        self.assertEqual(len(plan(rows,m,"content",full_fraction=1)[0]),9)
        for bad in ({**m,rows[0].key:float("nan")},{rows[0].key:.5}):
            with self.assertRaises(ValueError):plan(rows,bad,"content")

    def test_recall_support_not_unlabelled_background(self):
        p=np.array([[.9,.1],[.8,.2],[.1,.9]])
        r=metrics([1,2,2],p,2)
        self.assertEqual(r["confusion"],[[1,0],[1,1]])
        self.assertEqual(r["recall"],[1.,.5])
        with self.assertRaises(ValueError):metrics([0,2,2],p,2)


if __name__=="__main__":unittest.main()
