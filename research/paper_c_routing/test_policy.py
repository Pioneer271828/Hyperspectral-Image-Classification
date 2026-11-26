import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
import json
import hashlib
import numpy as np
import csv

from policy import admit, plan, metrics, verify_completed
from catalog import build


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

    def test_all_thirty_directions_are_exhaustive_and_development_only(self):
        table=Path(__file__).resolve().parents[1]/'paper1_c/ordered-pair-feasibility.csv'
        catalog=build(table);admitted=0
        with table.open(encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                args=(table,row['source'],row['target'])
                kw={'scope':'cross_device_development','role':'devval',
                    'ancestry':{'source':row['source'],'target':row['target'],'kind':'verified_source_pretrain_target_adaptation'}}
                if row['status']=='READY_DEVELOPMENT_ONLY':
                    self.assertFalse(admit(*args,**kw)['confirmation']);admitted+=1
                else:
                    with self.assertRaises(ValueError):admit(*args,**kw)
                with self.assertRaises(ValueError):admit(*args,**{**kw,'role':'confirmation'})
        self.assertEqual(admitted,6)
        self.assertEqual(sum(x['development_execution'] for x in catalog['directions']),1)
        self.assertTrue(all(not x['independent_confirmation'] and not x['independent_calibration'] for x in catalog['directions']))

    def test_nine_original_models_have_explicit_immutable_file_pins(self):
        manifest=json.loads((Path(__file__).parent/'g3-profiles.json').read_text())
        self.assertEqual(set(manifest['profiles']),{'2036','2046','2056'})
        models=[]
        for seed,files in manifest['profiles'].items():
            models.append(files['model.pt'])
            self.assertEqual(set(manifest['variants'][seed]),{'target_scratch_full','source_pretrained_head_only'})
            for method,pins in manifest['variants'][seed].items():
                self.assertEqual(set(pins),{method+'/model.pt',method+'/model.pt.json'})
                models.append(pins[method+'/model.pt'])
            for h in [*files.values(),*(v for pins in manifest['variants'][seed].values() for v in pins.values())]:
                self.assertEqual(len(h),64);int(h,16)
        self.assertEqual(len(set(models)),9)

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

    def test_completed_reentry_protects_outputs_and_original_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original=root/"input.bin";original.write_bytes(b"model")
            setup=root/"setup.json"
            setup.write_text(json.dumps({"protected":{str(original):hashlib.sha256(b"model").hexdigest()}}))
            (root/"summary.json").write_text(json.dumps({"files":{"setup.json":hashlib.sha256(setup.read_bytes()).hexdigest()}}))
            verify_completed(root)
            original.write_bytes(b"changed")
            with self.assertRaises(ValueError):verify_completed(root)
            original.write_bytes(b"model");setup.write_text("{}")
            with self.assertRaises(ValueError):verify_completed(root)


if __name__=="__main__":unittest.main()
