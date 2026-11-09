"""Hand geometry and scoring tests for observed objects, not true instances."""
import copy
import json
import unittest
import numpy as np
from evaluation.annotated_components import component_geometry,ComponentPlan,score_components


def freeze(y,*,score=None,geometry=None,small_area=4,thin_depth=1):
    geometry=np.ones(y.shape,dtype=bool) if geometry is None else geometry
    score=geometry.copy() if score is None else score
    kw=dict(geometry_mask=geometry,score_mask=score,small_area=small_area,thin_depth=thin_depth)
    return ComponentPlan.freeze(y,[7,2],**kw),kw


class ComponentTests(unittest.TestCase):
    def test_square_line_diagonal_and_class_connectivity(self):
        y=np.full((9,12),2);y[2:5,2:5]=7;y[2:7,8]=7;y[7,9]=7
        p,kw=freeze(y);value,c,d=component_geometry(y,[7,2],**kw)
        sevens=[r for r in value['components'] if r['class_id']==7]
        self.assertEqual([r['observed_area'] for r in sevens],[9,5,1]);self.assertEqual([r['max_inward_depth'] for r in sevens],[2,1,1])
        self.assertEqual([r['thin'] for r in sevens],[False,True,True]);self.assertEqual([r['small'] for r in sevens],[False,False,True])
        self.assertEqual(int(d[3,3]),2);self.assertNotEqual(c[6,8],c[7,9]);self.assertFalse(any(r['incomplete_observation'] for r in sevens))
        self.assertEqual(p.plan_hash,value['plan_hash']);self.assertEqual(int((d>0).sum()),y.size)

    def test_holes_exclusions_and_crop_are_not_physical_background(self):
        y=np.full((7,7),2.);y[1:6,1:6]=7;y[3,3]=0;geo=np.ones(y.shape,bool);geo[2,2]=False;y[2,2]=np.nan
        p,kw=freeze(y,geometry=geo);rows=p.to_dict()['components'];r=next(r for r in rows if r['class_id']==7)
        self.assertEqual(r['observed_area'],23);self.assertEqual(r['touches'],dict(array_border=False,excluded=True,unlabeled=True))
        self.assertTrue(r['incomplete_observation']);self.assertTrue(rows[0]['touches']['array_border'])
        a=score_components(y,y,[7,2],plan=p,expected_plan_hash=p.plan_hash,**{k:kw[k] for k in ('geometry_mask','score_mask')})
        self.assertEqual(a['groups']['all']['count'],47);self.assertIsNone(a['groups']['observed_closed']['oa'])
        self.assertEqual(p.to_dict()['score_unlabeled_count'],1)

    def test_geometry_outside_scoring_and_per_class_metrics(self):
        y=np.full((7,7),2);y[2:5,2:5]=7;score=np.zeros_like(y,dtype=bool);score[2,2:4]=True;score[0,0]=True
        p,kw=freeze(y,score=score,small_area=9);pred=y.copy();pred[2,2]=2
        r=score_components(y,pred,[7,2],geometry_mask=kw['geometry_mask'],score_mask=score,plan=p,expected_plan_hash=p.plan_hash)
        self.assertEqual(r['groups']['all']['confusion_matrix'],[[1,1],[0,1]])
        self.assertEqual(r['groups']['small']['confusion_matrix'],[[1,1],[0,0]])
        self.assertEqual(r['groups']['small']['missing_class_ids'],[2]);self.assertIsNone(r['groups']['small']['aa_full'])
        self.assertEqual(r['groups']['small']['aa'],.5)
        seven=next(c for c in p.to_dict()['components'] if c['class_id']==7);self.assertEqual((seven['observed_area'],seven['scored_count']),(9,2))
        self.assertEqual(r['components'][seven['component_id']-1]['recall'],.5)
        self.assertEqual(p.to_dict()['geometry_labeled_outside_scoring'],46)

    def test_empty_geometry_empty_scoring_and_undefined_values(self):
        for y,score in ((np.zeros((3,4)),np.ones((3,4),bool)),(np.full((3,4),7),np.zeros((3,4),bool))):
            p,kw=freeze(y,score=score);r=score_components(y,np.full(y.shape,np.nan),[7,2],geometry_mask=kw['geometry_mask'],score_mask=score,plan=p,expected_plan_hash=p.plan_hash)
            for group in r['groups'].values():self.assertEqual(group['count'],0);self.assertIsNone(group['oa']);self.assertEqual(group['macro_class_ids'],[])
            self.assertTrue(all(c['recall'] is None for c in r['components']));json.dumps(r,allow_nan=False)

    def test_freeze_access_labels_classes_and_prediction_validation(self):
        y=np.full((4,5),7.);geo=np.ones(y.shape,bool);geo[0,0]=False;y[0,0]=np.nan;p,kw=freeze(y,geometry=geo)
        args=dict(geometry_mask=geo,score_mask=geo,plan=p,expected_plan_hash=p.plan_hash)
        base=p.to_dict();pred=y.copy()
        first=score_components(y,pred,[7,2],**args);pred[1,1]=2;score_components(y,pred,[7,2],**args);self.assertEqual(base,p.to_dict())
        y2=y.copy();y2[0,0]=-999;self.assertEqual(first,score_components(y2,y,[7,2],**args))
        for changes in (dict(expected_plan_hash='0'*64),dict(score_mask=np.ones(y.shape,bool)),dict(geometry_mask=np.zeros(y.shape,bool))):
            with self.assertRaises(ValueError):score_components(y,pred,[7,2],**(args|changes))
        for bad in (np.zeros(y.shape),np.full(y.shape,99),np.full(y.shape,np.nan),np.zeros((2,2))):
            with self.assertRaises(ValueError):score_components(y,bad,[7,2],**args)
        with self.assertRaises(ValueError):score_components(y,pred,[2,7],**args)
        y2=y.copy();y2[1,1]=2
        with self.assertRaises(ValueError):score_components(y2,pred,[7,2],**args)
        edited=copy.deepcopy(base);edited['small_area']=999
        with self.assertRaises(ValueError):ComponentPlan(edited,expected_hash=p.plan_hash)

    def test_invalid_shapes_thresholds_ids_and_input_immutability(self):
        y=np.full((5,5),7);score=np.ones(y.shape,bool);old=y.copy();p,kw=freeze(y)
        for changed in (dict(small_area=0),dict(thin_depth=True),dict(geometry_mask=score.astype(int)),dict(score_mask=np.ones((2,2),bool))):
            with self.assertRaises(ValueError):component_geometry(y,[7,2],**(kw|changed))
        for bad in (np.full(y.shape,3),np.full(y.shape,-1),np.full(y.shape,.5),np.full(y.shape,np.inf)):
            with self.assertRaises(ValueError):component_geometry(bad,[7,2],**kw)
        for ids in ([],[7,7],[0,7],[True,7]):
            with self.assertRaises(ValueError):component_geometry(y,ids,**kw)
        np.testing.assert_array_equal(y,old);self.assertTrue(score.all())

    def test_thick_object_with_thin_tail_not_called_whole_thin(self):
        y=np.full((10,12),2);y[2:7,2:7]=7;y[4,7:10]=7
        p,_=freeze(y,thin_depth=2);r=next(c for c in p.to_dict()['components'] if c['class_id']==7)
        self.assertEqual(r['observed_area'],28);self.assertEqual(r['max_inward_depth'],3);self.assertFalse(r['thin'])
        p,_=freeze(y,thin_depth=3);self.assertTrue(next(c for c in p.to_dict()['components'] if c['class_id']==7)['thin'])


if __name__=='__main__':unittest.main()
