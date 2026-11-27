"""Budget, semantic and role failures must precede spectral payload access."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from build_plan import select_scene
from c6_bridge import canonical
from pair_inputs import PairInputs,phase_binding
from review_metadata import validate_ontology


class PairGates(unittest.TestCase):
    def setUp(self):
        self.ontology=json.loads((Path(__file__).parent/'ontology.json').read_text())
        rows=[{'id':10,'role':'devtrain','parent_tile':1,'box':[0,20,0,20]},{'id':20,'role':'devval','parent_tile':2,'box':[30,50,0,20]}]
        samples={}
        for row in rows:
            start=row['box'][0]*100;samples[row['id']]=[[start+band*100+i,label] for band,label in enumerate((1,7,6)) for i in range(5)]
        self.bank=SimpleNamespace(scene='xuzhou',pin={'bank':'synthetic','registration':'synthetic','shape':(100,100,436)},rows={r['id']:r for r in rows},samples=samples,registration={'statistics':{'synthetic':True},'bands':{'synthetic':True}},payload_reads=0)
        self.bank.role_rows=lambda role:tuple(r for r in self.bank.rows.values() if r['role']==role)
    def test_deterministic_merge_preserves_total_budget_and_original_ids(self):
        a=select_scene(self.bank,self.ontology);original={(p,c) for points in self.bank.samples.values() for p,c in points}
        for rows in a['roles'].values():
            self.assertEqual(len(rows),10);self.assertEqual(sum(p['common_id']==0 for p in rows),5);self.assertTrue(all((p['point_id'],p['original_label']) in original for p in rows))
        for points in self.bank.samples.values():points.reverse()
        self.assertEqual(canonical(a),canonical(select_scene(self.bank,self.ontology)));self.assertEqual(self.bank.payload_reads,0)
    def test_missing_tree_does_not_borrow_validation_or_add_a_label(self):
        self.bank.samples[10]=self.bank.samples[10][:-1];before=deepcopy(self.bank.samples)
        with self.assertRaisesRegex(ValueError,'lacks original'):select_scene(self.bank,self.ontology)
        self.assertEqual(self.bank.samples,before);self.assertEqual(self.bank.payload_reads,0)
    def test_shared_parent_is_excluded_instead_of_reassigned(self):
        self.bank.rows[20]['parent_tile']=1
        with self.assertRaisesRegex(ValueError,'lacks original'):select_scene(self.bank,self.ontology)
        self.assertEqual(self.bank.rows[20]['role'],'devval');self.assertEqual(self.bank.payload_reads,0)
    def test_geometry_overlap_rejected_even_with_distinct_parent_names(self):
        self.bank.rows[20]['box']=[0,20,0,20]
        with self.assertRaisesRegex(ValueError,'boxes overlap'):select_scene(self.bank,self.ontology)
    def test_historical_wrong_tree_or_roof_equivalence_not_silently_accepted(self):
        for raw in (5,9):
            ontology=deepcopy(self.ontology);ontology['classes'][1]['xuzhou_original_ids']=[raw]
            with self.assertRaisesRegex(ValueError,'Unreviewed'):validate_ontology(ontology)
    def test_raw_unannotated_zero_never_becomes_common_ground_zero(self):
        self.bank.samples[10].append([505,0]);plan=select_scene(self.bank,self.ontology)
        self.assertTrue(all(p['original_label']>0 for rows in plan['roles'].values() for p in rows))
    def test_fit_phase_never_maps_to_exposed_development_validation(self):
        self.assertEqual(phase_binding(('xuzhou','whu-honghu'),'source_fit'),('xuzhou','devtrain'));self.assertEqual(phase_binding(('xuzhou','whu-honghu'),'target_development'),('whu-honghu','devval'))
        for phase in ('query','test','confirmation','target_fit_on_validation'):
            with self.assertRaises(ValueError):phase_binding(('xuzhou','whu-honghu'),phase)
    def test_unregistered_training_and_confirmation_fail_before_any_source_open(self):
        with patch('pair_inputs.check_source',side_effect=AssertionError('Source opened')):
            for purpose in ('training','independent_confirmation','causal_sensor_identification'):
                with self.assertRaisesRegex(ValueError,'not authorized'):PairInputs(expected_commit=None,run_root=None,inputs=None,plan_file=None,direction=('xuzhou','whu-honghu'),purpose=purpose)
    def test_same_acquisition_cannot_be_both_devices(self):
        with patch('pair_inputs.check_source',side_effect=AssertionError('Source opened')):
            with self.assertRaisesRegex(ValueError,'independent acquisitions'):PairInputs(expected_commit=None,run_root=None,inputs=None,plan_file=None,direction=('xuzhou','xuzhou'))


if __name__=='__main__':unittest.main()
