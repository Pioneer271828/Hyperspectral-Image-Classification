import unittest,torch,json,subprocess,sys
from decision_evaluation import verify_decisions,verify_state


class DecisionEvaluationTests(unittest.TestCase):
    def test_pretest_entry_has_explicit_consumer_commit(self):
        r=subprocess.run([sys.executable,'research/reproduction/num2/evaluate.py','--help'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertIn('--expected-commit',r.stdout)

    def test_old_probability_failure_is_retained_in_decision_report(self):
        a=torch.tensor([[.8,.2]]);b=torch.tensor([[.8001,.1999]])
        r=verify_decisions(a,b);self.assertTrue(r['decision_gate_passed'])
        self.assertFalse(r['original_probability_bound_passed'])
        self.assertFalse(r['probability_calibration_validated']);json.dumps(r)

    def test_near_tie_class_change_fails_even_under_old_probability_tolerance(self):
        a=torch.tensor([[.500001,.499999]])
        with self.assertRaisesRegex(ValueError,'Classification'):verify_decisions(a,a.flip(1))
        with self.assertRaises(ValueError):verify_decisions(a,torch.tensor([[float('nan'),0.]]))

    def test_exact_state_rejects_changed_weight_buffer_dtype_and_missing_key(self):
        state={'weight':torch.tensor([.1]),'running_mean':torch.tensor([.3]),'counter':torch.tensor(2)}
        self.assertTrue(verify_state(state,{k:v.clone() for k,v in state.items()}))
        for key in state:
            bad={k:v.clone() for k,v in state.items()};bad[key]+=1
            with self.assertRaises(ValueError):verify_state(state,bad)
        bad=dict(state);bad['weight']=bad['weight'].double()
        with self.assertRaises(ValueError):verify_state(state,bad)
        bad=dict(state);bad.pop('counter')
        with self.assertRaises(ValueError):verify_state(state,bad)


if __name__=='__main__':unittest.main()
