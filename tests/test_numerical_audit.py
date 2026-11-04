import json,unittest
import torch
from numerical_audit import compare,verify_exact_double_cast


class NumericalAuditTests(unittest.TestCase):
    def test_exposes_bound_failure_without_relabeling_it_passed(self):
        a=torch.tensor([[.8,.2],[.500001,.499999]])
        b=a.clone();b[0]+=torch.tensor([3e-5,-3e-5]);b[1]=b[1].flip(0)
        r=compare(a,b);self.assertFalse(r['original_probability_bound_passed'])
        self.assertEqual(r['classification_changes'],1);self.assertEqual(r['elements_over_original_1e_5'],2)
        self.assertLess(r['changed_reference_margin_max'],3e-6);json.dumps(r)
        exact=compare(a,a.double());self.assertEqual(exact['max_abs_error'],0)
        self.assertTrue(exact['original_probability_bound_passed'])

    def test_bad_shape_nonfinite_and_non_probability_rejected(self):
        a=torch.tensor([[.8,.2]])
        for b in (torch.ones(1,3)/3,torch.tensor([[float('nan'),0.]]),torch.tensor([[.8,.3]])):
            with self.assertRaises(ValueError):compare(a,b)

    def test_double_cast_requires_exact_parameters_buffers_and_keys(self):
        old={'w':torch.tensor([.1,.2]),'counter':torch.tensor(3)}
        new={'w':old['w'].double(),'counter':old['counter'].clone()}
        self.assertTrue(verify_exact_double_cast(old,new))
        new['w'][0]+=1e-10
        with self.assertRaises(ValueError):verify_exact_double_cast(old,new)
        with self.assertRaises(ValueError):verify_exact_double_cast(old,{'w':old['w'].double()})


if __name__=='__main__':unittest.main()
