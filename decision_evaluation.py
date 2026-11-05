"""Explicit decision-only evaluation guard; old probability failures stay failures."""
import torch
from numerical_audit import compare

POLICY='exact_state_trainval_decisions_two_precision_sensitivity_v1'


def verify_decisions(reference, candidate):
    result=compare(reference,candidate)
    if result['classification_changes']:
        raise ValueError('Classification changed; decision evaluation refused')
    return {**result,'decision_gate_passed':True,'probability_calibration_validated':False}


def verify_state(reference, candidate):
    if set(reference)!=set(candidate):raise ValueError('Selected state keys changed')
    for name,old in reference.items():
        new=candidate[name]
        if (not torch.is_tensor(old) or not torch.is_tensor(new) or old.dtype!=new.dtype
                or old.shape!=new.shape or not torch.equal(old.cpu(),new.cpu())):
            raise ValueError('Selected state changed: '+name)
    return True
