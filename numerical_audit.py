"""Descriptive probability/precision audit; never relaxes an acceptance gate."""
import torch


def compare(reference, candidate):
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] < 2 or not reference.numel():
        raise ValueError('Matching nonempty class probability matrices required')
    for value in (reference, candidate):
        if (not value.is_floating_point() or not torch.isfinite(value).all()
                or (value < 0).any() or (value > 1).any()
                or not torch.allclose(value.sum(1), torch.ones_like(value[:,0]), rtol=0, atol=2e-6)):
            raise ValueError('Invalid probabilities')
    a, b = reference.double(), candidate.double()
    delta = (a-b).abs(); changed = a.argmax(1) != b.argmax(1)
    top = a.topk(2, dim=1).values; margins = top[:,0]-top[:,1]
    return {'reference_dtype':str(reference.dtype),'candidate_dtype':str(candidate.dtype),
            'max_abs_error':float(delta.max()),'mean_abs_error':float(delta.mean()),
            'elements_over_original_1e_5':int((delta > 1e-5).sum()),
            'original_probability_bound_passed':bool(delta.max() <= 1e-5),
            'classification_changes':int(changed.sum()),'rows':len(a),
            'changed_reference_margin_max':float(margins[changed].max()) if changed.any() else None}


def verify_exact_double_cast(original, converted):
    if set(original) != set(converted): raise ValueError('State keys changed')
    for name, old in original.items():
        value = converted[name].cpu()
        if old.is_floating_point():
            if value.dtype != torch.float64 or not torch.equal(old.double(),value):
                raise ValueError('Nonexact double state cast: '+name)
        elif not torch.equal(old,value): raise ValueError('Integer state changed: '+name)
    return True
