"""Paired point bookkeeping; no fitting, training, or independence assumptions."""
import hashlib
import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(scene, bank, tile, ordinal=None, band=None):
    raw = '|'.join(map(str, (scene, bank, tile, ordinal, band)))
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def nll(probabilities, label):
    require(1 <= label <= len(probabilities), 'Class outside probability vector')
    require(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities), 'Invalid probability')
    require(abs(sum(probabilities)-1) < 1e-5, 'Probability mass differs')
    return -math.log(max(probabilities[label-1], 1e-8))


def decompose(content, coefficient_mean, trained_mean, label):
    """A-C = (A-B)+(B-C), with B using content-trained weights and mean coefficients."""
    a, b, c = (nll(p, label) for p in (content, coefficient_mean, trained_mean))
    total, coefficient, residual = a-c, a-b, b-c
    require(abs(total-coefficient-residual) <= 1e-12, 'Point decomposition differs')
    pred_a = max(range(len(content)), key=content.__getitem__)+1
    pred_c = max(range(len(trained_mean)), key=trained_mean.__getitem__)+1
    pred_b = max(range(len(coefficient_mean)), key=coefficient_mean.__getitem__)+1
    transition = ('correct' if pred_c == label else 'wrong')+'->'+('correct' if pred_a == label else 'wrong')
    return {'nll_content': a, 'nll_content_weights_mean_coeff': b, 'nll_trained_mean': c,
            'total_delta': total, 'coefficient_delta': coefficient, 'training_residual_delta': residual,
            'pred_content': pred_a, 'pred_content_weights_mean_coeff': pred_b,
            'pred_trained_mean': pred_c, 'transition': transition,
            'coefficient_transition': ('correct' if pred_b == label else 'wrong')+'->'+('correct' if pred_a == label else 'wrong'),
            **{'true_class_margin_'+name: p[label-1]-max(v for i,v in enumerate(p) if i != label-1)
               for name,p in [('content',content), ('content_weights_mean_coeff',coefficient_mean), ('trained_mean',trained_mean)]},
            'p_true_content': content[label-1], 'p_true_content_weights_mean_coeff': coefficient_mean[label-1],
            'p_true_trained_mean': trained_mean[label-1]}


def concentration(weights):
    require(weights and all(math.isfinite(v) and v >= 0 for v in weights), 'Invalid proxy weights')
    mass = sum(weights)
    require(mass > 0, 'Empty proxy distribution')
    return mass*mass/sum(v*v for v in weights)
