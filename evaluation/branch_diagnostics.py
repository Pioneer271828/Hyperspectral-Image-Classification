"""Branch correctness strata and explicit fusion correction/harm denominators."""
import numpy as np

from .classification_metrics import evaluation_mask, class_order, class_indices


def branch_diagnostics(labels, cnn_prediction, graph_prediction, fused_prediction, class_ids, mask=None):
    labels = np.asarray(labels)
    valid = evaluation_mask(labels, mask)
    ids = class_order(class_ids)
    truth = class_indices(labels[valid], ids, 'Labels')
    vectors = []
    for name, value in (('CNN', cnn_prediction), ('Graph', graph_prediction), ('Fused', fused_prediction)):
        value = np.asarray(value)
        if value.shape != labels.shape:
            raise ValueError(name + ' prediction shape differs from labels')
        vectors.append(class_indices(value[valid], ids, name + ' predictions'))
    cnn, graph, fused = vectors
    c, g, f = cnn == truth, graph == truth, fused == truth
    strata = {}
    for name, selected in (('both_correct', c & g), ('cnn_only_correct', c & ~g),
                            ('graph_only_correct', ~c & g), ('both_wrong', ~c & ~g)):
        denominator = int(selected.sum())
        corrected = int((selected & f).sum())
        strata[name] = {'count': denominator, 'fused_correct': corrected,
                        'fused_accuracy': corrected / denominator if denominator else None}
    count = len(truth)
    disagreement = cnn != graph
    correctable = c ^ g
    rescue = int((correctable & f).sum())
    harm = int((c & g & ~f).sum())
    return dict(count=count, strata=strata, cnn_accuracy=float(c.mean()) if count else None,
                graph_accuracy=float(g.mean()) if count else None, fused_accuracy=float(f.mean()) if count else None,
                branch_oracle_accuracy=float((c | g).mean()) if count else None,
                prediction_disagreement_count=int(disagreement.sum()),
                correction={'numerator': rescue, 'denominator': int(correctable.sum()),
                            'rate': rescue / int(correctable.sum()) if correctable.any() else None,
                            'population': 'exactly_one_branch_correct'},
                harm={'numerator': harm, 'denominator': int((c & g).sum()),
                      'rate': harm / int((c & g).sum()) if (c & g).any() else None,
                      'population': 'both_branches_correct'})
