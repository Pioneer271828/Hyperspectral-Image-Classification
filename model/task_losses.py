"""Named legacy probability losses; no implicit migration to logits CE."""
import torch

from .utils import compute_loss


def probability_loss_components(outputs, onehot, mask, *, class_balance=False,
                                graph_loss_weights=None, cnn_loss_weight=1.0,
                                graph_branch_loss_weight=1.0):
    y, y_graph, y_cnn = outputs[:3]
    terms = {
        'fused_supervised': compute_loss(y, onehot, mask, class_balance=class_balance),
        'graph_supervised': compute_loss(y_graph, onehot, mask, class_balance=class_balance),
        'cnn_supervised': compute_loss(y_cnn, onehot, mask, class_balance=class_balance),
    }
    terms['weighted_fused_supervised'] = terms['fused_supervised']
    terms['weighted_graph_supervised'] = graph_branch_loss_weight * terms['graph_supervised']
    terms['weighted_cnn_supervised'] = cnn_loss_weight * terms['cnn_supervised']
    # Preserve the legacy operation order, normalization and clamped probability loss.
    total = (terms['weighted_fused_supervised'] + terms['weighted_graph_supervised']
             + terms['weighted_cnn_supervised'])
    info = outputs[3] if len(outputs) > 3 else {}
    use_graph_losses = len(outputs) > 3 and graph_loss_weights is not None
    for name, legacy_name in (('compactness', 'compactness_loss'),
                              ('entropy', 'assignment_entropy_loss'),
                              ('balance', 'assignment_balance_loss')):
        value = info.get(legacy_name, 0.0)
        value = value if torch.is_tensor(value) else y.new_tensor(value)
        terms['graph_' + name] = value
        weight = graph_loss_weights[name] if use_graph_losses else 0.0
        terms['weighted_graph_' + name] = weight * value
    if use_graph_losses:
        total = (total + terms['weighted_graph_compactness'] + terms['weighted_graph_entropy']
                 + terms['weighted_graph_balance'])
    terms['total'] = total
    return terms


def detached_loss_values(components):
    """Log scalars with one device transfer; do not use these values to train."""
    names = tuple(components)
    values = torch.stack([components[name].detach() for name in names]).cpu().tolist()
    return dict(zip(names, values))
