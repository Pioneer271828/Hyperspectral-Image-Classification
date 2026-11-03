"""Compact, independently auditable readouts for frozen calibrated baselines."""
import numpy as np
import torch

from evaluation.classification_metrics import classification_metrics

ARMS = ('raw_original_selected', 'calibrated_original_selected', 'calibrated_reselected')
SEEDS = (64, 512, 2048)


def compact_readout(labels, probabilities, classes):
    y = np.asarray(labels)
    q = probabilities.detach().cpu()
    if (y.ndim != 1 or type(classes) is not int or classes <= 0
        or q.ndim != 2 or q.shape != (len(y), classes) or not len(y)
        or not torch.isfinite(q).all() or (q < 0).any() or (q > 1).any()
        or not np.isfinite(y).all() or not np.equal(y, np.floor(y)).all()
        or (y < 1).any() or (y > classes).any() or classes > 32767):
        raise ValueError('Nonempty aligned finite probabilities and foreground labels required')
    truth = torch.as_tensor(y.astype(np.int64)-1)
    compact = {'prediction': (q.argmax(1)+1).to(torch.int16),
               'true_probability': q[torch.arange(len(y)), truth].clone()}
    return score_compact(y, compact, classes), compact


def score_compact(labels, compact, classes):
    y = np.asarray(labels)
    pred = compact['prediction'].cpu().numpy()
    true_q = compact['true_probability'].cpu()
    if (true_q.shape != (len(y),) or pred.shape != y.shape or not torch.isfinite(true_q).all()
        or (true_q < 0).any() or (true_q > 1).any()):
        raise ValueError('Invalid compact prediction record')
    result = classification_metrics(y, pred, list(range(1, classes+1)))
    if not result['count'] or result['count'] != len(y) or not result['full_class_coverage']:
        raise ValueError('All foreground classes must be represented')
    result['nll'] = float(-true_q.clamp_min(1e-8).log().mean())
    return result


def require_reuse_receipts(receipts, code):
    if set(receipts) != {'cpu', 'cuda'}:
        raise ValueError('CPU and CUDA reuse gates required before scoring')
    for device, receipt in receipts.items():
        if (receipt.get('status') != 'passed' or receipt.get('code') != code
            or receipt.get('device') != device or receipt.get('seeds') != list(SEEDS)
            or receipt.get('raw_training_unchanged') is not True
            or receipt.get('rng_unchanged') is not True or receipt.get('real_optimizer_updates') != 0):
            raise ValueError('Incomplete or mismatched reuse gate')


def scheduler_reuse(raw_history, calibrated_history, states, settings):
    """Use original raw NLL for exact saved-scheduler authentication."""
    from joint_training import _tree_fingerprint
    if not raw_history or len(raw_history) != len(calibrated_history) or len(states) != len(raw_history):
        raise ValueError('Complete paired validation scheduler histories required')
    if (any(b['step']<=a['step'] for a,b in zip(raw_history,raw_history[1:]))
        or any(not np.isfinite(e['val_loss']) for h in (raw_history,calibrated_history) for e in h)):
        raise ValueError('Finite chronological scheduler events required')
    def build():
        training = settings['config']['training']
        optimizer = torch.optim.Adam([
            {'params': [torch.nn.Parameter(torch.zeros(1))], 'lr': training['lr']},
            {'params': [torch.nn.Parameter(torch.zeros(1))], 'lr': training['head_lr']}])
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
            **settings['scheduler'], threshold_mode='rel')
    raw, calibrated = build(), build(); rows = []
    for a, b, saved in zip(raw_history, calibrated_history, states):
        if a['step'] != b['step'] or a['step'] != saved['cursor']:
            raise ValueError('Scheduler cursor mismatch')
        raw.step(a['val_loss']); calibrated.step(b['val_loss'])
        if _tree_fingerprint(raw.state_dict()) != _tree_fingerprint(saved['scheduler']):
            raise ValueError('Historical raw scheduler state did not reproduce')
        if raw.get_last_lr() != calibrated.get_last_lr():
            raise ValueError('Calibrated scheduler changes LR; trajectory reuse rejected')
        rows.append({'step': a['step'], 'raw_lr': raw.get_last_lr(), 'calibrated_lr': calibrated.get_last_lr()})
    return rows
