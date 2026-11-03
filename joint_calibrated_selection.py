"""BN-ALIGN2 opt-in policy; training continuation always retains raw BN state."""
import copy
import math
from pathlib import Path
import tempfile

import torch

from loadData.dataset_registry import file_hash
from model.bn_calibration import CONTEXT, POLICY, calibrate_denoise, verify_calibration_binding
from model.checkpoint_io import load_model_checkpoint

PROTOCOL = 'base97_honghu_calibrated_selection_v1'
VALIDATION_POLICY = {'calibration': POLICY, 'context': CONTEXT,
                     'scheduler': 'calibrated_fused_validation_nll',
                     'selection': 'macro_fused_validation_oa_strict_first_maximum',
                     'continuation': 'raw_training_state'}


def calibrated_settings(settings):
    if settings.get('protocol') != 'base97_honghu_joint_training_v1' or 'validation_policy' in settings:
        raise ValueError('Expected unmodified legacy joint settings')
    result = copy.deepcopy(settings)
    result.update(protocol=PROTOCOL, validation_policy=copy.deepcopy(VALIDATION_POLICY))
    return result


def enabled(settings):
    if 'validation_policy' not in settings:
        if settings.get('protocol') == PROTOCOL:
            raise ValueError('Calibrated protocol requires its validation policy')
        return False
    if settings.get('protocol') != PROTOCOL or settings['validation_policy'] != VALIDATION_POLICY:
        raise ValueError('Unknown calibrated selection policy')
    return True


def observations(items):
    return {name: {key: item[key] for key in ('net_input', 'wavelengths', 'fwhm')}
            for name, item in items.items()}


def validation_candidate(model, items, source_path):
    """Persist the exact raw source before calibrating a separate eval clone."""
    source_path = Path(source_path)
    raw = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    # A repeated interrupted validation may replace its owned source file, never a
    # historical run. run_training guards ownership and the immutable completion.
    # Save via a file object so the ZIP root does not encode a random temporary
    # basename; identical raw states have identical source hashes after resume.
    with tempfile.NamedTemporaryFile(dir=source_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(raw, stream)
        except Exception:
            stream.close(); temporary.unlink(missing_ok=True)
            raise
    try: temporary.replace(source_path)
    finally: temporary.unlink(missing_ok=True)
    candidate, receipt = calibrate_denoise(model, observations(items), mode=POLICY,
        context=CONTEXT, source_checkpoint_sha256=file_hash(source_path))
    return candidate, receipt


def load_selected(path, items, raw_source_path, *, device='cpu'):
    """Bound inference entry; the legacy general loader alone does not enforce it."""
    model = load_model_checkpoint(path, device=device)
    verify_calibration_binding(model, observations(items), context=CONTEXT,
                               source_checkpoint_sha256=file_hash(raw_source_path))
    return model


def first_maximum(events):
    """Chronological strict-greater selection, preserving the first tie."""
    if not events or any(b['step'] <= a['step'] for a, b in zip(events, events[1:])):
        raise ValueError('Nonempty chronological validation events required')
    if any(not math.isfinite(v['macro_oa']) for v in events):
        raise ValueError('Finite selection scores required')
    best = events[0]
    for value in events[1:]:
        if value['macro_oa'] > best['macro_oa']:
            best = value
    return best
