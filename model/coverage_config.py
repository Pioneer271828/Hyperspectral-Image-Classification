"""Pure-Python E1 settings; preserve canonical pre-E1 default configurations."""
import math

COVERAGE_DEFAULTS = dict(coverage_mode='none', coverage_hidden=16, coverage_strength=.1,
                         coverage_sigma=.08, coverage_window=.16, coverage_init_seed=0,
                         coverage_descriptor_version=1)


def coverage_options(values):
    result = {k: values.get(k, v) for k, v in COVERAGE_DEFAULTS.items()}
    if result['coverage_mode'] not in ('none', 'affine'):
        raise ValueError('coverage_mode must be none or affine')
    for k in ('coverage_hidden', 'coverage_init_seed', 'coverage_descriptor_version'):
        if type(result[k]) is not int:raise ValueError(k+' must be an integer')
    if result['coverage_hidden'] <= 0 or not 0 <= result['coverage_init_seed'] < 2**32 or result['coverage_descriptor_version'] != 1:
        raise ValueError('Invalid coverage hidden/seed/schema')
    for k in ('coverage_strength', 'coverage_sigma', 'coverage_window'):
        if type(result[k]) not in (int, float) or not math.isfinite(result[k]):raise ValueError(k+' must be finite')
    if (not 0 <= result['coverage_strength'] <= 1 or result['coverage_sigma'] <= 0
            or not 1e-12 <= result['coverage_window'] <= 1):raise ValueError('Invalid coverage strength/sigma/window')
    if result['coverage_mode'] != 'none' and not 0 < values.get('wavelength_min',400.) < values.get('wavelength_max',2500.):
        raise ValueError('Enabled coverage needs an ordered positive global nm reference')
    return result


def strip_default_coverage(values):
    result = dict(values)
    if all(result.get(k,v) == v for k,v in COVERAGE_DEFAULTS.items()):
        for k in COVERAGE_DEFAULTS:result.pop(k,None)
    return result


def coverage_signature(values):
    return {**coverage_options(values), **{k:values[k] for k in
        ('wavelength_min','wavelength_max','adapter_attention_mode')}}


def coverage_enabled(model):
    return getattr(getattr(model,'backbone',None),'coverage_mode','none') != 'none'


def coverage_forward_kwargs(model, metadata):
    if not coverage_enabled(model):return {}
    result = {'coverage_metadata':metadata}
    if metadata is not None:
        result.update(wavelength_unit=metadata.wavelength_unit, fwhm_unit=metadata.fwhm_unit or 'auto')
    return result


def coverage_trace(model, metadata):
    return {'coverage_metadata_hash':getattr(metadata,'fingerprint',None)} if coverage_enabled(model) else {}
