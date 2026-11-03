"""Label-blind, real spectral subsets with explicit synchronized metadata (M1)."""
from dataclasses import dataclass, asdict, replace
import hashlib
import json
import math
import numpy as np


def _plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [_plain(v) for v in value]
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value),sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class BandMetadata:
    source_id: str
    band_ids: tuple
    wavelengths: tuple
    wavelength_unit: str
    wavelength_quality: str
    valid: tuple
    fwhm: tuple | None = None
    fwhm_unit: str | None = None

    def __post_init__(self):
        if not isinstance(self.source_id,str) or not self.source_id: raise ValueError('Explicit source identity required')
        if self.wavelength_unit not in ('nm','normalized') or self.wavelength_quality not in ('verified','approximate','index_only'):
            raise ValueError('Explicit wavelength unit/quality required')
        if self.wavelength_quality=='index_only' and self.wavelength_unit!='normalized':
            raise ValueError('Index-only positions must not be labelled nanometers')
        ids=tuple(self.band_ids);valid=tuple(self.valid)
        if not ids or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or x<0 for x in ids) or len(set(ids))!=len(ids):
            raise ValueError('Unique nonnegative original band IDs required')
        wave=np.asarray(self.wavelengths,dtype=np.float64)
        if wave.shape!=(len(ids),) or not np.isfinite(wave).all(): raise ValueError('Finite wavelength vector matching bands required')
        if self.wavelength_unit=='nm' and (wave<=0).any(): raise ValueError('Physical wavelengths must be positive')
        if self.wavelength_unit=='normalized' and ((wave<0).any() or (wave>1).any()): raise ValueError('Normalized positions must be within [0,1]')
        if len(valid)!=len(ids) or any(not isinstance(x,(bool,np.bool_)) for x in valid) or not any(valid):
            raise ValueError('Boolean validity vector must retain at least one band')
        if self.fwhm is None:
            if self.fwhm_unit is not None: raise ValueError('Missing FWHM has no unit')
        else:
            widths=np.asarray(self.fwhm,dtype=np.float64)
            if self.fwhm_unit!=self.wavelength_unit or widths.shape!=wave.shape or not np.isfinite(widths).all() or (widths<=0).any():
                raise ValueError('Positive FWHM must match wavelengths and their unit')
            object.__setattr__(self,'fwhm',tuple(float(x) for x in widths))
        object.__setattr__(self,'band_ids',tuple(int(x) for x in ids))
        object.__setattr__(self,'valid',tuple(bool(x) for x in valid))
        object.__setattr__(self,'wavelengths',tuple(float(x) for x in wave))

    @property
    def fingerprint(self): return _digest(asdict(self))

    def subset(self,positions):
        return BandMetadata(self.source_id,tuple(self.band_ids[i] for i in positions),
            tuple(self.wavelengths[i] for i in positions),self.wavelength_unit,self.wavelength_quality,
            tuple(self.valid[i] for i in positions),
            tuple(self.fwhm[i] for i in positions) if self.fwhm is not None else None,self.fwhm_unit)


@dataclass(frozen=True)
class ObservationView:
    observation: object
    bands: BandMetadata


def _validate_observation(observation,bands):
    if (not isinstance(bands,BandMetadata) or not isinstance(observation.spectral,np.ndarray)
            or observation.spectral.ndim!=3 or min(observation.spectral.shape[:2])<=0 or observation.spectral.shape[-1]!=len(bands.band_ids)):
        raise ValueError('HWC spectral observation and matching BandMetadata required')
    if observation.partition not in ('train','val','test') or not observation.plan_hash:
        raise ValueError('Partition and spatial plan identity required')


def _apply(observation,bands,positions):
    positions=tuple(positions)
    if (not positions or any(type(i) is not int or not 0<=i<len(bands.band_ids) or not bands.valid[i] for i in positions)
            or list(positions)!=sorted(set(positions))):
        raise ValueError('Nonempty increasing valid source positions required')
    if positions==tuple(range(len(bands.band_ids))): return ObservationView(observation,bands)
    values=np.take(observation.spectral,positions,axis=-1)
    values.setflags(write=False)
    return ObservationView(replace(observation,spectral=values),bands.subset(positions))


def sample(observation,bands,config,rng=None):
    """Return (view, record). Select all pixels consistently; never inspect labels or refit stats."""
    _validate_observation(observation,bands)
    modes={'none':set(),'random_subset':{'keep_fraction'},'uniform_stride':{'stride','offset'},
           'contiguous_gap':{'lower','upper','unit'},'range_crop':{'lower','upper','unit'}}
    if not isinstance(config,dict) or config.get('mode') not in modes or set(config)!={'mode'}|modes[config['mode']]:
        raise ValueError('Unknown mode or missing/extra sampling fields')
    config=json.loads(json.dumps(config,allow_nan=False));mode=config['mode']
    valid=np.flatnonzero(bands.valid);before=after=None
    if mode=='none': selected=valid
    elif mode=='random_subset':
        fraction=config['keep_fraction']
        if type(fraction) not in (int,float) or not 0<fraction<=1 or not isinstance(rng,np.random.Generator):
            raise ValueError('Fraction in (0,1] and explicit NumPy Generator required')
        before=_digest(rng.bit_generator.state)
        selected=np.sort(rng.choice(valid,size=max(1,math.ceil(len(valid)*fraction)),replace=False))
        after=_digest(rng.bit_generator.state)
    elif mode=='uniform_stride':
        stride,offset=config['stride'],config['offset']
        if type(stride) is not int or stride<=0 or type(offset) is not int or not 0<=offset<stride:
            raise ValueError('Positive integer stride and offset in [0,stride) required')
        selected=valid[np.asarray(bands.band_ids)[valid]%stride==offset]
    else:
        lower,upper=config['lower'],config['upper']
        if (type(lower) not in (int,float) or type(upper) not in (int,float) or not math.isfinite(lower)
                or not math.isfinite(upper) or lower>upper or config['unit']!=bands.wavelength_unit
                or bands.wavelength_quality=='index_only'):
            raise ValueError('Ordered finite range with matching physical metadata required; index-only range is not a sensor crop')
        inside=(np.asarray(bands.wavelengths)[valid]>=lower)&(np.asarray(bands.wavelengths)[valid]<=upper)
        selected=valid[inside if mode=='range_crop' else ~inside]
    positions=[int(i) for i in selected];view=_apply(observation,bands,positions)
    record={'schema_version':1,'source_metadata_hash':bands.fingerprint,'source_band_count':len(bands.band_ids),
            'partition':observation.partition,'plan_hash':observation.plan_hash,'config':config,
            'selected_positions':positions,'selected_band_ids':list(view.bands.band_ids),
            'selected_metadata_hash':view.bands.fingerprint,'rng_before_hash':before,'rng_after_hash':after,
            'random_count_rule':'ceil(valid_count*keep_fraction), minimum 1' if mode=='random_subset' else None}
    record['record_hash']=_digest(record)
    return view,record


def replay(observation,bands,record):
    """Use a frozen record, without generating a new subset or consuming an RNG."""
    _validate_observation(observation,bands)
    clean=json.loads(json.dumps(record,allow_nan=False));record_hash=clean.pop('record_hash',None)
    if (record_hash!=_digest(clean) or clean.get('schema_version')!=1
            or clean.get('source_metadata_hash')!=bands.fingerprint or clean.get('source_band_count')!=len(bands.band_ids)
            or clean.get('partition')!=observation.partition or clean.get('plan_hash')!=observation.plan_hash):
        raise ValueError('Frozen sampling record identity/hash mismatch')
    view=_apply(observation,bands,clean['selected_positions'])
    if clean['selected_band_ids']!=list(view.bands.band_ids) or clean['selected_metadata_hash']!=view.bands.fingerprint:
        raise ValueError('Frozen selected metadata mismatch')
    return view


def subset_statistics(statistics,bands,record):
    """Slice the existing per-band min/max values; never fit new target/subset statistics."""
    from model.research_config import digest
    if statistics.get('statistics_hash')!=digest({k:v for k,v in statistics.items() if k!='statistics_hash'}):
        raise ValueError('Source statistics hash mismatch')
    clean={k:v for k,v in record.items() if k!='record_hash'}
    if record.get('record_hash')!=_digest(clean) or record['source_metadata_hash']!=bands.fingerprint:
        raise ValueError('Sampling record mismatch')
    if statistics.get('plan_hash')!=record['plan_hash'] or statistics.get('fit_partition')!='train':
        raise ValueError('Require statistics fitted on the original training plan')
    if any(len(statistics.get(key,[]))!=len(bands.band_ids) for key in ('minimum','maximum')):
        raise ValueError('Statistics band count mismatch')
    positions=record['selected_positions']
    if not positions or positions!=sorted(set(positions)) or any(type(i) is not int or not 0<=i<len(bands.band_ids) or not bands.valid[i] for i in positions):
        raise ValueError('Invalid source positions')
    return {'kind':'selected_training_statistics','parent_statistics_hash':statistics['statistics_hash'],
            'sampling_record_hash':record['record_hash'],'band_ids':list(record['selected_band_ids']),
            'minimum':[statistics['minimum'][i] for i in positions],
            'maximum':[statistics['maximum'][i] for i in positions]}


