"""Versioned synthetic land-cover fields and frozen observation conditions."""

from dataclasses import asdict, dataclass, replace
from itertools import product

import numpy as np

from loadData.observation_sampling import BandMetadata, sample
from loadData.spatial_protocol import PartitionObservation
from model.research_config import digest
from transfer.development_blocks import array_hash
from transfer.development_bank_training import _condition
from transfer.native_scale_observation import FACTORS, Lattice, _immutable, _materialize
from transfer.native_scale_supervision import complete_targets


VERSION = "spatial_spectral_fields_v1"
STREAMS = (("devtrain",61317), ("devval",73241), ("confirmation",91577))
BANDS = ("full","even_retained_positions")
INTERVENTIONS = ("matched","missing","permuted","perturbed","spectral_only")
SCENE_SHAPE = (82, 81)
PROFILES = ("a","b","c","d")
HOLDOUT = "factor4 profiles c,d, both band views"


@dataclass(frozen=True)
class Physical:
    key: str
    base_gsd_m: float
    unit: str
    wavelength_shift_nm: float
    fwhm_nm: float

    def bands(self, count):
        centers = np.linspace(450,850,count)+self.wavelength_shift_nm
        return BandMetadata(f"{VERSION}:{self.key}",tuple(range(count)),tuple(centers),
                            "nm","verified",(True,)*count,(self.fwhm_nm,)*count,"nm")


def profiles():
    return tuple(Physical(key,gsd,"m/pixel",shift,fwhm) for key,gsd,shift,fwhm in
                 zip(PROFILES,(0.65,1.1,1.75,2.6),(0.,18.,42.,72.),(9.,12.,15.,19.),strict=True))


def registration(classes):
    if type(classes) is not int or classes not in (9,22):
        raise ValueError("Synthetic fixture requires nine or twenty-two classes")
    count = classes+2
    fields = []
    for role, seed in STREAMS:
        if role=="devtrain":
            keys, boxes = (3,10,20,36,48,57), ((3,16,2,19),(5,20,23,38),(4,21,43,56),
                                                (27,40,4,21),(25,40,25,42),(28,45,46,61))
        elif role=="devval":
            keys, boxes = (81,99), ((49,64,4,21),(52,69,29,44))
        else:
            keys, boxes = (121,147,165,183), ((7,22,61,78),(28,45,63,78),(51,64,55,72),(66,81,6,19))
        for order,(tile,box) in enumerate(zip(keys,boxes,strict=True)):
            fields.append(dict(tile=tile,role=role,seed=seed+classes*100+order,box=list(box)))
    return dict(version=VERSION,classes=classes,dataset={9:"xuzhou",22:"whu-honghu"}[classes],
                shape=list(SCENE_SHAPE),count=count,fields=fields,profiles=[asdict(p) for p in profiles()],
                statistics=dict(fit="fixed_generator_radiometric_bounds_never_refit",minimum=[-1.]*count,
                                maximum=[2.]*count), holdout=HOLDOUT)


def _field(rng, h, w, classes):
    yy,xx = np.mgrid[:h,:w]
    # Six coarse patches, long thin strips, fragmented islands and singleton rare classes.
    common = classes-4
    patterns = np.arange(1,common+1)
    rng.shuffle(patterns)
    base = ((yy//4)*((w+4)//5)+xx//5) % common
    labels = patterns[base]
    labels[:,w//2] = classes-3
    labels[h//3,:w//2] = classes-2
    labels[h-3:h-1,w-4:w-2] = classes-1
    labels[1,w-2] = classes
    # The 22-class fixture may have more common classes than the block has
    # macropatches; reserved individual points ensure every class is observed.
    locations = rng.permutation(h*w)
    missing = sorted(set(range(1,classes+1))-set(labels.ravel()))
    for label in missing:
        for index in locations:
            if np.count_nonzero(labels == labels.flat[index])>1:
                labels.flat[index]=label
                break
    return np.ascontiguousarray(labels,dtype=np.int64)


def _spectra(labels, profile, count, rng):
    bands = profile.bands(count)
    wave = np.asarray(bands.wavelengths)
    peak = 465 + (labels-1)*380/(count-3)
    width = 24+profile.fwhm_nm/3
    signal = np.exp(-0.5*((wave-peak[:,:,None])/width)**2)
    yy,xx = np.mgrid[:labels.shape[0],:labels.shape[1]]
    texture = 0.075*np.sin((yy+2*xx)/3.0)+0.035*np.cos(yy/1.8)
    noise = rng.normal(0,0.018,signal.shape)
    values = 0.18+texture[:,:,None]+signal+noise+0.10*np.sin(labels[:,:,None]*0.73+wave/124)
    return values.astype(np.float32),bands


@dataclass(frozen=True)
class Condition:
    catalog: str
    observation: str
    intervention: str
    source_key: tuple
    bands: BandMetadata | None
    coverage: float | None
    gsd_m: float | None
    unit: str | None

    def values(self):
        return self.bands,self.coverage,self.gsd_m

    def validate(self, actual, receipt):
        # Resolve the exact descriptor from the registered catalog, including
        # its donor. A caller-created or replaced dataclass cannot change it.
        source = receipt["source"]
        if (receipt["synthetic_profile"] is None or self.catalog != digest(registration(actual.classes))
                or self.observation != receipt["observation"]
                or self != condition_for(actual, self.intervention)):
            raise ValueError("Selector condition differs from the frozen observation and catalog")
        if source["fixture"] != VERSION:
            raise ValueError("Unsupported synthetic generator")


def _catalog_key(role, factor, band, profile):
    if role not in dict(STREAMS) or band not in BANDS or factor not in FACTORS or profile not in PROFILES:
        raise ValueError("Unknown condition slot")
    heldout = factor==4 and profile in ("c","d")
    if heldout != (role=="confirmation"):
        raise ValueError("Observation condition escapes its registered stream")
    return (factor,band,profile)


def condition_for(actual, intervention):
    if intervention not in INTERVENTIONS or actual.receipt["synthetic_profile"] is None:
        raise ValueError("Physical controls are available on the frozen synthetic catalog only")
    info = actual.receipt
    band = info["source"]["band_view"]
    role = info["role"]
    key = _catalog_key(role,info["factor"],band,info["synthetic_profile"])
    group = ("c","d") if role=="confirmation" else (("a","b") if key[0]==4 else PROFILES)
    index = group.index(key[2])
    donor = group[(index+1)%len(group)] if intervention=="permuted" else key[2]
    profile = next(p for p in profiles() if p.key==donor)
    all_bands = profile.bands(actual.classes+2)
    positions = info["source"]["positions"]
    selected = all_bands.subset(positions)
    coverage = len(positions)/len(all_bands.band_ids)
    gsd = profile.base_gsd_m*key[0]
    if intervention=="missing":
        selected,coverage,gsd = None,None,None
    elif intervention=="perturbed":
        selected = replace(selected,wavelengths=tuple(wave+30. for wave in selected.wavelengths))
        gsd *= 1.25
    elif intervention=="spectral_only":
        gsd = None
    return Condition(digest(registration(actual.classes)),info["observation"],intervention,
                     (key[0],band,donor),selected,coverage,gsd,None if gsd is None else "m/pixel")


class Fixture:
    def __init__(self, classes, role, *, _confirmation=None):
        if role not in dict(STREAMS) or (role=="confirmation" and _confirmation is not _CONFIRM):
            raise ValueError("Confirmation stream requires the completed freeze gate")
        self._classes = classes
        self._role = role
        self._registration = registration(classes)
        self._fields = {row["tile"]:row for row in self._registration["fields"] if row["role"]==role}
        self._payloads = {}
        self._physical = {p.key:p for p in profiles()}
        self._plan = digest(self._registration)

    @property
    def tiles(self):
        return tuple(self._fields)

    def _payload(self,tile,profile):
        key = (tile,profile.key)
        if key not in self._payloads:
            row = self._fields[tile]
            y0,y1,x0,x1 = row["box"]
            rng = np.random.default_rng(row["seed"])
            labels = _field(rng,y1-y0,x1-x0,self._classes)
            raw,bands = _spectra(labels,profile,self._registration["count"],rng)
            self._payloads[key] = _immutable(labels),_immutable(raw),bands
        return self._payloads[key]

    def materialize(self, tile, factor, band, profile_key, producer):
        _catalog_key(self._role,factor,band,profile_key)
        if type(tile) is not int or tile not in self._fields:
            raise ValueError("Synthetic tile escapes its registered stream")
        row = self._fields[tile]
        lattice = Lattice.make(row["box"],self._registration["shape"],factor)
        profile = self._physical[profile_key]
        labels,raw,original_bands = self._payload(tile,profile)
        fine = Lattice.make(row["box"],self._registration["shape"],1)
        y0,y1,x0,x1 = fine.box
        yy,xx = np.mgrid[y0:y1,x0:x1]
        partition = "train" if self._role=="devtrain" else ("val" if self._role=="devval" else "test")
        # Bounds are fixed by the generator contract, shared across streams
        # and profiles, and never estimated from observed evaluation values.
        normalized = ((raw+1.)/3.).astype(np.float32)
        original = PartitionObservation(partition,normalized,fine.coordinates(),yy*SCENE_SHAPE[1]+xx,self._plan)
        selected, record = sample(original,original_bands,_condition(band))
        source = dict(fixture=VERSION,registration=self._plan,seed=row["seed"],tile=tile,
                      raw=array_hash(raw),labels=array_hash(labels),positions=record["selected_positions"],
                      band_view=band,selection=record["record_hash"],statistics=digest(self._registration["statistics"]))
        actual = _materialize(self._registration["dataset"],self._classes,selected,lattice,source,
                              self._role,producer,profile=profile)
        return actual,labels,complete_targets(actual,labels)


# Used only by the confirmation consumer after all freezes have been verified.
_CONFIRM = object()


def confirmation_catalog():
    """The full held-out Cartesian roster, with a frozen cyclic bijection within band view."""
    return tuple(product((4,), BANDS, ("c","d")))
