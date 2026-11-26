"""Private WP07 bridge for authenticated band indices, distinct from physical wavelengths."""
from contextlib import contextmanager
from dataclasses import dataclass, replace
from copy import deepcopy
import math
import torch
from torch import nn
from model.research_config import digest
from transfer.conditional_multiscale import ObservationConditionEncoder
from transfer.multiscale_training import MultiscaleTrial
from transfer.native_scale_fixture import condition_for
from transfer.wp07_graph_range import require, require_checks

MODES = ('mean', 'content', 'metadata', 'joint')
SCOPES = ('both', 'cnn', 'graph')
INTERVENTIONS = ('matched', 'missing', 'swap_band', 'unknown_wavelength', 'missing_gsd', 'permuted_profile', 'equal_count_positions')
FEATURES = ('retained_fraction','minimum_position','maximum_position','mean_position',
            'position_std','mean_gap','maximum_gap','gap_std')


def index_features(positions, master_count):
    require(type(master_count) is int and master_count > 1, 'Explicit original processed-band count required')
    require(positions and all(type(v) is int and 0 <= v < master_count for v in positions)
            and len(set(positions))==len(positions), 'Invalid or duplicate original processed-band position')
    values=sorted(v/(master_count-1) for v in positions)
    avg=sum(values)/len(values);gaps=[b-a for a,b in zip(values,values[1:])]
    gapavg=sum(gaps)/len(gaps) if gaps else 0.
    return (len(values)/master_count,values[0],values[-1],avg,
        math.sqrt(sum((v-avg)**2 for v in values)/len(values)),gapavg,max(gaps,default=0.),
        math.sqrt(sum((v-gapavg)**2 for v in gaps)/len(gaps)) if gaps else 0.)


@dataclass(frozen=True)
class IndexPacket:
    bands: object
    positions: tuple
    master_count: int
    original_physical_mapping_authenticated: bool = False


class IndexConditionEncoder(nn.Module):
    """Use declared index sampling without upgrading approximate wavelengths to measured metadata."""
    def __init__(self, physical=None):
        super().__init__()
        self.physical = ObservationConditionEncoder() if physical is None else physical
        self.indices = nn.Sequential(nn.Linear(len(FEATURES),32),nn.GELU())

    def forward(self, packet, coverage_fraction, gsd_m, reference):
        require(packet is None or isinstance(packet,IndexPacket), 'Registered index packet required')
        if packet is None:
            return self.physical(None,coverage_fraction,gsd_m,reference)
        require(not packet.original_physical_mapping_authenticated, 'This bridge has no authenticated sensor mapping')
        vector=index_features(packet.positions,packet.master_count)
        require(coverage_fraction is None or coverage_fraction==vector[0], 'Coverage does not match actual positions')
        # The existing encoder still rejects approximate/index-only wavelengths.
        physical=self.physical(packet.bands,None,gsd_m,reference)
        observed=self.indices(reference.new_tensor(vector))
        return observed if physical is None else observed+physical


@dataclass(frozen=True)
class BoundCondition:
    catalog: str
    observation: str
    master_count: int
    intervention: str
    packet: IndexPacket | None
    gsd_m: float | None
    source_key: tuple

    def values(self):
        return self.packet,None,self.gsd_m

    def validate(self, actual, receipt):
        require(self.observation==receipt['observation'] and self==bind_condition(actual,self.master_count,self.catalog,self.intervention),
                'Condition packet, donor, missing mask or observed input changed')


def bind_condition(actual, master_count, catalog, intervention='matched'):
    require_checks();require(intervention in INTERVENTIONS and isinstance(catalog,str) and catalog, 'Unknown condition catalog/intervention')
    receipt=actual.receipt;synthetic=receipt['synthetic_profile'] is not None
    if synthetic:require(master_count==actual.classes+2, 'Synthetic original-band count differs')
    positions=tuple(int(i) for i in receipt['source']['positions'])
    require(len(positions)==len(actual.view.bands.band_ids), 'Actual view/position count differs')
    index_features(positions,master_count)
    bands=actual.view.bands;gsd=None
    if synthetic:
        source=condition_for(actual,'permuted' if intervention=='permuted_profile' else 'matched')
        bands,_,gsd=source.values()
    else:
        require(intervention not in ('permuted_profile','equal_count_positions'), 'Unobserved physical/index donors are fixture-only')
    if intervention=='swap_band':
        positions=tuple(range(0,master_count,2)) if len(positions)==master_count else tuple(range(master_count))
        # Physical centers of an unmaterialized donor are deliberately missing.
        bands=None;gsd=None if not synthetic else gsd
    elif intervention=='equal_count_positions':
        require(len(positions)<master_count and len(positions)>2, 'No equal-count alternative for a full/small set')
        proposed=list(positions);require(proposed[1]-1 not in positions,'Fixture alternate position overlaps');proposed[1]-=1;positions=tuple(proposed)
    elif intervention=='unknown_wavelength':
        bands=None
    elif intervention=='missing_gsd':
        gsd=None
    packet=IndexPacket(bands,positions,master_count)
    if intervention=='missing':packet,gsd=None,None
    source_key=(catalog,intervention,digest({'positions':positions,'master_count':master_count,
        'physical_band_metadata':None if bands is None else bands.fingerprint,'gsd_m':gsd}))
    return BoundCondition(catalog,receipt['observation'],master_count,intervention,packet,gsd,source_key)


def install_encoders(model, seed, *, neutral=False):
    """Extend a copy only; leave existing kernels/heads/selector tensors unchanged unless neutral is explicit."""
    first=model.backbone.CNN_Branch[0];backbone=model.backbone
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for owner,encoder_name,selector_name in ((first,'condition_encoder','condition_selector'),
                (backbone,'range_encoder','range_metadata')):
            # A fixed CNN has no conditional candidate selector: do not invent one for a trained checkpoint.
            if owner is first and getattr(first,'mode','fixed')=='fixed':continue
            old=getattr(owner,encoder_name,None)
            setattr(owner,encoder_name,IndexConditionEncoder(old))
            if getattr(owner,selector_name,None) is None:
                setattr(owner,selector_name,nn.Linear(32,3 if owner is first else 2,bias=False))
            if neutral:nn.init.zeros_(getattr(owner,selector_name).weight)
    model.to(next(model.parameters()).device)
    return model


def initialize(baseline, seed):
    model=MultiscaleTrial(cnn_multiscale='joint',graph_multiscale='joint').initialize(baseline,seed).eval()
    install_encoders(model,seed+41017)
    model.backbone.adapter.cache_attention=False
    return model


@contextmanager
def selection_policy(model, cnn='joint', graph='joint'):
    """Mask contributions while executing the same candidate pools and allocated selectors."""
    require(cnn in MODES and graph in MODES, 'Unknown selection policy')
    hooks=[];first=model.backbone.CNN_Branch[0];backbone=model.backbone
    try:
        for owner,mode,content_name,metadata_name in ((first,cnn,'content_selector','condition_selector'),
                (backbone,graph,'range_content','range_metadata')):
            for name,allowed in ((content_name,mode in ('content','joint')),(metadata_name,mode in ('metadata','joint'))):
                layer=getattr(owner,name,None)
                if layer is not None and not allowed:
                    hooks.append(layer.register_forward_hook(lambda module,args,result:result*0))
        yield
    finally:
        for hook in reversed(hooks):hook.remove()


def policies(case):
    scope,mode=case.split('.')
    require(scope in SCOPES and mode in MODES,'Unknown branch-separated policy')
    return (mode if scope in ('both','cnn') else 'mean',mode if scope in ('both','graph') else 'mean')


def copy_trained_state(model, state):
    """Only index/physical metadata tensors may be absent; source tensors must match exactly."""
    current=model.state_dict();extra=set(current)-set(state)
    allowed=lambda name:name.startswith(('backbone.range_encoder.','backbone.range_metadata.',
        'backbone.CNN_Branch.0.condition_encoder.','backbone.CNN_Branch.0.condition_selector.'))
    require(set(state)<=set(current) and all(allowed(n) for n in extra),'Graft changed a source/shared tensor inventory')
    model.load_state_dict({**current,**state},strict=True)
    require(all(torch.equal(model.state_dict()[n].detach().cpu(),v.detach().cpu()) for n,v in state.items()),'Source tensor copy differs')
    return sorted(extra)
