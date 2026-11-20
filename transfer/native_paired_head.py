"""Opt-in paired 64-channel head adaptation using the existing native optimizer."""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import torch

from evaluation.frozen_training_stages import require
from model.region_pixel_alignment import broadcast_regions
from model.research_config import digest
from research_runtime import tensor_hash
from transfer import native_paired_head_fixture as fixture
from transfer.g3_training import state_hash
from transfer.native_production_model import Switches
from transfer.native_production_provider import ProductionProvider
from transfer.native_released_points import MODE
from transfer.native_scale_observation import forward
from transfer.spatial_coverage_protocol import bank_view, retained_positions

SCHEMA="wp05_wp06_native_paired_head_v1"
ARMS={
    "xuzhou":("full_support","rotating_support"),
    "honghu":("fusion","pooled"),
    "fallback":("legacy","ordinal","permuted_ordinal"),
}
BUDGET={"xuzhou":128,"honghu":128,"fallback":48}
EXPOSURE="first_min_released_point_fused_nll"
HEAD_PREFIX="heads."
# Graph/CNN are measured throughout. Their frozen predictions contribute no trainable gradients.
OBJECTIVE=dict(mode=MODE, optimize="only_fused_linear", selection="first_min_fused_calibration_nll",
               frozen_diagnostic_heads=["graph","cnn"], weights=[1,1,1])


def selected_axis_policy(policy, metadata):
    if metadata in ("declared","misaligned"):
        return metadata
    require(metadata=="missing" and policy in ARMS["fallback"],"Unsupported deployment metadata state")
    return "missing" if policy=="legacy" else policy


def state_inventory(model):
    return {key:tensor_hash(value) for key,value in model.state_dict().items()}


def freeze_for_head(model, scene):
    head=model.heads[scene]
    require(head.in_features==64 and isinstance(head,torch.nn.Linear),"Paired head requires a linear 64-channel input")
    for param in model.parameters():
        param.requires_grad_(False)
    for param in head.parameters():
        param.requires_grad_(True)
    return model


def pooled_forward(model, actual, producer, role, execution, timing):
    backbone=model.backbone
    stage=backbone.forward_stages
    pooled=[]
    head=model.heads[actual.dataset]

    def capture(cnn,regions,q,a,*args,**kwargs):
        require(not pooled and regions.shape[1]==head.in_features,"Ambiguous pooled head coordinates")
        pooled.append(broadcast_regions(q,regions))
        return stage(cnn,regions,q,a,*args,**kwargs)

    def replace(_head,arguments):
        value,=arguments
        require(len(pooled)==1 and pooled[0].shape==value.shape and pooled[0].device==value.device
                and pooled[0].dtype==value.dtype,"Pooled and fused head inputs have different role, shape or alignment")
        return (pooled[0],)

    handle=head.register_forward_pre_hook(replace)
    try:
        with patch.object(backbone,"forward_stages",new=capture):
            return forward(model,actual,producer,role,timed=timing,execution=execution)
    finally:
        handle.remove()


class PairedHeadProvider(ProductionProvider):
    """Freeze the original encoder and supervise the same two-parameter linear head."""

    def __init__(self,root,bank,roles,*,kind,policy,seed,producer,source):
        require(bank.release.profile=="synthetic" and kind in ARMS and policy in ARMS[kind] and seed in fixture.SEEDS,
                "A published synthetic paired head is required for optimizer admission")
        super().__init__(root,bank,roles,axis=bank.release.bands.band_ids,config=Switches(),
                         source=source,seed=seed,budget=BUDGET[kind],producer=producer)
        self.kind,self.policy=kind,policy

    def model_factory(self,source,arm,seed):
        original=super().model_factory(source,arm,seed)
        return freeze_for_head(original,self.scene)

    def key(self,role,event):
        return super().key(role,event)+"/"+event["metadata"]

    def make_from_bank(self,bank,role,event):
        mode=selected_axis_policy(self.policy if self.kind=="fallback" else "legacy",event["metadata"])
        row=bank.role_row(role,event["tile"])
        actual,labels,targets=bank_view(bank,row,self.producer,event["band"],factor=1,metadata=mode)
        expected=[bank.release.bands.band_ids[i] for i in retained_positions(bank.release.bands,event["band"])]
        require(list(actual.view.bands.band_ids)==expected==actual.receipt["source"]["band_ids"],
                "Paired head attempted to restore deleted columns")
        return actual,labels,targets

    def make_observation(self,role,event):
        return self.make_from_bank(self.bank,role,event)

    def confirmation(self,checkpoint,tile,band,metadata,*,bank=None,allowed=None):
        require(checkpoint and tile in (self.roles["evaluation"] if allowed is None else allowed),
                "Paired confirmation cannot open support or calibration as evaluation")
        return self.make_from_bank(self.bank if bank is None else bank,"devval",
                                  dict(tile=tile,band=band,metadata=metadata,factor=1))

    def registered_forward(self,model,actual,role,execution):
        model.backbone.eval()
        timing={}
        if self.policy=="pooled":
            out,graph=pooled_forward(model,actual,self.producer,role,execution,timing)
        else:
            out,graph=forward(model,actual,self.producer,role,timed=timing,execution=execution)
        return out,graph,timing

    def clock(self):
        tiles=self.roles["support"]
        training=[]
        rng=np.random.default_rng(self.seed)
        indexed=("full","central_gap","matched_dispersed")
        while len(training)<self.budget:
            epoch=len(training)//len(tiles)
            for tile in rng.permutation(tiles).tolist():
                rank=tiles.index(tile)
                band="full"
                metadata="declared"
                if self.policy=="rotating_support" or self.kind=="fallback":
                    band=indexed[(epoch+rank)%3]
                if self.kind=="fallback" and (epoch+rank//2)%2:
                    metadata="missing"
                training.append(dict(step=len(training)+1,tile=tile,band=band,metadata=metadata,
                                     factor=1,condition="paired_native_head"))
        training=training[:self.budget]
        conditions=[("full","declared")] if self.kind!="fallback" else [
            ("full","declared"),("central_gap","missing"),("matched_dispersed","missing")]
        calibration=[dict(tile=tile,band=band,metadata=metadata,factor=1,condition="paired_native_head")
                     for tile in self.roles["calibration"] for band,metadata in conditions]
        return training,calibration

    def validate_history(self,spec,packet):
        super().validate_history(spec,packet)
        expected=spec["initialization"]["inventory"]
        for key in ("model","selected"):
            values=packet["model"] if key=="model" else packet["selected"]["state"]
            require(set(values)==set(expected),"Model tensor inventory differs")
            require(all(tensor_hash(value)==expected[name] for name,value in values.items()
                        if not name.startswith(HEAD_PREFIX)),"An optimizer mutated the frozen trunk or diagnostics")

    def contract(self,classes,seed,arm,factor,producer,*,numeric=None):
        require((classes,seed,arm,factor,producer)==(self.classes,self.seed,"native",1,self.producer),
                "Paired head identity differs from its contract")
        self.bank.verify_authority(self.bank.pin)
        training,calibration=self.clock()
        if self._roster is None:
            self._roster={}
            for role,events in (("devtrain",training),("devval",calibration)):
                for event in events:
                    key=self.key(role,event)
                    if key not in self._roster:
                        self._roster[key]=self.evidence(*self.materialize(role,event,producer))
        base=self.baseline(seed)
        model=self.model_factory(base,arm,seed+7321)
        trainable=[name for name,value in model.named_parameters() if value.requires_grad]
        require(trainable==[f"heads.{self.scene}.weight",f"heads.{self.scene}.bias"],
                "Unexpected trainable tensor in a paired head")
        original=state_inventory(model)
        return dict(format=SCHEMA,producer=producer,classes=classes,seed=seed,arm=arm,factor=factor,kind=self.kind,
                    policy=self.policy,supervision=MODE,objective=OBJECTIVE,training=training,validation=calibration,
                    steps=self.budget,prefix=41 if self.budget==128 else 17,
                    events=list(range(0,self.budget+1,self.budget//4)),learning_rate=.006,
                    module_seed=seed+7321,rng_seed=seed+15000,inputs=deepcopy(self._roster),
                    initialization=dict(state=state_hash(model),inventory=original,fresh={},source=self.source["state"]),
                    trainable=trainable,parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    recognition_dim=64,classes_1_based=list(range(1,classes+1)),bank=self.bank.pin,
                    roles=digest(self.roles),selection=EXPOSURE,numeric=numeric,
                    real_updates=0,raw_gt_loaded=False,query_accessed=False,frozen_stage_protocol=SCHEMA)
