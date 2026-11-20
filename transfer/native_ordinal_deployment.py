"""Bind a trusted delivered-band order to a native observation using its retained bytes only."""

from dataclasses import dataclass, replace

import numpy as np

from loadData.observation_sampling import BandMetadata
from model.research_config import canonical_json, digest
from transfer.spatial_coverage_protocol import metadata_view


@dataclass(frozen=True)
class DeliveredAxis:
    bank_sha256: str
    source_id: str
    band_ids: tuple

    @classmethod
    def from_bank(cls, bank):
        # This path authenticates manifest and bank identity. It does not read pixel arrays.
        bank.verify_authority(bank.pin)
        axis=bank.release.bands
        return cls(bank.pin["sha256"],axis.source_id,axis.band_ids)


def bind_retained_ordinal(actual,producer,role, delivery):
    """Use delivery ranks, or preserve the original missing-metadata fallback."""
    raw=actual.verify(producer,role)
    source=raw["source"]
    if delivery is None or source.get("spectral_metadata_mode")!="missing":
        return actual
    if (delivery.bank_sha256!=source["bank"] or delivery.source_id!=actual.view.bands.source_id
            or len(delivery.band_ids)<2):
        raise ValueError("Retained-only ordinal delivery receipt is foreign")
    ids=delivery.band_ids
    positions=tuple(source["positions"])
    if any(type(i) is not int or not 0<=i<len(ids) for i in positions):
        raise ValueError("Retained-only input exposes an unknown delivery position")
    master=BandMetadata(delivery.source_id,ids,tuple(np.linspace(0.,1.,len(ids))),
                        "normalized","index_only",(True,)*len(ids))
    view,mode=metadata_view(actual.view,master,positions,"declared","ordinal")
    source.update(spectral_metadata_mode=mode,band_order=view.bands.fingerprint,
                  independent_metadata_policy="ordinal")
    namespace=digest(dict(schema="native_cell_yx_v1",dataset=actual.dataset,source=source,
                        box=actual.lattice.box,scene_shape=actual.lattice.scene_shape,
                        factor=actual.lattice.factor,role=role,edge=raw["edge"]))
    adjusted=replace(view,observation=replace(view.observation,plan_hash=namespace))
    raw.update(namespace=namespace,bands=adjusted.bands.fingerprint)
    raw.pop("observation")
    raw["observation"]=digest(raw)
    result=replace(actual,view=adjusted,_manifest=canonical_json(raw))
    result.verify(producer,role)
    return result
