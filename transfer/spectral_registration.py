"""Bind opt-in metadata and the actual M1 observations reused on reload."""

from dataclasses import asdict

import numpy as np

from loadData.observation_sampling import BandMetadata, replay
from loadData.spatial_protocol import PartitionObservation
from model.research_config import digest


def register(dataset, observation, bands, sampling_record):
    view = replay(observation, bands, sampling_record)
    value = dict(schema="spectral-observation-v1", dataset=dataset, source=asdict(bands),
                 source_metadata_hash=bands.fingerprint, sampling_record=sampling_record,
                 selected_metadata_hash=view.bands.fingerprint)
    return {**value, "registration_sha256": digest(value)}


def verify_registration(value, datasets):
    if not isinstance(value, dict) or set(value) != {"schema", "dataset", "source", "source_metadata_hash",
                                                    "sampling_record", "selected_metadata_hash", "registration_sha256"}:
        raise ValueError("Invalid spectral registration")
    body = {key: item for key, item in value.items() if key != "registration_sha256"}
    if (value["schema"] != "spectral-observation-v1" or value["dataset"] not in datasets
            or value["registration_sha256"] != digest(body)):
        raise ValueError("Invalid spectral registration identity")
    bands = BandMetadata(**value["source"])
    record = value["sampling_record"]
    if (bands.fingerprint != value["source_metadata_hash"] or record["source_metadata_hash"] != bands.fingerprint
            or record["selected_metadata_hash"] != value["selected_metadata_hash"]):
        raise ValueError("Spectral registration source or selected mapping differs")
    stub = PartitionObservation(record["partition"], np.zeros((1, 1, len(bands.band_ids)), dtype=np.float32),
                                np.zeros((1, 1, 2), dtype=np.float32), np.zeros((1, 1), dtype=np.int64),
                                record["plan_hash"])
    replay(stub, bands, record)
    return bands


def restore(value, dataset, observation, bands):
    verify_registration(value, (dataset,))
    return replay(observation, bands, value["sampling_record"])
