"""Bind a single dataset and one fixed M1 observation to an opt-in checkpoint."""

from dataclasses import asdict

import numpy as np

from loadData.observation_sampling import BandMetadata, replay
from loadData.spatial_protocol import PartitionObservation
from model.research_config import digest
from research_runtime import tensor_hash


def model_input_sha256(observation):
    return tensor_hash(np.concatenate((observation.spectral, observation.coordinates), axis=-1))


def register(dataset, observation, bands, sampling_record):
    view = replay(observation, bands, sampling_record)
    value = dict(schema="spectral-observation-v2", dataset=dataset, source=asdict(bands),
                 source_metadata_hash=bands.fingerprint, sampling_record=sampling_record,
                 selected_metadata_hash=view.bands.fingerprint,
                 input_sha256=model_input_sha256(view.observation),
                 observation_ids_sha256=tensor_hash(view.observation.global_ids))
    return {**value, "registration_sha256": digest(value)}


def verify_registration(value, datasets):
    if not isinstance(value, dict) or set(value) != {"schema", "dataset", "source", "source_metadata_hash",
                                                    "sampling_record", "selected_metadata_hash", "registration_sha256",
                                                    "input_sha256", "observation_ids_sha256"}:
        raise ValueError("Invalid spectral registration")
    body = {key: item for key, item in value.items() if key != "registration_sha256"}
    if (value["schema"] != "spectral-observation-v2" or value["dataset"] not in datasets
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
    view = replay(observation, bands, value["sampling_record"])
    if (model_input_sha256(view.observation) != value["input_sha256"] or
            tensor_hash(view.observation.global_ids) != value["observation_ids_sha256"]):
        raise ValueError("Spectral registration observation differs")
    return view
