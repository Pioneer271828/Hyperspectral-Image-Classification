"""Explicit optical metadata admission for an already-authorized observation Bank."""

from dataclasses import dataclass, replace
import json
import math
from urllib.parse import urlsplit

from evaluation.frozen_training_stages import pinned, require
from loadData.observation_sampling import BandMetadata, ObservationView

SCHEMA = "wp02_optical_sidecar_v1"
APPROVAL = "wp02_optical_authority_v1"
UNITS = {"nm": 1., "um": 1000.}


def source_record(root, source):
    require(set(source) == {"version", "publisher", "uri", "terms", "evidence"}
            and all(isinstance(source[key], str) and source[key].strip()
                    for key in ("version", "publisher", "uri", "terms")),
            "Publisher, release version, provenance URI and authorized terms required")
    uri = urlsplit(source["uri"])
    require(uri.scheme == "https" and bool(uri.netloc) and uri.username is None and uri.password is None,
            "Require a public provenance URI")
    pinned(root, source["evidence"])


def detector_axis(values, *, allow_empty=False):
    require(isinstance(values, list) and (allow_empty or bool(values))
            and all(type(value) is int and value >= 0 for value in values)
            and len(values) == len(set(values)), "Require ordered unique detector identifiers")
    return tuple(values)


def vector(values, count, unit):
    require(unit in UNITS and isinstance(values, list) and len(values) == count
            and all(type(item) in (float, int) and math.isfinite(item) and item > 0 for item in values),
            "Optical values need an explicit physical unit and finite positive values")
    return tuple(item * UNITS[unit] for item in values)


def spectral_response(curves, centers, unit):
    require(isinstance(curves, list) and len(curves) == len(centers),
            "Response curves must follow the delivered band order")
    values = []
    for curve, center in zip(curves, centers, strict=True):
        require(set(curve) == {"wavelengths", "response"} and isinstance(curve["wavelengths"], list)
                and len(curve["wavelengths"]) >= 3, "Need a sampled response curve")
        x = vector(curve["wavelengths"], len(curve["wavelengths"]), unit)
        y = curve["response"]
        require(list(x) == sorted(set(x)) and x[0] < center < x[-1]
                and isinstance(y, list) and len(y) == len(x)
                and all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in y)
                and max(y) > 0, "Invalid normalized spectral response samples")
        values.append(tuple(zip(x, y, strict=True)))
    return tuple(values)


@dataclass(frozen=True)
class OpticalAdmission:
    bank_sha256: str
    approximate: BandMetadata
    measured: BandMetadata
    detector_ids: tuple
    removed_ids: tuple
    response: tuple | None

    def view(self, actual, producer, role):
        """Keep the historical receipt intact and issue a separate optical metadata view."""
        raw = actual.verify(producer, role)
        old = actual.view
        require(raw["source"]["bank"] == self.bank_sha256, "Optical sidecar is bound to a different Bank")
        lookup = {key: i for i, key in enumerate(self.approximate.band_ids)}
        require(set(old.bands.band_ids).issubset(lookup), "Observation contains a foreign band")
        positions = tuple(lookup[key] for key in old.bands.band_ids)
        require(tuple(sorted(set(positions))) == positions and old.bands == self.approximate.subset(positions),
                "Observation is not an ordered view of the approved base axis")
        return ObservationView(old.observation, self.measured.subset(positions))


def admit(root, bank, sidecar_pin, authority_pin):
    """Verify separate holder approval and bind metadata to exact delivered bytes."""
    sidecar = json.loads(pinned(root, sidecar_pin).read_bytes())
    approval = json.loads(pinned(root, authority_pin).read_bytes())
    bank.verify_authority(bank.pin)
    release = bank.release
    base = release.bands
    expected = dict(schema=APPROVAL, sidecar_sha256=sidecar_pin["sha256"], bank_sha256=bank.pin["sha256"],
                    scene=release.scene, permission="read_only_optical_metadata",
                    mode="synthetic" if release.profile == "synthetic" else "external_real")
    require(approval == expected, "Independent holder approval does not bind this Bank and optical sidecar")
    fields = {"schema", "scene", "sensor", "bank_sha256", "base_band_sha256", "source", "columns",
              "detector_axis", "removed_detector_ids", "unit", "fwhm", "srf"}
    require(set(sidecar) == fields and sidecar["schema"] == SCHEMA
            and sidecar["scene"] == release.scene and sidecar["bank_sha256"] == bank.pin["sha256"]
            and sidecar["base_band_sha256"] == base.fingerprint and isinstance(sidecar["sensor"], str)
            and bool(sidecar["sensor"].strip()), "Optical sidecar does not bind the scene, sensor and exact base axis")
    source_record(root, sidecar["source"])
    axis = detector_axis(sidecar["detector_axis"])
    removed = detector_axis(sidecar["removed_detector_ids"], allow_empty=True)
    columns = sidecar["columns"]
    require(isinstance(columns, list) and len(columns) == len(base.band_ids)
            and all(set(row) == {"delivered_id", "detector_id", "center"} for row in columns)
            and detector_axis([row["delivered_id"] for row in columns]) == base.band_ids,
            "Optical table must match the exact ordered delivered IDs")
    detectors = detector_axis([row["detector_id"] for row in columns])
    require(set(axis) == set(detectors) | set(removed) and not set(detectors) & set(removed)
            and tuple(i for i in axis if i in set(removed)) == removed,
            "Original detector inventory and explicit removals do not reconcile")
    unit = sidecar["unit"]
    centers = vector([row["center"] for row in columns], len(columns), unit)
    widths = sidecar["fwhm"]
    if widths is not None:
        widths = vector(widths, len(centers), unit)
    srf = sidecar["srf"]
    if srf is not None:
        srf = spectral_response(srf, centers, unit)
    identity = f'{base.source_id}:optical:{sidecar_pin["sha256"]}:{authority_pin["sha256"]}'
    bands = replace(base, source_id=identity, wavelengths=centers, wavelength_quality="verified", wavelength_unit="nm",
                    fwhm=widths, fwhm_unit=None if widths is None else "nm")
    return OpticalAdmission(bank.pin["sha256"], base, bands, detectors, removed, srf)
