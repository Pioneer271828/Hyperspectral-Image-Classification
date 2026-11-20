"""Replay a representative released tile from each approved frozen archive."""

from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from evaluation.representation_localization import taps
from research_runtime import hash_file, require_frozen_code, tensor_hash
from scripts import run_frozen_band_region as frozen
from transfer import band_scale_real as archives
from transfer.band_scale import evidence, measure, released_view
from transfer.development_blocks import array_hash
from transfer.g3_training import state_hash

EXPECTED_ARCHIVES = {
    "paviaU-64": "ca25100f0d3283db5d06ac5c16fbf87891f18aa6254b9d20ee169e08a39a877b",
    "xuzhou": "9501787b0e27fad18291059531c06c0d201f04c4af1a94ec0967ef0e923fa8f6",
    "whu-honghu": "c135dc35688627762d40cf16b32ec44669a6d2d9e9f83537854f4119d7775c1b",
}
SCHEMA = "wp05_wp06_native_frozen_compatibility_v1"


def archived_receipts(root, pins):
    found = {}
    for path, sha in pins:
        path = Path(path).resolve(strict=True)
        name = path.stem
        frozen.check(root in path.parents and EXPECTED_ARCHIVES.get(name) == sha and name not in found,
                     "Frozen replay needs the approved in-root archive and its exact digest")
        value = frozen._read(frozen.pin(path, sha))
        frozen.check(value["case"] == name and value["role"] == "devval" and value["real_updates"] == 0
                     and value["query_accessed"] is value["raw_gt_loaded"] is False, "Foreign historical receipt")
        found[name] = value
    frozen.check(found.keys() == EXPECTED_ARCHIVES.keys(), "Frozen replay requires the three approved archives")
    return found


def frozen_model(root, name, bank, intake, documents):
    scene, seed, arm = archives.CASES[name]
    if scene == "paviaU":
        archive = frozen._read(archives.source_pins(root)[f"pavia_{seed}"])
        return archives.open_pavia(archive, bank, intake, seed)
    if scene == "xuzhou":
        model, seed, identity, step, _, checkpoint = frozen._xuzhou(documents, arm)
    else:
        context = frozen._honghu_context(documents["hh_scope"])
        model, seed, identity, step, _, checkpoint = frozen._new_honghu(root, documents, arm, context)
    return model, seed, identity, step, checkpoint


def representative(bank, truth):
    for tile in bank.role_rows("devval"):
        points = [j for j, (_, label) in enumerate(tile["samples"]) if label in truth]
        if points:
            # The archive counts only tiles with a released focus label.
            return tile, 0, points
    raise ValueError("Approved archive has no compatible published validation tile")


def slot(model, bank, tile, index, points, band, archived, code):
    clock = perf_counter()
    actual, _, targets = released_view(bank, tile, code["commit"], band, 1)
    data = evidence(actual, targets)
    materialize = perf_counter()-clock
    selected, = [row for row in archived["rows"] if row["tile"] == index and
                 row["event"] == dict(band=band, factor=1, policy="native")]
    old = selected["evidence"]
    # The observation digest binds the producer commit. Compare its constituent data instead.
    members = set(old) - {"view"}
    frozen.check(all(old[key] == data[key] for key in members), "Released input bytes, labels or original bands differ")
    baseline = measure(model, actual, targets, code["commit"])
    with taps(model, actual.dataset) as features:
        probed = measure(model, actual, targets, code["commit"])
    frozen.check(baseline["scores"] == probed["scores"] and baseline["graph"] == probed["graph"],
                 "WP06 capture changed released predictions or native Q/A")
    probabilities = {head: [rows[j] for j in points] for head, rows in probed["scores"].items()}
    truth = [targets.points[j][1] for j in points]
    old_probability = selected["probabilities"]
    errors = {head: float(np.max(np.abs(np.asarray(old_probability[head])-rows)))
              for head, rows in probabilities.items()}
    exact_probabilities = all(array_hash(np.asarray(value, dtype=np.float32)) ==
                              array_hash(np.asarray(old_probability[head], dtype=np.float32))
                              for head, value in probabilities.items())
    frozen.check(exact_probabilities and truth == selected["labels"]
                 and selected["computation"]["graph"] == probed["graph"],
                 "Historical native probabilities or Q/A differ")
    return dict(band=band, factor=1, role="devval", tile_ordinal=index, labels=truth, probabilities=probabilities,
                graph=probed["graph"], archive_observation=old["view"], input_observation=data["view"],
                same_input_members=sorted(members), max_probability_error=errors, probabilities_fp32_equal=True,
                graph_equal=True, untouched_by_probes=True, input_spectral_sha256=data["spectral"],
                original_band_ids=data["band_ids"], fit_statistics=data["retained_stats"],
                probes={name: dict(shape=list(value.shape), sha256=tensor_hash(value)) for name, value in features.items()},
                materialize_seconds=materialize, native_forward_seconds=baseline["forward_seconds"],
                native_graph_seconds=baseline["build_seconds"], probed_forward_seconds=probed["forward_seconds"])


@torch.no_grad()
def replay(root, out, code, pins):
    archived = archived_receipts(root, pins)
    parents, documents = frozen.source_documents(root)
    results = {}
    for name, previous in archived.items():
        scene = previous["scene"]
        bank, intake = frozen._freeze_bank(root, scene, {"sources": parents})
        model, seed, identity, step, checkpoint = frozen_model(root, name, bank, intake, documents)
        frozen.check(identity == previous["model"] and step == previous["selected_step"]
                     and seed == previous["model_seed"] and checkpoint == previous["checkpoint"]
                     and bank.pin["sha256"] == previous["preflight_population"],
                     "Frozen archive checkpoint, model, seed or Bank differs")
        model.eval()
        model.backbone.adapter.cache_attention = False
        model.backbone.adapter.clear_attention_cache()
        tile, index, points = representative(bank, archives.FOCUS[scene])
        rows = [slot(model, bank, tile, index, points, band, previous, code)
                for band in ("full", "central_gap")]
        frozen.check(state_hash(model) == identity, "Frozen released model changed")
        results[scene] = dict(archive_sha256=EXPECTED_ARCHIVES[name], bank=bank.pin["sha256"], model=identity,
                              checkpoint=checkpoint, selection=step, model_seed=seed, rows=rows)
    require_frozen_code(code)
    out.mkdir(exist_ok=True)
    return frozen._write(out/"replay.json", dict(schema=SCHEMA, producer=code["commit"], scenes=results,
                        scope="representative_released_development_compatibility", real_updates=0,
                        raw_gt_loaded=False, query_accessed=False, formal_research_claim_allowed=False,
                        diagnostics_sha256=hash_file(Path(taps.__code__.co_filename))))
