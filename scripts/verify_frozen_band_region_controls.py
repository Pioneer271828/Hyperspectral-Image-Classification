"""Audit actual bank normalization, retained metadata, and the VCA eval cache."""

import argparse
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from loadData.observation_sampling import replay, sample
from research_runtime import code_snapshot, external_directory, require_frozen_code
from scripts.run_frozen_band_region import (
    POLICY, SCENES, _freeze_bank, _honghu_context, _new_honghu, _pavia, _read, _shapes,
    _write, _xuzhou, check, pin, source_documents,
)
from transfer import development_bank as bank_runtime, frozen_band_region as runtime
from transfer.g3_training import state_hash


def _normalize_control(bank, row, name, spec, kept):
    _, _, expected, baseline = bank.materialize(row, spec)
    dropped = sorted(set(bank.release.bands.band_ids) - set(kept))
    original_decode = bank_runtime._decode

    def altered(*args, **kwargs):
        values = original_decode(*args, **kwargs).copy()
        values[..., dropped] += np.float32(137.)
        return values

    with patch.object(bank_runtime, "_decode", altered):
        _, _, poison, audit = bank.materialize(row, spec)
    check(poison.bands == expected.bands and np.array_equal(poison.observation.spectral, expected.observation.spectral)
          and audit["raw"] == baseline["raw"] and audit["normalized"] == baseline["normalized"],
          "Deleted raw columns affected the real bank materializer")
    return dict(scene=name, original_representation=baseline["representation"],
                retained_raw_equal=True, retained_normalized_equal=True)


def _cache_control(model, original, view, record, scene, options, offsets, classes, scale):
    adapter = model.backbone.adapter
    model.eval()
    _, clean, _ = runtime.infer(model, scene, view, offsets, classes, scale, **options)
    adapter.cache_attention = True
    adapter.clear_attention_cache()
    try:
        with patch.object(adapter, "_attention", wraps=adapter._attention) as calls:
            _, first, _ = runtime.infer(model, scene, view, offsets, classes, scale, **options)
            _, second, _ = runtime.infer(model, scene, view, offsets, classes, scale, **options)
            check(calls.call_count == 1 and first == second == clean, "Repeated retained view missed exact eval caching")
            shuffled = replace(view.bands, wavelengths=tuple(reversed(view.bands.wavelengths)))
            _, changed, _ = runtime.infer(model, scene, replace(view, bands=shuffled),
                                          offsets, classes, scale, **options)
            check(calls.call_count == 2 and changed["metadata"] != first["metadata"]
                  and changed["spectral"] == first["spectral"] and changed["adapter_input"] == first["adapter_input"],
                  "Same-size reordered metadata failed exact cache isolation")
            _, restored, _ = runtime.infer(model, scene, view, offsets, classes, scale, **options)
            check(calls.call_count == 3 and restored == clean, "Restored nominal metadata differs")
        try:
            replay(original.observation, replace(original.bands, wavelengths=tuple(reversed(original.bands.wavelengths))),
                   record)
        except ValueError:
            rejected = True
        else:
            raise ValueError("Frozen record accepted reordered source metadata")
    finally:
        adapter.cache_attention = False
        adapter.clear_attention_cache()
    return dict(repeated_attention_evaluations=1, changed_metadata_cache_miss=True,
                reordered_metadata_rejected_by_frozen_replay=rejected, restored_nominal_equal=True,
                shuffled_diagnostic=dict(same_spectral_hash=True,
                    probability_hash_changed={head: changed["probabilities"][head] != first["probabilities"][head]
                                              for head in first["probabilities"]}))


def verify(root, scope_pin, output, code):
    scope = _read(scope_pin)
    links, docs = source_documents(root)
    check(scope["policy"] == POLICY and scope["sources"] == links and not scope["optimizer_updates"],
          "Control scope differs from accepted released datasets")
    rows = []
    for scene in SCENES:
        bank, intake = _freeze_bank(root, scene, scope)
        case = scope["cases"][scene][0]
        model, _, identity, _, _, _ = (_pavia(docs, case, intake, bank) if scene == "paviaU" else
                                       _xuzhou(docs, case) if scene == "xuzhou" else
                                       _new_honghu(root, docs, case, _honghu_context(docs["hh_scope"])))
        model.eval()
        model.backbone.adapter.cache_attention = False
        model.backbone.adapter.clear_attention_cache()
        row = next(row for row in bank.role_rows("devval") if row["samples"])
        _, offsets = _shapes(row, bank.release.shape[1])
        recipes = runtime.band_specs(bank.release.bands)
        recipe = recipes["contiguous"]
        _, _, original, _ = bank.materialize(row, recipes["full"])
        view, record = sample(original.observation, original.bands, recipe)
        norm = _normalize_control(bank, row, scene, recipe, record["selected_positions"])
        options = dict(role="devval", wavelength_unit=scene != "xuzhou")
        cache = _cache_control(model, original, view, record, scene, options, offsets, bank.release.classes,
                               50 if scene == "paviaU" else 100)
        check(state_hash(model) == identity, "Frozen model state changed during engineering controls")
        rows.append(dict(scene=scene, case=case, normalization=norm, cache=cache, physical_forwards=5))
    require_frozen_code(code)
    return _write(output, dict(status="passed", scope=scope_pin["sha256"], control_commit=code["commit"],
                              raw_gt_loaded=False, query_accessed=False, optimizer_updates=0, rows=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "scope", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--scope-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    root, output = external_directory(args.root), external_directory(args.output)
    scope = pin(args.scope, args.scope_sha256)
    check(output.is_relative_to(root) and Path(scope["path"]).is_relative_to(root), "Unauthorized control path")
    code = code_snapshot()
    require_frozen_code(code)
    check(code["commit"] == args.expected_commit, "Unexpected control checkout")
    torch.set_num_threads(1)
    print(verify(root, scope, output, code))


if __name__ == "__main__":
    main()
