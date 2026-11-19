"""Planted fitting and failure controls for the support-only readout."""

import io
from time import perf_counter

import torch

from evaluation.frozen_band_region import HEADS
from transfer.support_readout import SupportReadout


def exercise(registration, device="cpu"):
    begin = perf_counter()
    results = {}
    for seed in (64, 512, 2048):
        rng = torch.Generator().manual_seed(seed)
        centers = torch.eye(6, dtype=torch.float64) * 3
        target = torch.arange(1, 7).repeat_interleave(5)
        features = centers[target - 1] + torch.randn((30, 6), generator=rng, dtype=torch.float64) * .03
        # Closest confusions 5 vs 4 and 6 vs 3, with a shifted negative control.
        rivals = torch.tensor([2, 1, 6, 5, 4, 3])[target - 1]
        scores = torch.full((30, 6), .02, dtype=torch.float64)
        scores[torch.arange(30), target - 1] = .47
        scores[torch.arange(30), rivals - 1] = .45
        rare = target >= 5
        scores[torch.arange(30)[rare], target[rare] - 1] = .44
        scores[torch.arange(30)[rare], rivals[rare] - 1] = .48
        for k in (1, 12):
            labels = torch.arange(1, 7).repeat_interleave(k)
            train = centers[labels-1] + torch.randn((6 * k, 6), generator=rng, dtype=torch.float64) * .03
            training_scores = torch.nn.functional.one_hot(labels - 1, 6).double() * .72 + .28/6
            def heads(value, scaled=False):
                return {h: (value * (1 + i) if scaled else value).to(device) for i,h in enumerate(HEADS)}
            binding = dict(scene="planted", checkpoint=str(seed), observation="missing_wavelengths", support=f"k{k}")
            current = SupportReadout.fit(binding, heads(train, True), heads(training_scores), labels.to(device), 6)
            native = heads(scores)
            predicted = current.predict(native, heads(features, True), binding)
            baseline = int((scores.argmax(1) + 1 == target).sum())
            improved = int((predicted["fused"].cpu().argmax(1) + 1 == target).sum())
            shifted = features.clone()
            shifted[rare] = centers[rivals[rare] - 1]
            shifted_result = current.predict(native, heads(shifted, True), binding)
            harmed = int((shifted_result["fused"].cpu().argmax(1) + 1 == target).sum())
            with io.BytesIO() as output:
                torch.save(current.state(), output)
                output.seek(0)
                restored = SupportReadout.from_state(torch.load(output, weights_only=True, map_location="cpu"), binding)
            resumed = restored.predict(heads(scores) if device == "cpu" else {h: scores for h in HEADS},
                                       {h: features * (i + 1) for i,h in enumerate(HEADS)}, binding)
            if not all(torch.allclose(resumed[h], predicted[h].cpu(), atol=1e-12, rtol=1e-12) for h in HEADS):
                raise ValueError("Portable support readout changed predictions")
            missing = labels == 1
            fallback = SupportReadout.fit(binding, heads(train[missing], True), heads(training_scores[missing]),
                                          labels[missing].to(device), 6)
            unchanged = fallback.predict(native, heads(features, True), binding)
            if not all(torch.equal(unchanged[h], native[h]) for h in HEADS):
                raise ValueError("Single supported class must preserve original scores")
            try:
                current.predict(native, heads(features, True), {**binding, "observation": "foreign"})
            except ValueError:
                rejected = True
            else:
                rejected = False
            if improved <= baseline or harmed >= improved or not rejected:
                raise ValueError("Planted support mechanism, shift control or input rejection failed")
            results[f"{seed}-k{k}"] = dict(support=len(labels), validation=len(target), rare=int(rare.sum()),
                                           native_correct=baseline, candidate_correct=improved,
                                           shifted_candidate_correct=harmed, portable_restore=True,
                                           single_supported_class_fallback=True, binding_rejected=rejected)
    return dict(status="passed", registration=registration, device=str(device), real_updates=0,
                learned_parameters=0, optimizer_updates=0, query_accessed=False, raw_gt_loaded=False,
                cases=results, seconds=perf_counter()-begin)
