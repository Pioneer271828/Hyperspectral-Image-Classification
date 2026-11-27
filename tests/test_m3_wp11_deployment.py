"""Regression checks for the native partition route and deferred state guard."""

import copy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from inference.native_deployment import Request, RouteFailure, Runtime
from tests.test_tile_cascade import setup
from transfer.g3_training import state_hash


def fixture():
    full, light, rows = setup("group_norm")
    requests = [Request(r["id"], r["dataset"], r["id"], "synthetic_parent", "original_full",
                        r["x"], r["wavelengths"], None, None, r["scale"], 1., True) for r in rows]
    return full, light, requests


def test_deferred_wrong_state_and_mode_retry_only_publish_an_admitted_model():
    full, light, requests = fixture()
    foreign = copy.deepcopy(full)
    with torch.no_grad():
        next(foreign.parameters()).add_(1.)
    training = copy.deepcopy(full).train()
    attempts = iter((foreign, training, full))
    model_states = {"full":state_hash(full), "light":state_hash(light)}
    runtime = Runtime(device="cpu", light=light, full_loader=lambda: next(attempts),
                      identity=model_states, partition="grid")
    before = runtime.version()
    skipped = runtime.run(requests, "light")
    assert skipped["counts"]["graph_builds"] == 0
    assert runtime.full is None and runtime.version() == before
    for expected in ("pinned CPU", "eval modules"):
        with pytest.raises(RouteFailure) as caught:
            runtime.run(requests, "full")
        assert expected in str(caught.value.__cause__)
        assert caught.value.stats["graph_builds"] == 0
        assert runtime.full is None and runtime.version() == before
        assert runtime.identity == model_states
    first = runtime.run(requests, "full")
    again = runtime.run(requests, "full")
    assert first["counts"]["graph_builds"] == again["counts"]["graph_builds"] == len(requests)
    assert runtime.full is full and runtime.identity == model_states
    assert {p.device.type for p in full.parameters()} == {"cpu"}
    for a, b in zip(first["predictions"], again["predictions"], strict=True):
        for x, y in zip(a, b, strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    full.train()
    with pytest.raises(ValueError, match="left evaluation mode"):
        runtime.run(requests, "full")


def test_partition_route_falls_back_to_selected_grid_and_rejects_invalid_metadata():
    full, light, requests = fixture()
    runtime = Runtime(device="cpu", full=full, light=light, partition="grid")
    reference = runtime.run(requests, "full")
    preferred = {r.key for r in requests}
    head = light.heads[requests[0].scene]
    original = head.forward
    calls = iter(range(len(requests)))

    def nonfinite_once(x):
        value = original(x)
        return torch.full_like(value, float("nan")) if next(calls) == 0 else value

    with patch.object(head, "forward", nonfinite_once):
        actual = runtime.run(requests, "random", random_fast=preferred)
    assert actual["counts"]["light_failures"] == 1
    assert actual["counts"]["graph_builds"] == actual["counts"]["full_calls"] == 1
    assert [row["route"] for row in actual["routes"]] == ["full", "light"]
    for x,y in zip(actual["predictions"][0], reference["predictions"][0], strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    with pytest.raises(ValueError, match="known wavelengths"):
        runtime.run([replace(requests[0], known=False)], "full")
