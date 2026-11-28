from pathlib import Path

import pytest

from scripts import run_wp11_execution as cli


@pytest.mark.parametrize("policy", [None, cli.wp03.MANIFEST])
def test_legacy_run_still_dispatches_to_the_registered_runtime(monkeypatch, tmp_path, policy):
    code = {"commit": "a" * 40, "dirty": False}
    monkeypatch.setattr(cli, "code_snapshot", lambda: code)
    monkeypatch.setattr(cli, "training_container_kind", lambda: "test")
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cli.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(cli, "external_directory", Path)
    monkeypatch.setattr(cli, "read", lambda _: {"policy": policy})
    received = []
    target, name = (cli.wp03, "run") if policy else (cli, "execute")
    monkeypatch.setattr(target, name, lambda *args, **kwargs: received.append((args, kwargs)))
    mode = cli.wp03.MODE if policy else cli.MODES[0]
    out = tmp_path / "out"
    cli.main(["--expected-commit", code["commit"], "run", "--input", "manifest.json",
              "--input-sha256", "b" * 64, "--mode", mode, "--save-predictions", "--out", str(out)])
    assert received == [(({"path": "manifest.json", "sha256": "b" * 64}, code, mode, out), {"export": True})]
