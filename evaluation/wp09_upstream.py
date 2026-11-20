"""Authenticate specific optional upstream files and run their original Python callables."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import torch

from model.research_config import digest
from research_runtime import hash_file


TENT = "e9e926a668d85244c66a6d5c006efbd2b82e83e8"
OPENOOD = "3c35632ee91b54b09d1f085d04f94744cece7d0b"
OPENOOD_FILES = ("openood/postprocessors/base_postprocessor.py",
                 "openood/postprocessors/ebo_postprocessor.py", "openood/evaluators/metrics.py")


class _DisabledComm(ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise RuntimeError(f"Distributed OpenOOD operation is disabled: {name}")


def authenticate(root, revision, files):
    root = Path(root).resolve(strict=True)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])
    if (Path(git("rev-parse", "--show-toplevel").decode().strip()) != root
            or git("rev-parse", "HEAD").decode().strip() != revision or git("status", "--porcelain")):
        raise ValueError("Clean, pinned upstream checkout required")
    pins = {}
    for file in ("LICENSE", *files):
        path = root / file
        if path.is_symlink() or path.read_bytes() != git("show", f"HEAD:{file}"):
            raise ValueError("Upstream file differs from the pinned commit")
        pins[file] = hash_file(path)
    return root, dict(commit=revision, files=pins)


def module(root, relative, name):
    spec = spec_from_file_location(name, root / relative)
    value = module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def tent_reference(root):
    root, evidence = authenticate(root, TENT, ("tent.py",))
    name = "_wp09_tent_" + TENT[:12]
    # TorchScript resolves module globals while importing its decorated function.
    value = module(root, "tent.py", name)
    return value, dict(evidence, torch=str(torch.__version__))


class OpenOODReference:
    """Run the upstream postprocess methods without importing its unrelated model registry."""

    def __init__(self, root):
        import sklearn
        import tqdm
        root, evidence = authenticate(root, OPENOOD, OPENOOD_FILES)
        packages = ("openood", "openood.utils", "openood.postprocessors", "openood.evaluators")
        comm = "openood.utils.comm"
        postprocessors = ("openood.postprocessors.base_postprocessor",
                          "openood.postprocessors.ebo_postprocessor", "openood.evaluators.metrics")
        names = (*packages, comm, *postprocessors)
        before = {name: sys.modules.get(name) for name in names}
        try:
            for name in packages:
                namespace = ModuleType(name)
                namespace.__path__ = []
                sys.modules[name] = namespace
            sys.modules[comm] = _DisabledComm(comm)
            loaded = [module(root, path, name) for path, name in zip(OPENOOD_FILES, postprocessors, strict=True)]
        finally:
            for name, previous in before.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous
        self.base, self.energy, self.metrics_module = loaded
        self.evidence = dict(evidence, distributed_comm="disabled_fail_on_use_v1",
                             sklearn=sklearn.__version__, tqdm=tqdm.__version__, torch=str(torch.__version__))
        self.identity = digest(self.evidence)

    def compare_logits(self, scorer, values, *, input_kind, column_ids):
        baseline, known = scorer.score(values, input_kind=input_kind)
        spec = scorer.as_dict()
        ids = [row["id"] for row in spec["known"]]
        if input_kind != "raw_logits" or list(column_ids) != ids:
            raise ValueError("Exact frozen raw logits and semantic column order required")
        x = torch.as_tensor(values.copy(), dtype=torch.float64)
        temperature = spec["temperature"]
        if spec["method"] == "msp":
            # The source postprocessor is unit-temperature. Pre-scale logits from the frozen contract.
            scaled = (x-x.amax(dim=1, keepdim=True))/temperature
            pred, conf = self.base.BasePostprocessor(None).postprocess(torch.nn.Identity(), scaled)
            score = 1-conf.numpy()
        elif spec["method"] == "energy":
            config = SimpleNamespace(postprocessor=SimpleNamespace(
                postprocessor_args=SimpleNamespace(temperature=temperature), postprocessor_sweep={}))
            pred, conf = self.energy.EBOPostprocessor(config).postprocess(torch.nn.Identity(), x)
            score = -conf.numpy()
        else:
            raise ValueError("No equivalent OpenOOD squared-distance mean prototype")
        import numpy as np
        pred = np.asarray(ids)[pred.numpy()]
        if (not np.isfinite(score).all() or not np.allclose(score, baseline, rtol=1e-5, atol=1e-6)
                or not np.array_equal(pred, known)):
            raise ValueError("OpenOOD output differs from the frozen score semantics")
        return score, pred
