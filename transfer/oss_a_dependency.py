"""Opt-in DepGraph oracle for the existing G4 stage-1 tensor axes."""

from copy import deepcopy

import torch
from torch import nn

from compression.g4_cnn_stage1 import FIRST, SECOND, _mapped_geometry


class _StagePair(nn.Module):
    """Trace all active domain BN leaves using the stock block implementations."""

    def __init__(self, model):
        super().__init__()
        self.first = model.get_submodule(FIRST)
        self.second = model.get_submodule(SECOND)
        normalizer = self.second.norm
        self.domains = (tuple(normalizer.norm.norms) if normalizer.norm_type == "domain_bn"
                        else (None,))

    def forward(self, image):
        return tuple(self.second(self.first(image, domain), domain) for domain in self.domains)


def _affected_axes(pair, dependency, indices):
    names = {module:name for name,module in pair.named_modules() if name}
    axes = set()
    for item, selected in dependency:
        module = item.target.module
        if module not in names:
            # DepGraph virtual autograd operators carry indices but no state.
            if any(True for _ in module.parameters(recurse=False)):
                raise ValueError("Unmapped learned dependency")
            continue
        name = names[module].replace("first",FIRST,1) if names[module].startswith("first.") else (
            names[module].replace("second",SECOND,1))
        if sorted(selected) != sorted(indices):
            raise ValueError("Dependency indices differ from the pinned stage-1 channels")
        if isinstance(module, nn.BatchNorm2d):
            axes.update((name+"."+key,0) for key in ("weight","bias","running_mean","running_var"))
        elif name in (FIRST+".point_conv",FIRST+".depth_conv",SECOND+".point_conv"):
            axis = int(name == SECOND+".point_conv")
            axes.add((name+".weight",axis))
            if module.bias is not None:
                axes.add((name+".bias",0))
        else:
            raise ValueError(f"Unexpected learned dependency: {name}")
    return axes


def compare_stage1(model, image, channels, *, depgraph):
    """Reconcile upstream dependencies with the existing compact template on a copy.

    The caller supplies the pinned upstream module. No production pruning receipt
    is produced, and the candidate returned is restricted to engineering checks.
    """
    compact, retained, expected_axes = _mapped_geometry(model, channels)
    pair = _StagePair(deepcopy(model)).eval()
    with torch.enable_grad():
        graph = depgraph.DependencyGraph().build_dependency(pair, example_inputs=image)
        group = graph.get_pruning_group(pair.first.point_conv, depgraph.prune_conv_out_channels, idxs=channels)
    if not graph.check_pruning_group(group):
        raise ValueError("DepGraph rejected the bounded stage-1 removal")
    actual_axes = _affected_axes(pair,group,channels)
    if actual_axes != set(expected_axes):
        raise ValueError("DepGraph and project stage-1 tensor axes differ")
    group.prune()
    mapped = compact.state_dict()
    for block,path in ((pair.first,FIRST),(pair.second,SECOND)):
        for name,value in block.state_dict().items():
            torch.testing.assert_close(value,mapped[f"{path}.{name}"],rtol=0,atol=0)
    return compact, dict(removed=list(channels),retained=retained,axes=sorted(actual_axes),
                         tensor_values_exact=True,candidate_accepted=False)
