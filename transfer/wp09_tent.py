"""Opt-in Tent entropy inside the existing WP09 science and persistence policies."""

from functools import partial

import torch

from evaluate_tta import _Session
from evaluation.wp09_upstream import tent_reference
from model.research_config import digest
from transfer.online_tta_run import execute


class TentEntropySession(_Session):
    def __init__(self, source, *, entropy, **settings):
        super().__init__(source, **settings)
        self.entropy = entropy

    def adapt(self, inputs):
        if self.optimizer is None:
            return super().adapt(inputs)
        x, waves = inputs
        kwargs = dict(wavelengths=waves, superpixel_scale=self.scale, wavelength_unit="nm")
        self.optimizer.zero_grad(set_to_none=True)
        if self.adapter is None:
            z = self.model(self.task, x, return_logits=True, **kwargs)[0]
        else:
            fused, graph, cnn = self.model.backbone.forward_features(x, dataset_name=self.task, **kwargs)
            z = self.model.predict_features(self.task, fused + self.adapter(fused), graph, cnn, return_logits=True)[0]
        p = z.softmax(dim=1)
        if not torch.isfinite(z).all():
            raise ValueError("Nonfinite Tent logits")
        # Tent assumes unclipped entropy. Preserve the original WP09 tail behavior.
        loss = (-(p*p.clamp_min(1e-8).log()).sum(dim=1) if (p < 1e-8).any() else self.entropy(z)).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite TTA entropy")
        loss.backward()
        gradients = [p.grad for group in self.optimizer.param_groups for p in group["params"] if p.grad is not None]
        if not gradients or any(not torch.isfinite(grad).all() for grad in gradients):
            raise ValueError("Invalid TTA parameter gradients")
        self.optimizer.step()
        return float(loss.detach())


def execute_tent(registration, out, root, *, to_frame=None, admission=None):
    upstream, evidence = tent_reference(root)
    provider = digest(dict(evidence, component="softmax_entropy", native_clamped_floor=1e-8))
    session = partial(TentEntropySession, entropy=upstream.softmax_entropy)
    return execute(registration, out, to_frame=to_frame, admission=admission, session_factory=session,
                   provider=provider)
