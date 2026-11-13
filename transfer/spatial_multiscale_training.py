"""Frozen synthetic-only WP07 exposure and checkpoint continuation."""

from copy import deepcopy
import io
from itertools import product

import torch

from joint_training import _restore_rng, _rng_state
from model.research_config import digest
from model.task_losses import detached_loss_values, probability_loss_components
from research_runtime import tensor_hash
from transfer import development_bank_training as parent
from transfer.development_bank import _atomic_new
from transfer.development_bank_consumers import heads
from transfer.g3_training import state_hash
from transfer.spatial_multiscale import ARMS, forward, initialize
from transfer.spatial_observation import MODES, materialize, transform_spec
from transfer.xuzhou_multiscale_audit import _exact


FORMAT = "wp07-spatial-conditions-synthetic-v1"
STEPS, PREFIX, EVENTS, SEEDS, CPU = parent.STEPS, parent.PREFIX, parent.EVENTS, parent.SEEDS, parent.CPU
BANDS = ("full", "even_retained_positions")
VALIDATION = tuple(product(MODES, BANDS))


def contract(bank, seed, arm, producer):
    if arm not in ARMS:
        raise ValueError("Unknown registered spatial arm")
    original = parent.contract(bank, seed, "off")
    baseline = parent._baseline(bank.release, seed)
    model = initialize(baseline, arm, original["module_seed"])
    exposure = original["exposure"]
    for i, event in enumerate(exposure):
        spatial, band = VALIDATION[i % len(VALIDATION)]
        event.update(view=band, spatial=spatial, condition="matched")
    return dict(original, kind=FORMAT, arm=arm, producer=producer,
                exposure=exposure, validation_views=[list(pair) for pair in VALIDATION],
                transforms={mode: transform_spec(mode) for mode in MODES},
                initialization=parent._initial(model, baseline),
                parameter_count=sum(p.numel() for p in model.parameters()))


def _details(predictions, labels):
    idx = torch.where(labels > 0)[0]
    y = labels[idx]-1
    result = {}
    for name, values in zip(("fused","graph","cnn"), predictions[:3], strict=True):
        p = values[idx]
        classes = values.shape[1]
        confusion = torch.bincount(y*classes+p.argmax(1), minlength=classes**2).reshape(classes, classes)
        result[name] = dict(correct=int(confusion.diagonal().sum()), count=len(y), confusion=confusion.tolist(),
                           nll_sum=float(-p[range(len(y)),y].clamp_min(1e-8).log().sum()))
    return result


class Run(parent.Run):
    def __init__(self, bank, registered, producer):
        if contract(bank, registered["seed"], registered["arm"], producer) != registered:
            raise ValueError("Synthetic spatial contract or producer differs")
        self._bank, self._contract = bank, deepcopy(registered)
        self.baseline = parent._baseline(bank.release, registered["seed"])
        self.model = initialize(self.baseline, registered["arm"], registered["module_seed"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=registered["learning_rate"])
        self._history, self._validation, self._selected = [], [], None
        self.added = tuple(registered["initialization"]["fresh"])

    def _view(self, row, bands, spatial):
        return materialize(self._bank, row, parent._condition(bands), spatial, self._contract["producer"])

    @torch.no_grad()
    def _evaluate(self, step):
        self.model.eval()
        rows = []
        for row in self._bank.role_rows("devval"):
            for spatial, bands in VALIDATION:
                actual = self._view(row, bands, spatial)
                predictions, trace = forward(self.model, self._contract["arm"], actual, condition="matched")
                bank_view = actual.bank_view()
                labels = parent._targets(row, bank_view)
                rows.append(dict(tile=row["id"], view=bands, spatial=spatial, clock=step,
                                 observation=actual.observation_id, heads=heads(predictions, bank_view),
                                 scores=_details(predictions, labels), graph=trace))
        nll = sum(r["scores"]["fused"]["nll_sum"] for r in rows)
        count = sum(r["scores"]["fused"]["count"] for r in rows)
        score = nll/count
        self._validation.append(dict(step=step, fused_nll=score, rows=rows))
        if self._selected is None or score < self._selected["fused_nll"]:
            self._selected = dict(step=step, fused_nll=score, state=parent._state(self.model))

    def update(self, event):
        self._verify_input()
        cursor = len(self._history)
        if (not self._validation or cursor >= STEPS or event != self._contract["exposure"][cursor]):
            raise ValueError("Spatial event differs from the registered next update")
        event = self._contract["exposure"][cursor]
        row = self._bank.role_row("devtrain", event["tile"])
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        actual = self._view(row, event["view"], event["spatial"])
        out, trace = forward(self.model, self._contract["arm"], actual, condition=event["condition"])
        bank_view = actual.bank_view()
        target = parent._targets(row, bank_view)
        if not (target > 0).any():
            raise ValueError("Synthetic observation lost its released training samples")
        classes = actual.scene.classes
        mask = (target > 0)[:,None].expand(-1,classes).float()
        onehot = torch.nn.functional.one_hot((target-1).clamp_min(0), classes).float()*mask
        loss = probability_loss_components(out, onehot, mask)
        if not torch.isfinite(loss["total"]):
            raise ValueError("Nonfinite synthetic loss")
        named = dict(self.model.named_parameters())
        before = {key: tensor_hash(named[key]) for key in self.added}
        loss["total"].backward()
        gradient = {key: None if named[key].grad is None else float(named[key].grad.abs().sum()) for key in self.added}
        self.optimizer.step()
        fresh = {key: dict(gradient_l1=gradient[key], changed=before[key] != tensor_hash(named[key])) for key in self.added}
        self._history.append(dict(**event, observation=actual.observation_id, heads=heads(out, bank_view),
                                  graph=trace, fresh=fresh, losses=detached_loss_values(loss)))
        if event["step"] in EVENTS:
            self._evaluate(event["step"])

    def pack(self):
        return dict(format=FORMAT, contract=digest(self._contract), cursor=len(self._history),
                    model=parent._state(self.model), optimizer=deepcopy(self.optimizer.state_dict()), rng=_rng_state(CPU),
                    history=deepcopy(self._history), validation=deepcopy(self._validation), selected=deepcopy(self._selected))

    def restore(self, packet, cursor):
        self._verify_input()
        packet = deepcopy(packet)
        expected = self._contract["exposure"][:cursor]
        if (packet["format"] != FORMAT or packet["contract"] != digest(self._contract)
                or packet["cursor"] != cursor or len(packet["history"]) != cursor
                or [x["step"] for x in packet["validation"]] != [i for i in EVENTS if i <= cursor]
                or any({k: actual[k] for k in event} != event for actual,event in zip(packet["history"],expected,strict=True))):
            raise ValueError("Spatial checkpoint, exposure or validation clock differs")
        selected = min(packet["validation"], key=lambda value: value["fused_nll"])
        if (selected["step"], selected["fused_nll"]) != (packet["selected"]["step"], packet["selected"]["fused_nll"]):
            raise ValueError("Spatial checkpoint selected state differs")
        self.model.load_state_dict(packet["model"], strict=True)
        self.optimizer.load_state_dict(packet["optimizer"])
        if not all(int(state["step"]) == cursor for state in self.optimizer.state.values()):
            raise ValueError("Adam cursor differs")
        self._history, self._validation, self._selected = packet["history"], packet["validation"], packet["selected"]
        _restore_rng(packet["rng"], CPU)


def execute(bank, registered, phase, target, producer, prior=None):
    run = Run(bank, registered, producer)
    registered = run.contract
    start,end = {"full":(0,STEPS), "prefix":(0,PREFIX), "resume":(PREFIX,STEPS), "completed":(STEPS,STEPS)}[phase]
    if (start > 0) != (prior is not None) or target.exists():
        raise ValueError("Spatial stage needs its exact predecessor and unused destination")
    if start == 0:
        run.start()
    else:
        run.restore(prior, start)
    for event in registered["exposure"][start:end]:
        run.update(event)
    packet = run.pack()
    if phase == "completed" and not _exact(packet, prior):
        raise ValueError("Completed spatial stage changed")
    with io.BytesIO() as buffer:
        torch.save(packet, buffer)
        _atomic_new(target, buffer.getvalue())
    return dict(stage=phase, updates=end-start, checkpoint=str(target), bank_reads=bank.reads,
                selected=packet["selected"]["step"], state=state_hash(run.model))
