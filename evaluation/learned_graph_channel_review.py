"""Independent dense oracle and functional interventions for trained channel states."""
from itertools import product

import numpy as np
import torch

from evaluation import learned_graph_channel as task
from research_runtime import tensor_hash


EPS = .002


def directed_distance(pairs, source=0, target=5):
    reach = {source}
    for step in range(task.NODES):
        if target in reach:
            return step
        reach |= {i for i, j in pairs if j in reach}
    return None


def numpy_logits(state, features, pairs, batch, stratum, arm):
    """Reconstruct dense, original-node-order predictions without the producer forward."""
    a = np.zeros((task.NODES, task.NODES), np.float64)
    for receiver, sender in pairs:
        a[receiver, sender] = 1
    a /= a.sum(axis=1, keepdims=True)
    get = lambda name: state[name].detach().numpy()
    conv = lambda x, name: a @ (x @ get(f"graph.{name}.weight") + get(f"graph.{name}.bias"))
    x = np.asarray(features.detach() if torch.is_tensor(features) else features, dtype=np.float64)
    first = conv(x, "hgc1")
    prefix = next(k.rsplit(".", 1)[0] for k in state if k.startswith("graph.") and k.endswith("running_mean"))
    first = (first-get(prefix+".running_mean"))/np.sqrt(get(prefix+".running_var")+1e-5)
    first = np.maximum(first*get(prefix+".weight")+get(prefix+".bias"), 0)
    if arm == "shuffle":
        rows = (2, 3) if stratum == "capacity" else (2,)
        first[:, rows, :] = first[batch["donors"]][:, rows, :]
    second = conv(first, "hgc2")[:, 5, :]
    linear = (second @ get("readout.weight").T + get("readout.bias")).reshape(len(x), 2, 2)
    return linear[np.arange(len(x)), batch["ask"]]


def independent_metrics(logits, targets):
    z = logits.astype(np.float64)
    exp = np.exp(z-z.max(1, keepdims=True))
    probabilities = exp/exp.sum(1, keepdims=True)
    predictions = probabilities.argmax(1)
    cm = [[int(np.count_nonzero((targets == i) & (predictions == j))) for j in (0, 1)] for i in (0, 1)]
    return {"correct": int(np.trace(cm)), "support": [sum(row) for row in cm], "confusion": cm,
            "accuracy": float(np.mean(targets == predictions)),
            "nll": float(-np.log(probabilities[np.arange(len(targets)), targets]).mean()), "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist()}


def analyze(run, role):
    result = {}
    state = run.model.state_dict()
    prior_hash = task.scientific_hash(run.model)
    for key in task.STRATA:
        batch = run.batch(role, key)
        features = task.model_features(batch)
        signs, asks, expected = batch["signs"], batch["ask"], batch["y"]
        if list(expected) != [int(bool(signs[i, ask] > 0)) for i, ask in enumerate(asks)]:
            raise ValueError("Independent answer disagrees with the generator")
        pairs = task.edges(key, run.arm)
        distances = [directed_distance(pairs, src) for src in (0, 1)]
        matrix = run.matrices[key]
        oracle = numpy_logits(state, features, pairs, batch, key, run.arm)

        def score(x, operation="baseline"):
            return run.model(x, matrix, batch, key, run.arm, operation)

        def margin(x):
            output = score(x)
            return output[:, 1]-output[:, 0]

        with torch.no_grad():
            baseline = score(features)
            if not np.allclose(oracle, baseline.numpy(), atol=2e-5, rtol=2e-5):
                raise ValueError("Independent dense reconstruction rejected actual selected predictions")
            source = features.clone()
            recipient = torch.arange(len(source))
            source[recipient, asks, 0] *= -1
            noise = features.clone()
            noise[:, 1:, 4:] = noise[:, 1:, 4:].roll(1, dims=0)
            observed = {"baseline": baseline, "sender_signal_flip": score(source), "noise_permutation": score(noise),
                        "relay_zero": score(features, "relay_zero"), "relay_replace": score(features, "relay_replace")}
            def shifted(step):
                # The independent dense oracle uses FP64 to resolve the local
                # linear segment near a ReLU switch selected by the trained model.
                raw = features.detach().numpy().astype(np.float64)
                if run.arm != "shuffle":
                    raw[np.arange(len(asks)), asks, 0] += step
                    value = numpy_logits(state, raw, pairs, batch, key, run.arm)
                    return value[:, 1]-value[:, 0]
                values = []
                for i, src in enumerate(asks):
                    x = raw.copy()
                    x[i, src, 0] += step
                    value = numpy_logits(state, x, pairs, batch, key, run.arm)[i]
                    values.append(value[1]-value[0])
                return np.asarray(values)

            global_fd = (shifted(EPS)-shifted(-EPS))/(2*EPS)
            epsilon = EPS/1000
            small_left = (shifted(0.)-shifted(-epsilon))/epsilon
            small_right = (shifted(epsilon)-shifted(0.))/epsilon
            finite = (small_left+small_right)/2
        inputs = features.clone().requires_grad_(True)
        values = margin(inputs)
        if run.arm == "shuffle":
            jacobian = torch.stack([torch.autograd.grad(values[i], inputs, retain_graph=i+1 < len(asks))[0][i, s, 0]
                                    for i, s in enumerate(asks)])
        else:
            jacobian = torch.autograd.grad(values.sum(), inputs)[0][recipient, asks, 0]
        slope = jacobian.detach().double().numpy()
        sharp = np.abs(small_left-small_right) > 1e-5
        regular = ~sharp
        tolerance = 5e-4
        if (not np.allclose(slope[regular], finite[regular], rtol=1e-4, atol=tolerance)
                or np.any(slope[sharp] < np.minimum(small_left, small_right)[sharp]-tolerance)
                or np.any(slope[sharp] > np.maximum(small_left, small_right)[sharp]+tolerance)):
            raise ValueError("Local trained signal Jacobian differs from the FP64 oracle slopes")
        base_pred = observed["baseline"].argmax(dim=1).numpy()
        base_prob = observed["baseline"].softmax(dim=1).numpy()
        interventions = {}
        for name, logits in observed.items():
            m = independent_metrics(logits.numpy(), expected)
            pred = np.array(m["predictions"])
            interventions[name] = {**m, "original_answer": expected.tolist(),
                "prediction_changes": int(np.count_nonzero(pred != base_pred)),
                "maximum_probability_change": float(np.abs(np.asarray(m["probabilities"])-base_prob).max()),
                "loss_change": m["nll"]-independent_metrics(baseline.numpy(), expected)["nll"]}
        if not torch.equal(baseline, score(features).detach()) or task.scientific_hash(run.model) != prior_hash:
            raise ValueError("Functional diagnostic changed the selected graph state")
        impossible = np.asarray([distances[a] is None or distances[a] > 2 for a in asks])
        if impossible.any() and (torch.count_nonzero(jacobian[impossible]) or
                not torch.equal(baseline[impossible], observed["sender_signal_flip"][impossible])):
            raise ValueError("An unreachable source leaked into the target output")
        targets = (1-expected)
        m = independent_metrics(observed["sender_signal_flip"].numpy(), targets)
        result[key] = {"instances": batch["fingerprint"], "input_hash": tensor_hash(features),
                       "edges": pairs, "source_to_target_hops": distances,
                       "source_query": asks.tolist(), "sending_values": signs.tolist(), "source_noise_free": True,
                       "with_flipped_answer": {k: m[k] for k in ("correct", "support", "accuracy", "nll", "confusion")},
                       "oracle_logits_max_abs": float(np.abs(oracle-baseline.numpy()).max()),
                       "jacobian": jacobian.tolist(), "finite_difference": finite.tolist(),
                       "finite_difference_large": global_fd.tolist(), "one_sided_small": [small_left.tolist(), small_right.tolist()],
                       "epsilon_large": EPS, "epsilon_small": epsilon, "small_nonsmooth": sharp.tolist(),
                       "jacobian_max_abs_error_at_smooth_points": float(np.max(np.abs(slope[regular]-finite[regular]))) if regular.any() else None,
                       "arms": interventions}
    return result
