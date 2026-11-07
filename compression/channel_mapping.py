"""Explicit state slicing for a smaller model after accepted MON intervention."""
from collections import Counter

import torch

from joint_training import _tree_fingerprint
from model.research_config import digest
from monitoring.dependency_groups import POLICY as INTERVENTION_POLICY


def compact_state(reference, template, intervention_plan, receipt, declaration, *, expected_sha256):
    """Return a fresh compact state and exact channel maps, without loading either model."""
    if (digest(declaration) != expected_sha256 or digest(receipt) != declaration["receipt_sha256"]
            or receipt["policy"] != INTERVENTION_POLICY or receipt["plan_sha256"] != digest(intervention_plan)
            or receipt["optimizer_updates"] != 0 or receipt["query_labels_accessed"] is not False):
        raise ValueError("Exact frozen reversible intervention evidence required")
    names = declaration["groups"]
    if not names or len(set(names)) != len(names):
        raise ValueError("Unique accepted removal groups required")
    matching = [r for r in receipt["trials"] if set(r["groups"]) == set(names)]
    if (len(matching) != 1 or matching[0]["passed"] is not True
            or matching[0]["protection"]["passes_metric_guards"] is not True
            or matching[0]["failed_single_groups"] or matching[0]["execution"]["state_unchanged"] is not True):
        raise ValueError("The exact removal combination must pass reversible validation")
    selected = {g["name"]: g for g in intervention_plan["groups"] if g["name"] in names}
    if set(selected) != set(names):
        raise ValueError("Removal groups must exist in the validated intervention plan")
    sites = {(g["name"], s["observation"], s["channel"]) for g in selected.values() for s in g["sites"]}
    applied = {(r["observation"], c) for r in matching[0]["execution"]["interventions"] for c in r["channels"]}
    if not sites or applied != {(observation, channel) for _, observation, channel in sites}:
        raise ValueError("Validated intervention sites do not match the removal groups")
    if reference is template:
        raise ValueError("An independent compact template is required")
    original, target = reference.state_dict(), template.state_dict()
    if (set(original) != set(target) or any(not torch.is_tensor(t) for t in (*original.values(), *target.values()))):
        raise ValueError("Same named tensor-only state schema required")
    state_sha = _tree_fingerprint(original)
    if state_sha != intervention_plan["reference_state"] or state_sha != matching[0]["execution"]["reference_state"]:
        raise ValueError("Reference state differs from the validated checkpoint")
    storage = Counter((str(t.device), t.untyped_storage().data_ptr()) for t in original.values()
                      if t.layout == torch.strided and t.numel())
    target_storage = Counter((str(t.device), t.untyped_storage().data_ptr()) for t in target.values()
                             if t.layout == torch.strided and t.numel())
    if storage.keys() & target_storage.keys():
        raise ValueError("Compact template must not share source tensor storage")
    axis_plans, mapped_sites = {}, set()
    for item in declaration["axes"]:
        name, axis = item["tensor"], item["axis"]
        if name not in original or type(axis) is not int or not 0 <= axis < original[name].ndim:
            raise ValueError("Every mapping needs an explicit tensor axis")
        old, new = original[name], target[name]
        if (old.layout != torch.strided or new.layout != torch.strided or (name, axis) in axis_plans
                or storage[(str(old.device), old.untyped_storage().data_ptr())] != 1
                or target_storage[(str(new.device), new.untyped_storage().data_ptr())] != 1):
            raise ValueError("Unique dense axes with independent source and template storage required")
        size = old.shape[axis]
        drops = {}
        for entry in item["remove"]:
            site = (entry["group"], entry["observation"], entry["channel"])
            index = entry["index"]
            if site not in sites or type(index) is not int or not 0 <= index < size or index in drops:
                raise ValueError("Distinct in-range indices must bind to validated dependency sites")
            drops[index] = entry
            mapped_sites.add(site)
        keep = [i for i in range(size) if i not in drops]
        if not drops or not keep:
            raise ValueError("A mapped axis must remove channels and retain at least one")
        old_to_new = [None] * size
        for index, previous in enumerate(keep):
            old_to_new[previous] = index
        axis_plans[(name, axis)] = {"tensor": name, "axis": axis, "old_to_new": old_to_new,
                                   "new_to_old": keep, "removed": [drops[i] for i in sorted(drops)]}
    if mapped_sites != sites:
        raise ValueError("Every validated dependency site needs explicit tensor mappings")

    result = {}
    for name, old in original.items():
        new = target[name]
        shape = tuple(len(axis_plans[name, i]["new_to_old"]) if (name, i) in axis_plans else size
                      for i, size in enumerate(old.shape))
        if (old.dtype != new.dtype or old.layout != new.layout or old.device != new.device or shape != tuple(new.shape)):
            raise ValueError("Compact template does not match the complete mapped state schema: " + name)
        value = old.detach()
        for axis in range(old.ndim):
            if (name, axis) in axis_plans:
                indices = torch.tensor(axis_plans[name, axis]["new_to_old"], dtype=torch.long, device=old.device)
                value = value.index_select(axis, indices)
        result[name] = value.clone()
    return result, {"policy": "mon_explicit_compact_channel_map_v1", "plan_sha256": expected_sha256,
                    "reference_state": state_sha, "compact_state": _tree_fingerprint(result), "groups": names,
                    "axes": list(axis_plans.values()), "optimizer_state_transferred": False}
