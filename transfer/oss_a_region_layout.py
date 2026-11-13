"""Dense mathematical reference for landmark-routed region messages."""

import torch


def routed_region_attention(query,key,value,landmarks,region_ids,k):
    """Use explicit region IDs with one selected landmark and a shared KV support.

    This CPU/GPU reference uses no upstream kernels. A single softmax over local
    selected keys and global landmark summaries is equivalent to log-sum-exp
    combination of the two attention groups.
    """
    if (query.ndim!=2 or key.shape!=query.shape or value.shape!=query.shape
            or landmarks.ndim!=2 or landmarks.shape[1]!=query.shape[1] or len(landmarks)==0
            or region_ids.shape!=(len(query),) or region_ids.dtype!=torch.long
            or len(torch.unique(region_ids))!=len(query) or type(k) is not int or not 1<=k<=len(query)):
        raise ValueError("Explicit unique region layout and feasible top-k required")
    if any(x.device!=query.device or x.dtype!=query.dtype for x in (key,value,landmarks)):
        raise ValueError("Same-device region query, key, value and landmark tensors required")
    order = region_ids.argsort()
    scores = landmarks @ key[order].T
    neighbors = order[scores.argsort(dim=1,descending=True,stable=True)[:,:k]]
    selected = (query @ landmarks.T).argmax(dim=1)
    scale = query.shape[1]**-.5
    global_values = (scores*scale).softmax(dim=1) @ value[order]
    local_keys,local_values = key[neighbors[selected]],value[neighbors[selected]]
    local_logits = (local_keys*query[:,None]).sum(dim=2)*scale
    global_logits = (query @ landmarks.T)*scale
    weights = torch.cat((local_logits,global_logits),dim=1).softmax(dim=1)
    output = (weights[:,:k,None]*local_values).sum(dim=1) + weights[:,k:] @ global_values
    return output,region_ids[neighbors[selected]],selected
