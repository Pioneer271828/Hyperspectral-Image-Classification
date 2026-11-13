"""Explicit PyG sum candidate for the existing two-dimensional COO multiply."""

from collections import Counter
from contextlib import contextmanager

import torch
from torch.overrides import TorchFunctionMode


def coo_product(matrix, features, scatter):
    if (matrix.layout!=torch.sparse_coo or matrix.ndim!=2 or features.layout!=torch.strided
            or features.ndim!=2 or matrix.shape[1]!=features.shape[0]
            or matrix.dtype!=features.dtype or matrix.device!=features.device):
        raise ValueError("Compatible two-dimensional COO and dense features required")
    coalesced = matrix.coalesce()
    row,col = coalesced.indices()
    messages = coalesced.values()[:,None]*features[col]
    return scatter(messages,row,dim=0,dim_size=matrix.shape[0],reduce="sum")


class _ProductMode(TorchFunctionMode):
    def __init__(self,scatter):
        self.scatter = scatter
        self.calls = Counter()

    def __torch_function__(self,func,types,args=(),kwargs=None):
        kwargs = kwargs or {}
        if func is torch.sparse.mm:
            if len(args)!=2 or kwargs:
                raise ValueError("Candidate only supports sum COO products")
            matrix,features = args
            result = coo_product(matrix,features,self.scatter)
            self.calls["coo_product"] += 1
            return result
        return func(*args,**kwargs)


@contextmanager
def use_product(scatter):
    """Intercept only the explicit bounded call and leave other callers unchanged."""
    with _ProductMode(scatter) as mode:
        yield mode.calls
