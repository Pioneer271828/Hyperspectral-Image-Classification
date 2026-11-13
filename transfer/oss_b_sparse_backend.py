"""Explicit PyG sum candidate for the existing two-dimensional COO multiply."""

from collections import Counter
from contextlib import contextmanager
import ast
from hashlib import sha256

import torch
from torch.overrides import TorchFunctionMode


PYG_SOURCE_SHA256 = "14fb794f8d706e0d2294b32d01c251e12c0faa5c43ae30ea3bb2a003e6276f0a"


def load_pyg_sum(path):
    """Load the verbatim sum branch and broadcast helper from a fixed source file."""
    raw = path.read_bytes()
    if sha256(raw).hexdigest()!=PYG_SOURCE_SHA256:
        raise ValueError("Fixed PyG source bytes required")
    source = ast.parse(raw)
    functions = {node.name:node for node in source.body if isinstance(node,ast.FunctionDef)}
    scatter = functions["scatter"]
    sum_branch = next(i for i,node in enumerate(scatter.body)
                      if isinstance(node,ast.If) and ast.unparse(node.test)=="reduce == 'sum' or reduce == 'add'")
    scatter.body = scatter.body[:sum_branch+1]
    unit = ast.Module(body=[functions["broadcast"],scatter],type_ignores=[])
    from typing import Optional

    namespace = dict(torch=torch,Tensor=torch.Tensor,Optional=Optional)
    exec(compile(unit,str(path),"exec"),namespace)
    pinned = namespace["scatter"]

    def sum_only(src,index,*,dim,dim_size,reduce):
        if reduce!="sum":
            raise ValueError("Only pinned PyG sum reduction is available")
        return pinned(src,index,dim=dim,dim_size=dim_size,reduce=reduce)

    return sum_only


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
