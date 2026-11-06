"""Read-only region-aligned descriptive statistics, independently implemented.

Centered covariance participation ratio follows arXiv:2601.22841v2 section3.2.
No channel selection, model alteration, accuracy or compression inference.
"""
import math

import numpy as np
import torch


def _matrix(value):
    if (not torch.is_tensor(value) or value.layout != torch.strided or value.ndim != 2
            or not value.is_floating_point() or not 1 <= value.shape[1] <= 512):
        raise ValueError("Finite floating dense [samples,1..512 channels] required")
    result = value.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite features")
    return result


def _center(value, variance_floor):
    centered = value - value.mean(0)
    variance = centered.square().sum(0) / (len(value) - 1)
    if not torch.isfinite(centered).all() or not torch.isfinite(variance).all():
        raise ValueError("Feature scale overflows FP64 covariance")
    return centered, variance, variance > variance_floor


def _correlation(first, second):
    # Unit-column normalization avoids multiplying two large norm values.
    a = first / torch.linalg.vector_norm(first, dim=0)
    b = second / torch.linalg.vector_norm(second, dim=0)
    return (a.T @ b).clamp(-1., 1.)


@torch.no_grad()
def matrix_summary(features, *, variance_floor=1e-24):
    x = _matrix(features)
    if (isinstance(variance_floor, bool) or not isinstance(variance_floor, (float, int))
            or not math.isfinite(variance_floor) or variance_floor < 0):
        raise ValueError("Finite nonnegative variance floor required")
    result = {"samples": len(x), "channels": x.shape[1], "centered": True, "covariance_ddof": 1,
        "variance_floor": variance_floor, "variance_units": "feature_squared",
        "covariance_eigenvalues": [], "implicit_zero_eigenvalues": None,
        "explained_variance_ratio": [], "cumulative_explained_variance_ratio": [],
        "numerical_rank": None, "rank_singular_value_tolerance": None, "effective_dimension": None,
        "components_at_variance": None, "valid_correlation_channels": [], "correlation_pairs": 0,
        "mean_absolute_channel_correlation": None, "correlation_reason": "fewer_than_two_samples"}
    if len(x) < 2:
        return {**result, "status": "not_available", "reason": "fewer_than_two_samples"}
    xc, variance, valid = _center(x, variance_floor)
    indices = torch.where(valid)[0]
    result["valid_correlation_channels"] = indices.tolist()
    result["correlation_reason"] = None if len(indices) >= 2 else "fewer_than_two_valid_channels"
    if len(indices) >= 2:
        corr = _correlation(xc[:, valid], xc[:, valid])
        pairs = torch.triu_indices(len(indices), len(indices), 1)
        result["correlation_pairs"] = pairs.shape[1]
        result["mean_absolute_channel_correlation"] = float(corr[pairs[0], pairs[1]].abs().mean())
    singular = torch.linalg.svdvals(xc)
    eigen = (singular / math.sqrt(len(x) - 1)).square()
    total = eigen.sum()
    if not torch.isfinite(eigen).all() or not torch.isfinite(total):
        raise ValueError("Feature scale overflows FP64 eigenspectrum")
    result["covariance_eigenvalues"] = eigen.tolist()
    result["implicit_zero_eigenvalues"] = x.shape[1] - len(eigen)
    if total <= variance_floor:
        return {**result, "status": "not_available", "reason": "covariance_below_variance_floor"}
    # Participation ratio from normalized eigenvalues avoids squaring the trace.
    ratios = eigen / total
    cumulative = ratios.cumsum(0).clamp(max=1.)
    tolerance = max(x.shape) * torch.finfo(torch.float64).eps * float(singular[0])
    result.update(status="computed", reason=None, explained_variance_ratio=ratios.tolist(),
        cumulative_explained_variance_ratio=cumulative.tolist(), numerical_rank=int((singular > tolerance).sum()),
        rank_singular_value_tolerance=tolerance, effective_dimension=float(1. / ratios.square().sum()),
        components_at_variance={str(level): min(len(eigen), int((cumulative < level).sum()) + 1)
                                for level in (.8, .9, .95)})
    return result


@torch.no_grad()
def aligned_summary(cnn, graph, Q, *, maximum_regions, seed, variance_floor=1e-24):
    if type(maximum_regions) is not int or not 1 <= maximum_regions <= 8192:
        raise ValueError("Explicit region cap in 1..8192 required")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Explicit local sampling seed required")
    c, g = _matrix(cnn), _matrix(graph)
    if (not torch.is_tensor(Q) or Q.ndim != 2 or Q.shape != (len(c), len(g))
            or not Q.is_floating_point() or Q.layout not in (torch.strided, torch.sparse_coo, torch.sparse_csr)):
        raise ValueError("Membership Q must match [CNN pixels,Graph regions]")
    q = Q.detach().to(device="cpu", dtype=torch.float64)
    values = q if q.layout == torch.strided else q._values() if q.layout == torch.sparse_coo else q.values()
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Nonnegative finite membership weights required")
    if q.layout == torch.strided:
        mass, pooled = q.sum(0), q.T @ c
    else:
        q = q.to_sparse_coo().coalesce()
        mass = torch.sparse.sum(q, dim=0).to_dense()
        pooled = torch.sparse.mm(q.transpose(0, 1).coalesce(), c)
    if not torch.isfinite(mass).all() or not torch.isfinite(pooled).all():
        raise ValueError("Membership pooling overflows FP64")
    eligible = torch.where(mass > 0)[0].numpy()
    chosen = eligible if len(eligible) <= maximum_regions else np.sort(
        np.random.default_rng(seed).choice(eligible, maximum_regions, replace=False))
    indices = torch.from_numpy(chosen.copy())
    a = pooled[indices] / mass[indices, None]
    b = g[indices]
    first, second = matrix_summary(a, variance_floor=variance_floor), matrix_summary(b, variance_floor=variance_floor)
    cross = {"mean_absolute_correlation": None, "valid_channel_pairs": 0, "reason": "fewer_than_two_samples"}
    if len(indices) >= 2:
        ac, _, av = _center(a, variance_floor); bc, _, bv = _center(b, variance_floor)
        count = int(av.sum()) * int(bv.sum())
        cross = {"mean_absolute_correlation": float(_correlation(ac[:, av], bc[:, bv]).abs().mean()) if count else None,
            "valid_channel_pairs": count, "reason": None if count else "no_valid_channel_pairs"}
    return {"backend": "CPU_FP64_centered_SVD", "pooling": "Q_transpose_CNN_divided_by_column_mass",
        "weighting": "equal_sampled_regions", "pixels": len(c), "regions": len(g),
        "eligible_regions": len(eligible), "zero_mass_regions": int((mass == 0).sum()),
        "sampled_regions": len(chosen), "maximum_regions": maximum_regions, "seed": seed,
        "sampled_region_indices": chosen.tolist(), "sampled_region_masses": mass[indices].tolist(),
        "cnn": first, "graph": second, "cross_branch": cross,
        "interpretation": "Descriptive covariance/correlation only; no functional redundancy, pruning or speed conclusion"}
