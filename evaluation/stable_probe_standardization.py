"""Fixed-order FP64 population standardization for the versioned linear probe."""
from fractions import Fraction
import math

import torch


def _above_floor(values, sd, floor):
    if abs(sd-floor) > 8*max(math.ulp(sd), math.ulp(floor)):
        return sd > floor
    # Resolve the branch exactly when rounded square roots straddle the threshold.
    exact = [Fraction(value) for value in values]
    count = len(exact)
    return count*sum(value**2 for value in exact)-sum(exact)**2 > (count*Fraction(floor))**2


def _statistics(values, floor):
    maximum = max(map(abs, values))
    # A power of two preserves nearby differences and bounds sums for very large inputs.
    exponent = math.frexp(maximum)[1] if maximum > 2.0**500 else 0
    working = [math.ldexp(value, -exponent) for value in values] if exponent else values
    anchor = working[0]
    offsets = [value-anchor for value in working]
    center = math.fsum(offsets)/len(offsets)
    centered = [value-center for value in offsets]
    magnitude = max(map(abs, centered))
    sd = magnitude*math.sqrt(math.fsum((value/magnitude)**2 for value in centered)/len(values)) if magnitude else 0.0
    population_sd = math.ldexp(sd, exponent)
    scaled = _above_floor(values, population_sd, floor)
    return (anchor, center, exponent, math.ldexp(anchor+center, exponent),
            population_sd if scaled else 1.0, sd if scaled else math.ldexp(1.0, -exponent), scaled)


def standardize(rows, floor):
    """Fit each column in train-row order and apply its same shifted coordinates to both roles."""
    train = rows["train"]
    stats = [_statistics(column.tolist(), floor) for column in train.unbind(dim=1)]
    anchors, offsets, exponents, means, scales, working_scales, scaled = zip(*stats, strict=True)
    anchor = torch.tensor(anchors, dtype=torch.float64)
    offset = torch.tensor(offsets, dtype=torch.float64)
    exponent = torch.tensor(exponents, dtype=torch.int32)
    mean = torch.tensor(means, dtype=torch.float64)
    scale = torch.tensor(scales, dtype=torch.float64)
    working_scale = torch.tensor(working_scales, dtype=torch.float64)
    # Apply the two-part center directly. Reconstructing mean before subtracting
    # would discard some of the low-variance signal that the shift preserves.
    standardized = {role: ((torch.ldexp(value, -exponent)-anchor)-offset)/working_scale
                    for role, value in rows.items()}
    return standardized, {"mean":mean, "scale":scale, "anchor":anchor, "center_offset":offset,
                          "working_exponent":exponent, "scaled_columns":list(scaled),
                          "near_constant_columns":scaled.count(False)}
