"""Independent Decimal population statistics and selected-head forward reference."""
from decimal import Decimal, localcontext

import numpy as np


def standardize(rows, floor, *, exact_roles=()):
    means, deviations, scales, scaled_columns = [], [], [], []
    output = {role: np.empty_like(value, dtype=np.float64) for role, value in rows.items()}
    exact = {role:[] for role in exact_roles}
    for col in range(rows["train"].shape[1]):
        values = {role: [Decimal.from_float(float(value)) for value in array[:,col]]
                  for role,array in rows.items()}
        exponents = [value.adjusted() for column in values.values() for value in column if value]
        with localcontext() as context:
            context.prec = max(96, 2*(max(exponents)-min(exponents))+50) if exponents else 96
            train = values["train"]
            count = len(train)
            mean = sum(train)/count
            variance = sum((value-mean)**2 for value in train)/count
            deviation = variance.sqrt()
            scaled = variance > Decimal.from_float(float(floor))**2
            scale = deviation if scaled else Decimal(1)
            means.append(float(mean)); deviations.append(float(deviation)); scales.append(float(scale))
            scaled_columns.append(scaled)
            for role, column in values.items():
                standardized = [(value-mean)/scale for value in column]
                output[role][:,col] = [float(value) for value in standardized]
                if role in exact:
                    exact[role].append(standardized)
    return output, {"mean":np.asarray(means), "sd":np.asarray(deviations), "scale":np.asarray(scales),
                    "scaled_columns":scaled_columns,
                    "exact_rows":{role:list(zip(*columns,strict=True)) for role,columns in exact.items()}}


def head(rows, weight, bias):
    """Use an independently accumulated dot and Decimal log-sum-exp."""
    with localcontext() as context:
        context.prec = 96
        def decimals(values):
            return [value if isinstance(value,Decimal) else Decimal.from_float(float(value)) for value in values]
        weights = [decimals(row) for row in weight]
        biases = decimals(bias)
        logits, logs, probs = [], [], []
        for raw in rows:
            x = decimals(raw)
            scores = [sum(a*b for a,b in zip(x,w,strict=True))+b for w,b in zip(weights,biases,strict=True)]
            maximum = max(scores)
            shifted = [score-maximum for score in scores]
            normalizer = sum(score.exp() for score in shifted).ln()
            logp = [score-normalizer for score in shifted]
            logits.append([float(v) for v in scores]); logs.append([float(v) for v in logp])
            probs.append([float(v.exp()) for v in logp])
    return tuple(np.asarray(value) for value in (logits, logs, probs))
