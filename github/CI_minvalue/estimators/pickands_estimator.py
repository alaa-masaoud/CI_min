import numpy as np


def pickands_estimator(data, k):
    

    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)

    n = len(x)

    if 4 * k >= n:
        raise ValueError("Need 4k < n")

    x1 = x[n - k]
    x2 = x[n - 2 * k]
    x4 = x[n - 4 * k]

    numerator = x1 - x2
    denominator = x2 - x4

    if numerator <= 0 or denominator <= 0:
        return np.nan

    xi = (1 / np.log(2)) * np.log(numerator / denominator)

    return xi