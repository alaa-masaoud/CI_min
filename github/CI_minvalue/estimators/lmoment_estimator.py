import numpy as np


def lmoment_estimator(exceedances):
    y = np.sort(np.asarray(exceedances, dtype=float))
    y = y[np.isfinite(y)]

    n = len(y)

    if n < 3:
        raise ValueError("Need at least 3 exceedances")

    if np.any(y < 0):
        raise ValueError("Exceedances must be nonnegative")

    # Probability-weighted moments
    b0 = np.mean(y)
    weights = np.array([i / (n - 1) for i in range(n)])
    b1 = np.mean(weights * y)

    # L-moments
    l1 = b0
    l2 = 2 * b1 - b0

    if l1 <= 0 or l2 <= 0:
        return np.nan, np.nan

    tau = l2 / l1

    if tau <= 0 or tau >= 1:
        return np.nan, np.nan

    # GPD L-moment estimator
    xi = 2 - (1 / tau)
    beta = l1 * (1 - xi)

    if not np.isfinite(xi) or not np.isfinite(beta) or beta <= 0:
        return np.nan, np.nan

    return xi, beta