import numpy as np


def pwm_estimator(exceedances):
    y = np.sort(np.asarray(exceedances, dtype=float))
    y = y[np.isfinite(y)]

    n = len(y)

    if n < 3:
        raise ValueError("Need at least 3 exceedances")

    if np.any(y < 0):
        raise ValueError("Exceedances must be nonnegative")

    b0 = np.mean(y)

    weights = np.array([(n - 1 - i) / (n - 1) for i in range(n)])
    b1 = np.mean(weights * y)

    if b0 <= 0 or b1 <= 0:
        return np.nan, np.nan

    xi = 2 - (b0 / (b0 - 2 * b1))
    beta = (2 * b0 * b1) / (b0 - 2 * b1)

    if not np.isfinite(xi) or not np.isfinite(beta) or beta <= 0:
        return np.nan, np.nan

    return xi, beta