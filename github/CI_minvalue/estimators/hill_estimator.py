import numpy as np


def hill_estimator(data, k):
    """
    Standard Hill estimator for upper-tail Frechet-domain data.

    In main.py, lower-tail Weibull data X is already transformed by:
        Y = 1 / (X - lower_endpoint)

    So this function estimates gamma from Y.
    Then main.py converts alpha = 1 / gamma.
    """

    y = np.asarray(data, dtype=float)
    y = y[np.isfinite(y)]
    y = y[y > 0]
    y = np.sort(y)

    n = len(y)

    if k <= 0 or k >= n:
        raise ValueError("Need 0 < k < n")

    threshold = y[-k - 1]
    top_k = y[-k:]

    if threshold <= 0:
        return np.nan

    gamma_hat = np.mean(np.log(top_k) - np.log(threshold))

    if not np.isfinite(gamma_hat) or gamma_hat <= 0:
        return np.nan

    return gamma_hat