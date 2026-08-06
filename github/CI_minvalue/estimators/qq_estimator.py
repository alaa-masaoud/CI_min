import numpy as np


def qq_estimator(data, k):
    """
    Kratz-Resnick-style QQ/Pareto quantile plot regression estimator.

    Regresses log top order statistics above the threshold against theoretical
    exponential quantiles. The slope estimates gamma for the Frechet tail.
    """
    y = np.asarray(data, dtype=float)
    y = y[np.isfinite(y)]
    y = y[y > 0]
    y = np.sort(y)
    n = len(y)
    if k <= 5 or k >= n:
        raise ValueError("Need 5 < k < n")

    threshold = y[-k - 1]
    top = y[-k:]
    if threshold <= 0 or np.any(top <= 0):
        return np.nan

    log_excess_ratio = np.log(top) - np.log(threshold)
    probs = (np.arange(1, k + 1, dtype=float) - 0.5) / k
    theo = -np.log(1.0 - probs)  # exponential QQ axis for Pareto log ratios
    X = np.column_stack([np.ones(k), theo])
    try:
        intercept, slope = np.linalg.lstsq(X, log_excess_ratio, rcond=None)[0]
        if np.isfinite(slope) and slope > 0:
            return float(slope)
    except Exception:
        pass
    return np.nan
