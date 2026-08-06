import numpy as np


def moment_estimator(data, k):
    
    y = np.asarray(data, dtype=float)
    y = y[np.isfinite(y)]
    y = y[y > 0]
    y = np.sort(y)

    n = len(y)
    if k <= 1 or k >= n:
        raise ValueError("Need 1 < k < n")

    threshold = y[-k - 1]
    top_k = y[-k:]
    if threshold <= 0:
        return np.nan

    log_ratios = np.log(top_k) - np.log(threshold)
    m1 = np.mean(log_ratios)
    m2 = np.mean(log_ratios ** 2)

    if not np.isfinite(m1) or not np.isfinite(m2) or m2 <= 0:
        return np.nan

    denom = 1.0 - (m1 * m1 / m2)
    if denom <= 0:
        return np.nan

    gamma_hat = m1 + 1.0 - 0.5 / denom
    return gamma_hat if np.isfinite(gamma_hat) else np.nan
