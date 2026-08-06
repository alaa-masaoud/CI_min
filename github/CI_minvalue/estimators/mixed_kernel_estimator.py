import numpy as np


def _hill_sorted(y_sorted, k):
    threshold = y_sorted[-k - 1]
    if threshold <= 0:
        return np.nan
    return float(np.mean(np.log(y_sorted[-k:]) - np.log(threshold)))


def mixed_moment_kernel_estimator(data, k, bandwidth=0.30):
    """
    Kernel-smoothed Hill/moment-style estimator over nearby k values.

    This smooths the Hill curve around k instead of committing to one threshold.
    It uses a Gaussian kernel across k' in [roughly (1-bandwidth)k, (1+bandwidth)k].
    """
    y = np.asarray(data, dtype=float)
    y = y[np.isfinite(y)]
    y = y[y > 0]
    y = np.sort(y)
    n = len(y)
    if k <= 5 or k >= n:
        raise ValueError("Need 5 < k < n")

    lo = max(5, int(np.floor(k * (1.0 - bandwidth))))
    hi = min(n - 1, int(np.ceil(k * (1.0 + bandwidth))))
    ks = np.arange(lo, hi + 1)
    vals = np.array([_hill_sorted(y, int(kk)) for kk in ks], dtype=float)
    mask = np.isfinite(vals) & (vals > 0)
    if mask.sum() == 0:
        return np.nan

    scale = max(1.0, bandwidth * k / 2.0)
    weights = np.exp(-0.5 * ((ks[mask] - k) / scale) ** 2)
    gamma_hat = float(np.sum(weights * vals[mask]) / np.sum(weights))
    return gamma_hat if np.isfinite(gamma_hat) and gamma_hat > 0 else np.nan
