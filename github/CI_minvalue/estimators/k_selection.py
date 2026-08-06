import numpy as np


def hill_for_k(data, k):
    y = np.asarray(data, dtype=float)
    y = y[np.isfinite(y)]
    y = y[y > 0]
    y = np.sort(y)
    n = len(y)
    if k <= 0 or k >= n:
        return np.nan
    threshold = y[-k - 1]
    if threshold <= 0:
        return np.nan
    out = float(np.mean(np.log(y[-k:]) - np.log(threshold)))
    return out if np.isfinite(out) and out > 0 else np.nan


def hill_k_sweep(data, k_values):
    """
    Return gamma_hat and alpha_hat for a sweep of k values.

    """
    rows = []
    for k in k_values:
        gamma = hill_for_k(data, int(k))
        alpha = 1.0 / gamma if np.isfinite(gamma) and gamma > 0 else np.nan
        rows.append({"k": int(k), "gamma_hat": gamma, "alpha_hat": alpha})
    return rows


def auto_k_selector(*args, **kwargs):
   
    raise NotImplementedError(
        "No automatic k-selector here passed validation against a Monte "
        "Carlo oracle. Use hill_k_sweep() and pick k by inspection, or "
        "implement Danielsson et al. (2001) with the original paper in hand."
    )