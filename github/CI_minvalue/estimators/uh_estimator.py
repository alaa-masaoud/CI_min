import numpy as np


def uh_exponential_regression_estimator(data, k):
    """
    Beirlant-style UH/exponential-regression tail-index estimator.

    Uses weighted log-spacings from the top k order stats and fits a simple
    exponential-regression trend. The intercept is the gamma estimate.
    """
    y = np.asarray(data, dtype=float)
    y = y[np.isfinite(y)]
    y = y[y > 0]
    y = np.sort(y)
    n = len(y)
    if k <= 5 or k >= n:
        raise ValueError("Need 5 < k < n")

    # Descending top k+1 values: z[0] largest, z[k] threshold.
    z = y[-(k + 1):][::-1]
    log_spacings = np.log(z[:-1]) - np.log(z[1:])
    j = np.arange(1, k + 1, dtype=float)
    weighted_spacings = j * log_spacings

    x = np.log((k + 1.0) / j)
    X = np.column_stack([np.ones(k), x])
    try:
        intercept, slope = np.linalg.lstsq(X, weighted_spacings, rcond=None)[0]
        if np.isfinite(intercept) and intercept > 0:
            return float(intercept)
    except Exception:
        pass

    fallback = float(np.mean(weighted_spacings))
    return fallback if np.isfinite(fallback) and fallback > 0 else np.nan
