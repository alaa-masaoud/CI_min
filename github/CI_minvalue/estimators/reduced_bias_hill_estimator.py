import numpy as np


def _hill_sorted(y_sorted, k):
    """
    Standard Hill estimator for already sorted positive observations.
    """
    n = len(y_sorted)

    if k <= 0 or k >= n:
        return np.nan

    threshold = y_sorted[-k - 1]

    if threshold <= 0:
        return np.nan

    logs = np.log(y_sorted[-k:]) - np.log(threshold)
    hill = float(np.mean(logs))

    return hill if np.isfinite(hill) and hill > 0 else np.nan


def _hill_process(y_sorted, k):
    """
    Compute H_j for j = 1, ..., k efficiently.

    H_j = mean(log(Y_(n-i+1)) - log(Y_(n-j))), i=1,...,j.
    """
    n = len(y_sorted)

    if k <= 0 or k >= n:
        raise ValueError("Need 0 < k < n")

    log_y = np.log(y_sorted)

    # Descending top-k log observations.
    top_logs = log_y[-k:][::-1]

    # Cumulative sums of the j largest log observations.
    cumulative_top_logs = np.cumsum(top_logs)

    js = np.arange(1, k + 1, dtype=int)

    # For each j, the Hill threshold is the (j+1)-th largest observation.
    threshold_logs = log_y[n - js - 1]

    hill_values = cumulative_top_logs / js - threshold_logs

    return hill_values


def reduced_bias_hill_estimator(
    data,
    k,
    rho=-1.0,
    min_j=None,
    min_fraction=0.10,
    weighted=True,
):
   
   
    y = np.asarray(data, dtype=float)
    y = y[np.isfinite(y)]
    y = y[y > 0.0]
    y = np.sort(y)

    n = len(y)

    if k <= 5 or k >= n:
        raise ValueError("Need 5 < k < n")

    if not np.isfinite(rho) or rho >= 0.0:
        raise ValueError("rho must be finite and negative")

    hill_k = _hill_sorted(y, k)

    if not np.isfinite(hill_k) or hill_k <= 0.0:
        return np.nan

    if min_j is None:
        min_j = max(5, int(np.ceil(min_fraction * k)))

    min_j = int(min_j)

    if min_j < 2:
        min_j = 2

    if min_j >= k:
        return hill_k

    try:
        h_all = _hill_process(y, k)
    except Exception:
        return hill_k

    js_all = np.arange(1, k + 1, dtype=float)

    # Remove the very noisy smallest-j region.
    use = js_all >= min_j

    js = js_all[use]
    h_vals = h_all[use]

    valid = np.isfinite(h_vals) & (h_vals > 0.0)

    js = js[valid]
    h_vals = h_vals[valid]

    if len(h_vals) < 5:
        return hill_k

    z = (js / float(k)) ** (-rho)

    if not np.all(np.isfinite(z)):
        return hill_k

    # Direct regression:
    #
    #     H_j = gamma + A*z_j
    #
    # The intercept is the extrapolated value at z=0.
    X = np.column_stack(
        [
            np.ones(len(z), dtype=float),
            z,
        ]
    )

    try:
        if weighted:
            # Larger j values are based on more upper order statistics and
            # therefore typically have lower variance.
            weights = js / float(k)
            sqrt_weights = np.sqrt(weights)

            X_fit = X * sqrt_weights[:, None]
            y_fit = h_vals * sqrt_weights
        else:
            X_fit = X
            y_fit = h_vals

        coefficients, _, rank, _ = np.linalg.lstsq(
            X_fit,
            y_fit,
            rcond=None,
        )

        if rank < 2:
            return hill_k

        gamma_hat = float(coefficients[0])

        if np.isfinite(gamma_hat) and gamma_hat > 0.0:
            return gamma_hat

    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        pass

    return hill_k