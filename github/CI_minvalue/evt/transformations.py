from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def validate_sample(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=float)

    if values.ndim != 1:
        raise ValueError("Data must be one-dimensional.")
    if values.size < 4:
        raise ValueError("At least four observations are required.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Data contains non-finite values.")

    return values


def estimate_transform_endpoint(
    data: np.ndarray,
    k: int,
) -> float:
    """
    Estimate the lower endpoint from the subsample only.

    For the first k lower order statistics, fit the Weibull-tail
    quantile approximation

        X_(j) = m + a * p_j**(1/alpha),

    where p_j = j/(n+1), m is the lower endpoint, a > 0 and alpha > 0.

    This is only preprocessing required by the uploaded upper-tail EVT
    estimators. The known endpoint of Uniform(0,1) is never supplied.
    """
    x = np.sort(validate_sample(data))
    n = x.size

    if not 4 <= k < n:
        raise ValueError("k must satisfy 4 <= k < len(data).")

    lower = x[:k]
    probabilities = np.arange(1, k + 1, dtype=float) / (n + 1.0)

    local_spacing = max(float(lower[-1] - lower[0]), np.finfo(float).eps)
    initial_m = float(lower[0] - local_spacing / max(k - 1, 1))
    initial_a = max(float(lower[-1] - initial_m), np.finfo(float).eps)
    initial_alpha = 1.0

    # Parameters are m, log(a), log(alpha). Bounds keep m strictly below
    # the observed minimum while preserving a fixed residual-vector length.
    def residuals(theta: np.ndarray) -> np.ndarray:
        m = float(theta[0])
        a = float(np.exp(theta[1]))
        alpha = float(np.exp(theta[2]))
        fitted = m + a * probabilities ** (1.0 / alpha)
        return fitted - lower

    endpoint_margin = max(
        np.finfo(float).eps * max(1.0, abs(float(lower[0]))),
        1e-12,
    )
    endpoint_lower_bound = float(lower[0] - 100.0 * local_spacing)
    endpoint_upper_bound = float(lower[0] - endpoint_margin)

    result = least_squares(
        residuals,
        x0=np.array([initial_m, np.log(initial_a), np.log(initial_alpha)]),
        bounds=(
            np.array([endpoint_lower_bound, -30.0, -5.0]),
            np.array([endpoint_upper_bound, 30.0, 8.0]),
        ),
        max_nfev=5_000,
    )

    if not result.success:
        raise ValueError(f"Endpoint fit failed: {result.message}")

    endpoint = float(result.x[0])
    sample_min = float(x[0])

    # Numerical guard only; it does not use the theoretical endpoint.
    if not np.isfinite(endpoint) or endpoint >= sample_min:
        smallest_spacing = np.diff(x[: min(k + 1, n)])
        smallest_spacing = smallest_spacing[smallest_spacing > 0.0]
        if smallest_spacing.size == 0:
            raise ValueError("Cannot estimate endpoint from constant lower tail.")
        endpoint = sample_min - float(np.median(smallest_spacing))

    return endpoint


def weibull_lower_to_frechet_upper(
    data: np.ndarray,
    lower_endpoint: float,
) -> np.ndarray:
    """Transform X to Y = 1 / (X - estimated lower endpoint)."""
    x = validate_sample(data)
    distances = x - float(lower_endpoint)

    if np.any(distances <= 0.0):
        raise ValueError("Estimated endpoint must be below every observation.")

    y = 1.0 / distances

    if not np.all(np.isfinite(y)) or np.any(y <= 0.0):
        raise ValueError("Transformation produced invalid values.")

    return y


def upper_tail_exceedances(
    frechet_data: np.ndarray,
    k: int,
) -> tuple[np.ndarray, float]:
    y = np.sort(validate_sample(frechet_data))
    n = y.size

    if not 1 <= k < n:
        raise ValueError("k must satisfy 1 <= k < len(data).")

    threshold = float(y[-k - 1])
    exceedances = y[-k:] - threshold
    exceedances = exceedances[np.isfinite(exceedances)]
    exceedances = exceedances[exceedances >= 0.0]

    if exceedances.size < 3 or np.max(exceedances) <= 0.0:
        raise ValueError("Not enough nonconstant exceedances.")

    return exceedances, threshold
