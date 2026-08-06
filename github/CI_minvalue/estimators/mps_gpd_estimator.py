import numpy as np
from scipy.optimize import minimize
from scipy.stats import genpareto


def mps_gpd_estimator(exceedances):
    """
    Maximum product of spacings estimator for a GPD fit to exceedances.
    Returns (xi, beta). Useful as an MLE alternative for unstable boundary cases.
    """
    x = np.asarray(exceedances, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x[x >= 0])
    n = len(x)
    if n < 5 or np.max(x) <= 0:
        raise ValueError("Need at least 5 nonnegative, nonconstant exceedances")

    mean_x = float(np.mean(x))
    try:
        xi0, _loc, beta0 = genpareto.fit(x, floc=0)
        if not np.isfinite(xi0):
            xi0 = 0.1
        if not np.isfinite(beta0) or beta0 <= 0:
            beta0 = max(mean_x, np.finfo(float).eps)
    except Exception:
        xi0, beta0 = 0.1, max(mean_x, np.finfo(float).eps)

    def neg_log_spacing(params):
        xi, log_beta = params
        beta = np.exp(log_beta)
        if beta <= 0:
            return np.inf
        t = 1.0 + xi * x / beta
        if np.any(t <= 0):
            return np.inf
        if abs(xi) < 1e-8:
            F = 1.0 - np.exp(-x / beta)
        else:
            F = 1.0 - t ** (-1.0 / xi)
        F = np.r_[0.0, F, 1.0]
        spacings = np.diff(F)
        if np.any(spacings <= 0) or np.any(~np.isfinite(spacings)):
            return np.inf
        return -float(np.sum(np.log(spacings)))

    best = None
    starts = [(xi0, beta0), (0.05, mean_x), (0.25, mean_x), (-0.1, mean_x)]
    for xi_s, beta_s in starts:
        res = minimize(
            neg_log_spacing,
            x0=np.array([xi_s, np.log(max(beta_s, 1e-12))]),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-8},
        )
        if np.all(np.isfinite(res.x)) and np.isfinite(res.fun):
            if best is None or res.fun < best.fun:
                best = res
    if best is None:
        return np.nan, np.nan

    xi, log_beta = best.x
    beta = float(np.exp(log_beta))
    if not np.isfinite(xi) or not np.isfinite(beta) or beta <= 0:
        return np.nan, np.nan
    return float(xi), beta
