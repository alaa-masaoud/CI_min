import numpy as np
from scipy.optimize import brentq


def grimshaw_mle_estimator(exceedances):
    
    x = np.asarray(exceedances, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0]
    n = len(x)
    if n < 5 or np.max(x) <= 0:
        raise ValueError("Need at least 5 nonnegative, nonconstant exceedances")

    mean_x = float(np.mean(x))

    def u(theta):
        return float(np.mean(np.log1p(theta * x)))

    def v(theta):
        return float(np.mean(1.0 / (1.0 + theta * x)))

    def h(theta):
        vv = v(theta)
        return u(theta) * vv + vv - 1.0

    # Search theta > 0 (corresponds to xi > 0, the case expected once
    # lower-tail Weibull-domain data has been reflected into the Frechet
    # domain by main.py's Y = 1/(X - lower_endpoint) transform). theta < 0
    # would need 1 + theta*x_i > 0 for all i (a finite-endpoint / xi<0 fit)
    # and is out of scope for this pipeline's use case.
    grid = np.geomspace(1e-6 / mean_x, 1e3 / mean_x, 400)
    vals = np.array([h(t) for t in grid])

    root = None
    for i in range(len(grid) - 1):
        a, b = vals[i], vals[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            try:
                root = brentq(h, grid[i], grid[i + 1], xtol=1e-12, maxiter=200)
                break
            except Exception:
                continue

    if root is None or root <= 0:
        return np.nan, np.nan

    xi_hat = u(root)
    if not np.isfinite(xi_hat) or xi_hat <= 0:
        return np.nan, np.nan

    beta_hat = xi_hat / root
    if not np.isfinite(beta_hat) or beta_hat <= 0:
        return np.nan, np.nan

    return float(xi_hat), float(beta_hat)