import numpy as np
from scipy.stats import genpareto


def gpd_mle_estimator(data, threshold_quantile=0.90):
   

    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]

    u = np.quantile(x, threshold_quantile)

    exceedances = x[x > u] - u

    xi, loc, beta = genpareto.fit(exceedances, floc=0)

    return {
        "threshold": u,
        "xi": xi,
        "beta": beta,
        "exceedances": exceedances
    }