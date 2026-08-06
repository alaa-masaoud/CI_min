import numpy as np
from scipy.optimize import brentq


def zhang_lme_estimator(
    exceedances,
    r=-0.5,
    grid_size=400,
    root_tolerance=1e-10,
):
   
    x = np.asarray(exceedances, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x >= 0.0]

    if len(x) < 5:
        raise ValueError(
            "Need at least 5 finite, nonnegative exceedances."
        )

    if np.max(x) <= 0.0:
        raise ValueError(
            "Exceedances must not all be zero."
        )

    if not np.isfinite(r) or r == 0.0 or r >= 0.5:
        raise ValueError(
            "Zhang LME requires a finite tuning parameter r < 0.5 "
            "with r != 0."
        )

    x_max = float(np.max(x))
    x_mean = float(np.mean(x))

    if x_mean <= 0.0:
        raise ValueError(
            "Mean exceedance must be positive."
        )

    target = 1.0 / (1.0 - r)

    def xi_from_theta(theta):
        """
        Profile estimate

            xi(theta) = mean(log(1 + theta*x)).
        """
        support = 1.0 + theta * x

        if np.any(support <= 0.0):
            return np.nan

        return float(np.mean(np.log1p(theta * x)))

    def estimating_equation(theta):
        """
        Profiled Zhang likelihood-moment equation.
        """
        support = 1.0 + theta * x

        if np.any(support <= 0.0):
            return np.nan

        log_support = np.log1p(theta * x)
        xi = float(np.mean(log_support))

        # theta = 0 implies xi = 0, producing a removable 0/0 form.
        # We avoid evaluating directly at zero and search on either side.
        if not np.isfinite(xi) or abs(xi) < 1e-14:
            return np.nan

        exponent = r * log_support / xi

        # Prevent numerical overflow while preserving the estimating
        # equation over the practically relevant parameter range.
        exponent = np.clip(exponent, -745.0, 700.0)

        moment = float(np.mean(np.exp(exponent)))

        if not np.isfinite(moment):
            return np.nan

        return moment - target

    def find_brackets(grid):
        """
        Return intervals on which the estimating equation changes sign.
        """
        brackets = []

        previous_theta = None
        previous_value = None

        for theta in grid:
            value = estimating_equation(float(theta))

            if not np.isfinite(value):
                continue

            if abs(value) <= root_tolerance:
                brackets.append((float(theta), float(theta)))

            elif (
                previous_value is not None
                and np.sign(value) != np.sign(previous_value)
            ):
                brackets.append(
                    (float(previous_theta), float(theta))
                )

            previous_theta = float(theta)
            previous_value = float(value)

        return brackets

    # The support restriction is
    #
    #     theta > -1 / max(x).
    #
    # Search both negative and positive theta so the implementation can
    # estimate negative, zero-adjacent, and positive GPD shape parameters.
    negative_limit = -1.0 / x_max

    theta_epsilon = max(
        np.finfo(float).eps / x_mean,
        1e-12 / x_mean,
    )

    # Negative grid approaches the support boundary but never touches it.
    negative_magnitude_max = (
        (1.0 - 1e-10) / x_max
    )

    negative_grid = -np.geomspace(
        theta_epsilon,
        negative_magnitude_max,
        num=grid_size,
    )
    negative_grid = np.sort(negative_grid)

    # Positive theta has no finite support boundary. A wide logarithmic
    # grid is used, scaled by the sample mean.
    positive_grid = np.geomspace(
        theta_epsilon,
        1e6 / x_mean,
        num=grid_size,
    )

    brackets = (
        find_brackets(negative_grid)
        + find_brackets(positive_grid)
    )

    roots = []

    for lower, upper in brackets:
        if lower == upper:
            theta_hat = lower
        else:
            try:
                theta_hat = brentq(
                    estimating_equation,
                    lower,
                    upper,
                    xtol=root_tolerance,
                    rtol=4.0 * np.finfo(float).eps,
                    maxiter=1000,
                )
            except (ValueError, RuntimeError, FloatingPointError):
                continue

        xi_hat = xi_from_theta(theta_hat)

        if (
            not np.isfinite(theta_hat)
            or not np.isfinite(xi_hat)
            or abs(theta_hat) < np.finfo(float).eps
        ):
            continue

        beta_hat = xi_hat / theta_hat

        if (
            not np.isfinite(beta_hat)
            or beta_hat <= 0.0
            or np.any(1.0 + theta_hat * x <= 0.0)
        ):
            continue

        residual = abs(estimating_equation(theta_hat))

        roots.append(
            {
                "theta": float(theta_hat),
                "xi": float(xi_hat),
                "beta": float(beta_hat),
                "residual": float(residual),
            }
        )

    if not roots:
        raise ValueError(
            "Could not find an admissible root of Zhang's "
            "likelihood-moment estimating equation."
        )

    # The theoretical equation has a unique root under Zhang's stated
    # conditions. If finite-precision scanning produces duplicates or
    # additional numerical candidates, choose the root with the smallest
    # estimating-equation residual.
    best = min(
        roots,
        key=lambda result: result["residual"],
    )

    return best["xi"], best["beta"]