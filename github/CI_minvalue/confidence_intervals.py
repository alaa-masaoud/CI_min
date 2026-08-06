from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _fit_lower_tail_gls(
    subsample: np.ndarray,
    alpha: float,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit the lower-tail quantile model

        Q(p) = m + a * p**(1/alpha)

    to the first k order statistics using generalized least squares.

    Unlike ordinary least squares, this uses the approximate covariance
    structure of order statistics. This matters because the first k order
    statistics are strongly correlated, and treating them as independent
    produces intervals that are much too narrow.

    Returns
    -------
    beta:
        Array [m_hat, a_hat].
    covariance:
        Approximate covariance matrix of [m_hat, a_hat].
    """
    values = np.sort(np.asarray(subsample, dtype=float))
    n = values.size

    if values.ndim != 1 or n < 4:
        raise ValueError(
            "At least four one-dimensional observations are required."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Subsample contains non-finite values.")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be positive and finite.")
    if not 4 <= k < n:
        raise ValueError("k must satisfy 4 <= k < len(subsample).")

    lower_order_statistics = values[:k]

    # Plotting positions for X_(1), ..., X_(k).
    probabilities = (
        np.arange(1, k + 1, dtype=float) / (n + 1.0)
    )

    transformed_probabilities = (
        probabilities ** (1.0 / alpha)
    )

    design = np.column_stack(
        [
            np.ones(k, dtype=float),
            transformed_probabilities,
        ]
    )

    # Initial OLS fit is used only to estimate the scale needed in the
    # order-statistic covariance matrix.
    initial_beta, *_ = np.linalg.lstsq(
        design,
        lower_order_statistics,
        rcond=None,
    )

    initial_scale = float(initial_beta[1])

    if not np.isfinite(initial_scale) or initial_scale <= 0.0:
        raise ValueError(
            "Initial lower-tail scale estimate must be positive."
        )

    # Under Q(p) = m + a p^(1/alpha),
    #
    #     f(Q(p)) = alpha/a * p^(1 - 1/alpha).
    #
    # For order statistics,
    #
    # Cov(X_(i), X_(j))
    #   approximately
    # [min(p_i,p_j)-p_i p_j]
    # / [n f(Q(p_i)) f(Q(p_j))].
    fitted_density = (
        alpha
        / initial_scale
        * probabilities ** (1.0 - 1.0 / alpha)
    )

    probability_covariance = (
        np.minimum.outer(probabilities, probabilities)
        - np.outer(probabilities, probabilities)
    )

    order_statistic_covariance = (
        probability_covariance
        / (
            n
            * np.outer(fitted_density, fitted_density)
        )
    )

    # A tiny diagonal ridge prevents numerical singularity without
    # materially changing the result.
    ridge = max(
        1.0e-14,
        float(np.trace(order_statistic_covariance))
        * 1.0e-12
        / k,
    )

    order_statistic_covariance = (
        order_statistic_covariance
        + ridge * np.eye(k)
    )

    inverse_order_covariance = np.linalg.pinv(
        order_statistic_covariance
    )

    information = (
        design.T
        @ inverse_order_covariance
        @ design
    )

    coefficient_covariance = np.linalg.pinv(information)

    beta = (
        coefficient_covariance
        @ design.T
        @ inverse_order_covariance
        @ lower_order_statistics
    )

    endpoint_hat = float(beta[0])
    scale_hat = float(beta[1])

    if not np.isfinite(endpoint_hat):
        raise ValueError(
            "GLS endpoint estimate is not finite."
        )
    if not np.isfinite(scale_hat) or scale_hat <= 0.0:
        raise ValueError(
            "GLS lower-tail scale estimate must be positive."
        )
    if not np.all(np.isfinite(coefficient_covariance)):
        raise ValueError(
            "GLS coefficient covariance is not finite."
        )

    return beta, coefficient_covariance


def estimate_minimum_ci(
    subsample: np.ndarray,
    alpha: float,
    k: int,
    full_sample_size: int,
    confidence_level: float,
    endpoint: float | None = None,
) -> tuple[float, float]:
    """
    Approximate confidence/prediction interval for the minimum of the
    original full sample.

    The subsample contains n observations from the full sample of size N.
    Therefore only r = N-n unseen observations can produce a value below
    the observed subsample minimum.

    The lower tail is modeled as

        Q(p) = m + a * p**(1/alpha).

    The endpoint m and scale a are fitted from the first k order statistics
    by GLS. Their uncertainty is included when forming the bounds.

    The optional `endpoint` argument is accepted for compatibility with the
    existing simulation framework. It is deliberately not used here: the
    candidate endpoint selected for the Pareto transformation is used to
    estimate alpha, while this CI independently estimates the original
    lower-tail endpoint and scale from the subsample.

    No bootstrap and no known theoretical endpoint are used.
    """
    values = np.sort(np.asarray(subsample, dtype=float))
    n = values.size

    if values.ndim != 1 or n < 4:
        raise ValueError(
            "At least four one-dimensional observations are required."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Subsample contains non-finite values.")
    if full_sample_size < n:
        raise ValueError(
            "full_sample_size cannot be smaller than the subsample."
        )
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must lie strictly between 0 and 1."
        )

    beta, beta_covariance = _fit_lower_tail_gls(
        subsample=values,
        alpha=alpha,
        k=k,
    )

    unseen_count = full_sample_size - n
    sample_minimum = float(values[0])

    if unseen_count == 0:
        return sample_minimum, sample_minimum

    tail_probability = 1.0 - confidence_level

    minimum_probabilities = (
        tail_probability / 2.0,
        1.0 - tail_probability / 2.0,
    )

    critical_value = float(
        norm.ppf(1.0 - tail_probability / 2.0)
    )

    quantile_intervals: list[tuple[float, float]] = []

    for minimum_probability in minimum_probabilities:
        # If U is the minimum of r unseen observations, then
        #
        # P(U <= x) = 1 - (1-F(x))^r.
        #
        # Solving for F(x) gives the single-observation probability below.
        observation_probability = -np.expm1(
            np.log1p(-minimum_probability)
            / unseen_count
        )

        transformed_probability = (
            observation_probability ** (1.0 / alpha)
        )

        prediction_vector = np.array(
            [1.0, transformed_probability],
            dtype=float,
        )

        fitted_quantile = float(
            prediction_vector @ beta
        )

        fitted_quantile_variance = float(
            prediction_vector
            @ beta_covariance
            @ prediction_vector
        )

        fitted_quantile_variance = max(
            fitted_quantile_variance,
            0.0,
        )

        standard_error = float(
            np.sqrt(fitted_quantile_variance)
        )

        quantile_intervals.append(
            (
                fitted_quantile
                - critical_value * standard_error,
                fitted_quantile
                + critical_value * standard_error,
            )
        )

    # Use the lower confidence limit for the 2.5% minimum quantile and
    # the upper confidence limit for the 97.5% minimum quantile.
    lower = float(quantile_intervals[0][0])
    upper = float(quantile_intervals[1][1])

    # Since the subsample is part of the full sample, the full-sample
    # minimum cannot be greater than the observed subsample minimum.
    upper = min(upper, sample_minimum)

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise RuntimeError(
            "The fitted interval contains non-finite bounds."
        )
    if lower > upper:
        raise RuntimeError(
            "The fitted interval is invalid because lower exceeds upper."
        )

    return lower, upper