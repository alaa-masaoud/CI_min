from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable

import numpy as np
import pandas as pd

from confidence_intervals import estimate_minimum_ci
from config import (
    CONFIDENCE_LEVEL,
    CONTINUE_AFTER_ESTIMATOR_ERROR,
    FULL_SAMPLE_SIZE,
    K_VALUES,
    N_REPETITIONS,
    PROGRESS_EVERY,
    SUBSAMPLE_SIZE,
)
from distributions import DistributionSpec
from evt.estimator_adapter import (
    call_estimator,
    load_estimators,
    prepare_input,
    result_to_alpha,
)
from evt.estimator_registry import EstimatorSpec
from evt.transformations import weibull_lower_to_frechet_upper
from k_selection import (
    KSelectionSettings,
    build_k_diagnostics,
    collapse_to_optimal_k_format,
    select_optimal_k,
)


N_ENDPOINT_CANDIDATES = 60


def result_columns(estimators: list[EstimatorSpec]) -> list[str]:
    columns = ["Repetition", "k"]
    for spec in estimators:
        columns.extend([
            f"alpha ({spec.display_name})",
            f"CI Lower ({spec.display_name})",
            f"CI Upper ({spec.display_name})",
            f"P (1 or 0) ({spec.display_name})",
            f"(Upper - Lower) ({spec.display_name})",
        ])
    columns.extend([
        "True Alpha",
        "CI Lower (True Alpha)",
        "CI Upper (True Alpha)",
        "P (1 or 0) (True Alpha)",
        "(Upper - Lower) (True Alpha)",
        "True Min",
    ])
    return columns


def _generate_endpoint_candidates(
    subsample: np.ndarray,
    k: int,
    n_candidates: int = N_ENDPOINT_CANDIDATES,
) -> np.ndarray:
    """
    Generate possible lower endpoints using only the subsample.

    Every candidate is strictly below the observed subsample minimum.
    The range is scaled by the lower-tail spacing X_(k+1:n)-X_(1:n).
    """
    values = np.sort(np.asarray(subsample, dtype=float))
    n = values.size

    if values.ndim != 1 or n < 3:
        raise ValueError("At least three one-dimensional observations are required.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Subsample contains non-finite values.")
    if not 1 <= k < n:
        raise ValueError(f"k must satisfy 1 <= k < {n}.")

    sample_minimum = float(values[0])
    tail_spacing = float(values[k] - sample_minimum)

    if not np.isfinite(tail_spacing) or tail_spacing <= 0.0:
        positive_spacings = values[values > sample_minimum] - sample_minimum
        if positive_spacings.size == 0:
            raise ValueError("Cannot generate endpoint candidates from constant data.")
        tail_spacing = float(np.median(positive_spacings))

    # Geometric spacing gives many candidates close to the sample minimum,
    # while still checking endpoints farther below it.
    gap_factors = np.geomspace(1.0e-3, 5.0, num=n_candidates)
    candidates = sample_minimum - gap_factors * tail_spacing
    candidates = candidates[
        np.isfinite(candidates) & (candidates < sample_minimum)
    ]

    candidates = np.unique(candidates)

    if candidates.size == 0:
        raise RuntimeError("No valid endpoint candidates were generated.")

    return candidates


def _pareto_ks_statistic(
    frechet_data: np.ndarray,
    alpha: float,
    k: int,
) -> float:
    """
    KS distance between the largest k transformed observations and
    a fitted Pareto(alpha) upper tail.
    """
    values = np.sort(np.asarray(frechet_data, dtype=float))

    if values.ndim != 1 or values.size <= k:
        return float("inf")
    if not np.all(np.isfinite(values)):
        return float("inf")
    if not np.isfinite(alpha) or alpha <= 0.0:
        return float("inf")

    threshold = float(values[-k - 1])
    if threshold <= 0.0 or not np.isfinite(threshold):
        return float("inf")

    ratios = np.sort(values[-k:] / threshold)

    if (
        ratios.size != k
        or np.any(~np.isfinite(ratios))
        or np.any(ratios < 1.0)
    ):
        return float("inf")

    fitted_cdf = 1.0 - np.power(ratios, -alpha)
    empirical_right = np.arange(1, k + 1, dtype=float) / k
    empirical_left = np.arange(0, k, dtype=float) / k

    d_right = np.max(np.abs(empirical_right - fitted_cdf))
    d_left = np.max(np.abs(empirical_left - fitted_cdf))

    return float(max(d_right, d_left))


def _estimate_endpoint_and_alpha(
    subsample: np.ndarray,
    spec: EstimatorSpec,
    estimator_function: Callable[..., Any],
    k: int,
) -> tuple[float, float]:
    """
    Select the endpoint separately for one estimator and one k.

    Each endpoint candidate is used in the transformation. The estimator
    produces alpha, and the endpoint-alpha pair with the smallest Pareto
    KS distance is retained.
    """
    best_endpoint: float | None = None
    best_alpha: float | None = None
    best_ks = float("inf")

    candidates = _generate_endpoint_candidates(subsample, k=k)

    for endpoint in candidates:
        try:
            frechet_data = weibull_lower_to_frechet_upper(
                subsample,
                lower_endpoint=float(endpoint),
            )

            estimator_input = prepare_input(
                frechet_data=frechet_data,
                spec=spec,
                k=k,
            )

            raw_result = call_estimator(
                function=estimator_function,
                spec=spec,
                estimator_input=estimator_input,
                k=k,
            )

            alpha_hat = result_to_alpha(raw_result, spec)

            ks_statistic = _pareto_ks_statistic(
                frechet_data=frechet_data,
                alpha=alpha_hat,
                k=k,
            )

            if np.isfinite(ks_statistic) and ks_statistic < best_ks:
                best_endpoint = float(endpoint)
                best_alpha = float(alpha_hat)
                best_ks = float(ks_statistic)

        except Exception:
            # Some candidates may make a particular estimator fail.
            # Continue until all candidates have been checked.
            continue

    if best_endpoint is None or best_alpha is None:
        raise RuntimeError(
            "No endpoint candidate produced a valid alpha estimate "
            "and finite Pareto KS statistic."
        )

    return best_endpoint, best_alpha


def _estimate_endpoint_for_fixed_alpha(
    subsample: np.ndarray,
    alpha: float,
    k: int,
) -> float:
    """
    Select an endpoint for the true-alpha benchmark without supplying
    the known theoretical endpoint.
    """
    best_endpoint: float | None = None
    best_ks = float("inf")

    candidates = _generate_endpoint_candidates(subsample, k=k)

    for endpoint in candidates:
        try:
            frechet_data = weibull_lower_to_frechet_upper(
                subsample,
                lower_endpoint=float(endpoint),
            )

            ks_statistic = _pareto_ks_statistic(
                frechet_data=frechet_data,
                alpha=alpha,
                k=k,
            )

            if np.isfinite(ks_statistic) and ks_statistic < best_ks:
                best_endpoint = float(endpoint)
                best_ks = float(ks_statistic)

        except Exception:
            continue

    if best_endpoint is None:
        raise RuntimeError(
            "No endpoint candidate produced a finite Pareto KS statistic "
            "for the true-alpha benchmark."
        )

    return best_endpoint


def _candidate_result_columns(estimators: list[EstimatorSpec]) -> list[str]:
    """Internal all-k format used only while selecting k."""
    columns = ["Repetition", "k"]
    for spec in estimators:
        columns.extend([
            f"alpha ({spec.display_name})",
            f"CI Lower ({spec.display_name})",
            f"CI Upper ({spec.display_name})",
            f"P (1 or 0) ({spec.display_name})",
            f"(Upper - Lower) ({spec.display_name})",
        ])
    columns.extend([
        "True Alpha",
        "CI Lower (True Alpha)",
        "CI Upper (True Alpha)",
        "P (1 or 0) (True Alpha)",
        "(Upper - Lower) (True Alpha)",
        "True Min",
    ])
    return columns


def _run_one_repetition(
    repetition: int,
    repetition_seed: np.random.SeedSequence,
    distribution: DistributionSpec,
    estimators: list[EstimatorSpec],
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one complete repetition inside a worker process."""
    candidate_columns = _candidate_result_columns(estimators)
    candidate_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    # Load estimator callables inside the worker. This avoids Windows
    # multiprocessing pickle errors caused by passing function objects.
    estimator_functions = load_estimators(estimators)

    # Each repetition receives a deterministic child seed spawned from
    # MASTER_SEED by main.py. Parallel execution does not change this seed.
    rng = np.random.default_rng(repetition_seed)

    full_sample = np.asarray(
        distribution.sampler(rng, FULL_SAMPLE_SIZE),
        dtype=float,
    )
    if full_sample.shape != (FULL_SAMPLE_SIZE,):
        raise ValueError(
            f"Expected ({FULL_SAMPLE_SIZE},), got {full_sample.shape}."
        )
    if not np.all(np.isfinite(full_sample)):
        raise ValueError("Generated sample contains non-finite values.")

    true_minimum = float(np.min(full_sample))
    indices = rng.choice(
        FULL_SAMPLE_SIZE,
        size=SUBSAMPLE_SIZE,
        replace=False,
    )
    subsample = full_sample[indices]

    for k in K_VALUES:
        row: dict[str, Any] = {
            column: np.nan for column in candidate_columns
        }
        row["Repetition"] = repetition
        row["k"] = k
        row["True Alpha"] = distribution.true_alpha
        row["True Min"] = true_minimum

        for spec in estimators:
            name = spec.display_name

            try:
                endpoint_hat, alpha_hat = _estimate_endpoint_and_alpha(
                    subsample=subsample,
                    spec=spec,
                    estimator_function=estimator_functions[name],
                    k=k,
                )

                lower, upper = estimate_minimum_ci(
                    subsample=subsample,
                    endpoint=endpoint_hat,
                    alpha=alpha_hat,
                    k=k,
                    full_sample_size=FULL_SAMPLE_SIZE,
                    confidence_level=CONFIDENCE_LEVEL,
                )

                row[f"alpha ({name})"] = alpha_hat
                row[f"CI Lower ({name})"] = lower
                row[f"CI Upper ({name})"] = upper
                row[f"P (1 or 0) ({name})"] = int(
                    lower <= true_minimum <= upper
                )
                row[f"(Upper - Lower) ({name})"] = upper - lower

            except Exception as error:
                errors.append({
                    "Distribution": distribution.name,
                    "Repetition": repetition,
                    "k": k,
                    "Estimator": name,
                    "Error": str(error),
                })
                if not CONTINUE_AFTER_ESTIMATOR_ERROR:
                    raise

        try:
            true_alpha_endpoint = _estimate_endpoint_for_fixed_alpha(
                subsample=subsample,
                alpha=distribution.true_alpha,
                k=k,
            )

            lower, upper = estimate_minimum_ci(
                subsample=subsample,
                endpoint=true_alpha_endpoint,
                alpha=distribution.true_alpha,
                k=k,
                full_sample_size=FULL_SAMPLE_SIZE,
                confidence_level=CONFIDENCE_LEVEL,
            )

            row["CI Lower (True Alpha)"] = lower
            row["CI Upper (True Alpha)"] = upper
            row["P (1 or 0) (True Alpha)"] = int(
                lower <= true_minimum <= upper
            )
            row["(Upper - Lower) (True Alpha)"] = upper - lower

        except Exception as error:
            errors.append({
                "Distribution": distribution.name,
                "Repetition": repetition,
                "k": k,
                "Estimator": "True Alpha",
                "Error": str(error),
            })
            if not CONTINUE_AFTER_ESTIMATOR_ERROR:
                raise

        candidate_rows.append(row)

    return repetition, candidate_rows, errors


def run_distribution(
    distribution: DistributionSpec,
    estimators: list[EstimatorSpec],
    estimator_functions: dict[str, Callable[..., Any]],
    seed_sequence: np.random.SeedSequence,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    """
    Run independent repetitions in parallel, test every configured k,
    and select one optimal k per estimator.

    estimator_functions remains in the signature so main.py does not need
    to change. Worker processes load their own estimator callables.
    """
    del estimator_functions  # Loaded independently inside each worker.

    candidate_columns = _candidate_result_columns(estimators)
    candidate_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    # These child seeds are determined only by the supplied seed_sequence.
    # The same MASTER_SEED therefore gives the same seed for each repetition,
    # regardless of worker count or completion order.
    repetition_seeds = seed_sequence.spawn(N_REPETITIONS)
    start_time = time.perf_counter()

    configured_workers = int(os.environ.get("EVT_MAX_WORKERS", "0"))
    if configured_workers > 0:
        max_workers = min(configured_workers, N_REPETITIONS)
    else:
        # Leave one logical CPU available for Windows and other programs.
        max_workers = min(max(1, (os.cpu_count() or 2) - 1), N_REPETITIONS)

    print(f"{distribution.name}: using {max_workers} worker processes")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_one_repetition,
                repetition,
                repetition_seed,
                distribution,
                estimators,
            ): repetition
            for repetition, repetition_seed in enumerate(
                repetition_seeds,
                start=1,
            )
        }

        completed = 0
        for future in as_completed(futures):
            repetition = futures[future]
            try:
                _, rows, repetition_errors = future.result()
            except Exception as error:
                raise RuntimeError(
                    f"{distribution.name}: repetition {repetition} failed."
                ) from error

            candidate_rows.extend(rows)
            errors.extend(repetition_errors)
            completed += 1

            if (
                completed == 1
                or completed % PROGRESS_EVERY == 0
                or completed == N_REPETITIONS
            ):
                elapsed = time.perf_counter() - start_time
                print(
                    f"{distribution.name}: {completed}/{N_REPETITIONS} "
                    f"repetitions completed; elapsed={elapsed:.2f}s"
                )

    candidate_results = pd.DataFrame(
        candidate_rows,
        columns=candidate_columns,
    )

    # Worker jobs finish in arbitrary order. Sorting restores the exact
    # serial-style order and makes output reproducible across worker counts.
    candidate_results = candidate_results.sort_values(
        ["Repetition", "k"],
        kind="stable",
    ).reset_index(drop=True)

    diagnostics = build_k_diagnostics(
        candidate_results=candidate_results,
        estimators=estimators,
        true_alpha=distribution.true_alpha,
        settings=KSelectionSettings(target_coverage=CONFIDENCE_LEVEL),
    )
    optimal_k = select_optimal_k(diagnostics)

    for method, chosen_k in optimal_k.items():
        print(f"{distribution.name}: optimal k for {method} = {chosen_k}")

    optimal_results = collapse_to_optimal_k_format(
        candidate_results=candidate_results,
        estimators=estimators,
        optimal_k=optimal_k,
    )

    diagnostics = diagnostics.copy()
    diagnostics["Selected"] = diagnostics.apply(
        lambda row: int(optimal_k[str(row["Method"])] == int(row["k"])),
        axis=1,
    )

    return optimal_results, errors, diagnostics