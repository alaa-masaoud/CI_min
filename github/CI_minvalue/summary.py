from __future__ import annotations

import numpy as np
import pandas as pd

from distributions import DistributionSpec
from evt.estimator_registry import EstimatorSpec


def _summary_row(
    distribution: DistributionSpec,
    method: str,
    k: int,
    alpha_values: pd.Series,
    coverage_values: pd.Series,
    length_values: pd.Series,
) -> dict[str, object]:
    alpha_values = pd.to_numeric(alpha_values, errors="coerce").dropna()
    coverage_values = pd.to_numeric(coverage_values, errors="coerce").dropna()
    length_values = pd.to_numeric(length_values, errors="coerce").dropna()

    mean_alpha = float(alpha_values.mean()) if not alpha_values.empty else np.nan
    return {
        "Distribution": distribution.name,
        "k": int(k),
        "Method": method,
        "True Alpha": distribution.true_alpha,
        "Valid Alpha Estimates": int(alpha_values.size),
        "Mean Estimated Alpha": mean_alpha,
        "Alpha Bias": (
            mean_alpha - distribution.true_alpha
            if np.isfinite(mean_alpha)
            else np.nan
        ),
        "Alpha Variance": (
            float(alpha_values.var(ddof=1))
            if alpha_values.size >= 2
            else np.nan
        ),
        "Valid CIs": int(coverage_values.size),
        "Coverage Proportion": (
            float(coverage_values.mean())
            if not coverage_values.empty
            else np.nan
        ),
        "Average CI Length": (
            float(length_values.mean())
            if not length_values.empty
            else np.nan
        ),
    }


def _constant_k(results: pd.DataFrame, column: str) -> int:
    values = pd.to_numeric(results[column], errors="coerce").dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"Expected exactly one selected k in {column!r}, found {values}."
        )
    return int(values[0])


def build_summary(
    distribution: DistributionSpec,
    estimators: list[EstimatorSpec],
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Build the same summary format, now using each method's selected k."""
    rows: list[dict[str, object]] = []

    for index, estimator in enumerate(estimators):
        name = estimator.display_name
        k_column = "k" if index == 0 else f"k ({name})"
        rows.append(_summary_row(
            distribution=distribution,
            method=name,
            k=_constant_k(results, k_column),
            alpha_values=results[f"alpha ({name})"],
            coverage_values=results[f"P (1 or 0) ({name})"],
            length_values=results[f"(Upper - Lower) ({name})"],
        ))

    rows.append(_summary_row(
        distribution=distribution,
        method="True Alpha Benchmark",
        k=_constant_k(results, "k (True Alpha)"),
        alpha_values=pd.Series(dtype=float),
        coverage_values=results["P (1 or 0) (True Alpha)"],
        length_values=results["(Upper - Lower) (True Alpha)"],
    ))

    return pd.DataFrame(rows)