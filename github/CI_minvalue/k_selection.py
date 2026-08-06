from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evt.estimator_registry import EstimatorSpec


@dataclass(frozen=True)
class KSelectionSettings:
    """Rules used to choose one k value for each estimator."""

    target_coverage: float
    undercoverage_weight: float = 2.0
    overcoverage_weight: float = 1.0


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _coverage_loss(
    coverage: float,
    target: float,
    undercoverage_weight: float,
    overcoverage_weight: float,
) -> float:
    if not np.isfinite(coverage):
        return float("inf")
    if coverage < target:
        return undercoverage_weight * (target - coverage)
    return overcoverage_weight * (coverage - target)


def build_k_diagnostics(
    candidate_results: pd.DataFrame,
    estimators: list[EstimatorSpec],
    true_alpha: float,
    settings: KSelectionSettings,
) -> pd.DataFrame:
    """
    Summarize every estimator-k combination before choosing the optimum.

    The selection is deliberately transparent. The winning row is chosen by:
      1. most valid confidence intervals,
      2. smallest weighted distance from target coverage,
      3. shortest average confidence interval,
      4. smallest absolute alpha bias,
      5. smallest alpha variance,
      6. smaller k as the final deterministic tie-break.
    """
    rows: list[dict[str, object]] = []
    k_values = sorted(
        int(k)
        for k in _numeric(candidate_results["k"]).dropna().unique()
    )

    for spec in estimators:
        name = spec.display_name
        for k in k_values:
            subset = candidate_results[_numeric(candidate_results["k"]) == k]
            alpha = _numeric(subset[f"alpha ({name})"]).dropna()
            coverage = _numeric(subset[f"P (1 or 0) ({name})"]).dropna()
            length = _numeric(subset[f"(Upper - Lower) ({name})"]).dropna()

            mean_alpha = float(alpha.mean()) if not alpha.empty else np.nan
            coverage_rate = float(coverage.mean()) if not coverage.empty else np.nan
            average_length = float(length.mean()) if not length.empty else np.nan
            alpha_bias = mean_alpha - true_alpha if np.isfinite(mean_alpha) else np.nan
            alpha_variance = (
                float(alpha.var(ddof=1)) if alpha.size >= 2 else np.nan
            )

            rows.append({
                "Method": name,
                "k": k,
                "Valid Alpha Estimates": int(alpha.size),
                "Valid CIs": int(coverage.size),
                "Coverage Proportion": coverage_rate,
                "Coverage Loss": _coverage_loss(
                    coverage_rate,
                    settings.target_coverage,
                    settings.undercoverage_weight,
                    settings.overcoverage_weight,
                ),
                "Average CI Length": average_length,
                "Mean Estimated Alpha": mean_alpha,
                "Alpha Bias": alpha_bias,
                "Absolute Alpha Bias": (
                    abs(alpha_bias) if np.isfinite(alpha_bias) else np.inf
                ),
                "Alpha Variance": alpha_variance,
            })

    # Select k for the true-alpha benchmark as well.
    for k in k_values:
        subset = candidate_results[_numeric(candidate_results["k"]) == k]
        coverage = _numeric(subset["P (1 or 0) (True Alpha)"]).dropna()
        length = _numeric(subset["(Upper - Lower) (True Alpha)"]).dropna()
        coverage_rate = float(coverage.mean()) if not coverage.empty else np.nan

        rows.append({
            "Method": "True Alpha Benchmark",
            "k": k,
            "Valid Alpha Estimates": 0,
            "Valid CIs": int(coverage.size),
            "Coverage Proportion": coverage_rate,
            "Coverage Loss": _coverage_loss(
                coverage_rate,
                settings.target_coverage,
                settings.undercoverage_weight,
                settings.overcoverage_weight,
            ),
            "Average CI Length": (
                float(length.mean()) if not length.empty else np.nan
            ),
            "Mean Estimated Alpha": np.nan,
            "Alpha Bias": np.nan,
            "Absolute Alpha Bias": np.inf,
            "Alpha Variance": np.nan,
        })

    return pd.DataFrame(rows)


def select_optimal_k(diagnostics: pd.DataFrame) -> dict[str, int]:
    """Return one optimal k for every estimator and the true-alpha benchmark."""
    selected: dict[str, int] = {}

    for method, subset in diagnostics.groupby("Method", sort=False):
        ranked = subset.copy()
        ranked["Average CI Length Rank"] = pd.to_numeric(
            ranked["Average CI Length"], errors="coerce"
        ).fillna(np.inf)
        ranked["Alpha Variance Rank"] = pd.to_numeric(
            ranked["Alpha Variance"], errors="coerce"
        ).fillna(np.inf)

        ranked = ranked.sort_values(
            by=[
                "Valid CIs",
                "Coverage Loss",
                "Average CI Length Rank",
                "Absolute Alpha Bias",
                "Alpha Variance Rank",
                "k",
            ],
            ascending=[False, True, True, True, True, True],
            kind="mergesort",
        )

        if ranked.empty:
            raise RuntimeError(f"No k candidates were available for {method}.")

        selected[str(method)] = int(ranked.iloc[0]["k"])

    return selected


def optimal_result_columns(estimators: list[EstimatorSpec]) -> list[str]:
    """
    Preserve the user's wide output format while placing a k column before
    every estimator block.
    """
    columns = ["Repetition"]

    for index, spec in enumerate(estimators):
        name = spec.display_name
        columns.append("k" if index == 0 else f"k ({name})")
        columns.extend([
            f"alpha ({name})",
            f"CI Lower ({name})",
            f"CI Upper ({name})",
            f"P (1 or 0) ({name})",
            f"(Upper - Lower) ({name})",
        ])

    columns.extend([
        "k (True Alpha)",
        "True Alpha",
        "CI Lower (True Alpha)",
        "CI Upper (True Alpha)",
        "P (1 or 0) (True Alpha)",
        "(Upper - Lower) (True Alpha)",
        "True Min",
    ])
    return columns


def collapse_to_optimal_k_format(
    candidate_results: pd.DataFrame,
    estimators: list[EstimatorSpec],
    optimal_k: dict[str, int],
) -> pd.DataFrame:
    """
    Convert the all-k internal table into one row per repetition.

    Every estimator block is taken from the row matching that estimator's own
    selected k. This is why k can differ across estimator blocks in one row.
    """
    columns = optimal_result_columns(estimators)
    rows: list[dict[str, object]] = []

    repetitions = sorted(
        int(r)
        for r in _numeric(candidate_results["Repetition"]).dropna().unique()
    )

    for repetition in repetitions:
        repetition_rows = candidate_results[
            _numeric(candidate_results["Repetition"]) == repetition
        ]
        output: dict[str, object] = {column: np.nan for column in columns}
        output["Repetition"] = repetition

        for index, spec in enumerate(estimators):
            name = spec.display_name
            chosen_k = optimal_k[name]
            chosen = repetition_rows[
                _numeric(repetition_rows["k"]) == chosen_k
            ]
            if chosen.empty:
                continue

            source = chosen.iloc[0]
            output["k" if index == 0 else f"k ({name})"] = chosen_k
            for metric in (
                "alpha",
                "CI Lower",
                "CI Upper",
                "P (1 or 0)",
                "(Upper - Lower)",
            ):
                output[f"{metric} ({name})"] = source[f"{metric} ({name})"]

        true_k = optimal_k["True Alpha Benchmark"]
        true_rows = repetition_rows[_numeric(repetition_rows["k"]) == true_k]
        if not true_rows.empty:
            source = true_rows.iloc[0]
            output["k (True Alpha)"] = true_k
            output["True Alpha"] = source["True Alpha"]
            output["CI Lower (True Alpha)"] = source["CI Lower (True Alpha)"]
            output["CI Upper (True Alpha)"] = source["CI Upper (True Alpha)"]
            output["P (1 or 0) (True Alpha)"] = source[
                "P (1 or 0) (True Alpha)"
            ]
            output["(Upper - Lower) (True Alpha)"] = source[
                "(Upper - Lower) (True Alpha)"
            ]
            output["True Min"] = source["True Min"]

        rows.append(output)

    return pd.DataFrame(rows, columns=columns)