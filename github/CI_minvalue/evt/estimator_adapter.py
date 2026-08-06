from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np

from evt.estimator_registry import EstimatorSpec
from evt.transformations import upper_tail_exceedances


def load_estimators(
    estimators: list[EstimatorSpec],
) -> dict[str, Callable[..., Any]]:
    loaded = {}

    for spec in estimators:
        module = importlib.import_module(spec.module_name)
        function = getattr(module, spec.function_name, None)

        if not callable(function):
            raise AttributeError(
                f"{spec.module_name}.{spec.function_name} was not found."
            )

        loaded[spec.display_name] = function
        print(
            f"Loaded {spec.display_name}: "
            f"{spec.module_name}.{spec.function_name}"
        )

    return loaded


def prepare_input(
    frechet_data: np.ndarray,
    spec: EstimatorSpec,
    k: int,
) -> np.ndarray:
    if spec.input_type in {
        "frechet_data",
        "frechet_data_internal_threshold",
    }:
        return np.asarray(frechet_data, dtype=float).copy()

    if spec.input_type == "frechet_exceedances":
        exceedances, _ = upper_tail_exceedances(
            frechet_data=frechet_data,
            k=k,
        )
        return exceedances

    raise ValueError(f"Unknown input type: {spec.input_type}")


def call_estimator(
    function: Callable[..., Any],
    spec: EstimatorSpec,
    estimator_input: np.ndarray,
    k: int,
) -> Any:
    if spec.display_name == "GPD MLE":
        # Match approximately k upper observations out of n.
        n = len(estimator_input)
        threshold_quantile = 1.0 - (k / n)
        return function(
            estimator_input,
            threshold_quantile=threshold_quantile,
        )

    if spec.uses_k:
        return function(estimator_input, k=k)

    return function(estimator_input)


def tail_index_to_alpha(tail_index: float) -> float:
    gamma_or_xi = float(tail_index)

    if not np.isfinite(gamma_or_xi) or gamma_or_xi <= 0.0:
        raise ValueError(
            f"Tail-index estimate must be positive; got {gamma_or_xi}."
        )

    alpha = 1.0 / gamma_or_xi

    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"Invalid alpha estimate: {alpha}.")

    return float(alpha)


def result_to_alpha(
    result: Any,
    spec: EstimatorSpec,
) -> float:
    """
    All uploaded estimators estimate gamma or xi, not alpha directly.
    For transformed Frechet data:

        alpha_hat = 1 / gamma_hat = 1 / xi_hat
    """
    if spec.output_type == "tail_index":
        return tail_index_to_alpha(float(result))

    if spec.output_type == "gpd_dict":
        if not isinstance(result, dict) or "xi" not in result:
            raise ValueError(
                f"{spec.display_name} did not return a dictionary with xi."
            )
        return tail_index_to_alpha(float(result["xi"]))

    if spec.output_type == "gpd_tuple":
        if not isinstance(result, (tuple, list, np.ndarray)):
            raise ValueError(
                f"{spec.display_name} did not return (xi, beta)."
            )
        if len(result) < 1:
            raise ValueError(f"{spec.display_name} returned an empty result.")
        return tail_index_to_alpha(float(result[0]))

    raise ValueError(f"Unknown output type: {spec.output_type}")
