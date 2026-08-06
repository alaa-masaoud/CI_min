from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


InputType = Literal[
    "frechet_data",
    "frechet_data_internal_threshold",
    "frechet_exceedances",
]

OutputType = Literal["tail_index", "gpd_dict", "gpd_tuple"]


@dataclass(frozen=True)
class EstimatorSpec:
    display_name: str
    module_name: str
    function_name: str
    input_type: InputType
    output_type: OutputType
    uses_k: bool


ESTIMATORS = [
    EstimatorSpec(
        "GPD MLE",
        "estimators.gpd_mle_estimator",
        "gpd_mle_estimator",
        "frechet_data_internal_threshold",
        "gpd_dict",
        False,
    ),
    EstimatorSpec(
        "Grimshaw MLE",
        "estimators.grimshaw_mle_estimator",
        "grimshaw_mle_estimator",
        "frechet_exceedances",
        "gpd_tuple",
        False,
    ),
    EstimatorSpec(
        "Hill",
        "estimators.hill_estimator",
        "hill_estimator",
        "frechet_data",
        "tail_index",
        True,
    ),
    EstimatorSpec(
        "L-Moment",
        "estimators.lmoment_estimator",
        "lmoment_estimator",
        "frechet_exceedances",
        "gpd_tuple",
        False,
    ),
    EstimatorSpec(
        "Mixed Kernel",
        "estimators.mixed_kernel_estimator",
        "mixed_moment_kernel_estimator",
        "frechet_data",
        "tail_index",
        True,
    ),
    EstimatorSpec(
        "Moment",
        "estimators.moment_estimator",
        "moment_estimator",
        "frechet_data",
        "tail_index",
        True,
    ),
    EstimatorSpec(
        "MPS GPD",
        "estimators.mps_gpd_estimator",
        "mps_gpd_estimator",
        "frechet_exceedances",
        "gpd_tuple",
        False,
    ),
    EstimatorSpec(
        "Pickands",
        "estimators.pickands_estimator",
        "pickands_estimator",
        "frechet_data",
        "tail_index",
        True,
    ),
    EstimatorSpec(
        "PWM",
        "estimators.pwm_estimator",
        "pwm_estimator",
        "frechet_exceedances",
        "gpd_tuple",
        False,
    ),
    EstimatorSpec(
        "QQ",
        "estimators.qq_estimator",
        "qq_estimator",
        "frechet_data",
        "tail_index",
        True,
    ),
    EstimatorSpec(
        "Reduced Bias Hill",
        "estimators.reduced_bias_hill_estimator",
        "reduced_bias_hill_estimator",
        "frechet_data",
        "tail_index",
        True,
    ),
    EstimatorSpec(
        "UH",
        "estimators.uh_estimator",
        "uh_exponential_regression_estimator",
        "frechet_data",
        "tail_index",
        True,
    ),
    EstimatorSpec(
        "Zhang LME",
        "estimators.zhang_lme_estimator",
        "zhang_lme_estimator",
        "frechet_exceedances",
        "gpd_tuple",
        False,
    ),
]
