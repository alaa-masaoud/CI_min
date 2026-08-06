from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    true_alpha: float
    sampler: Callable[[np.random.Generator, int], np.ndarray]


def sample_uniform_0_1(
    rng: np.random.Generator,
    size: int,
) -> np.ndarray:
    return rng.uniform(0.0, 1.0, size=size)


# The experiment requested by the project: Uniform(0,1) only.
DISTRIBUTIONS: list[DistributionSpec] = [
    DistributionSpec(
        name="Uniform(0,1)",
        true_alpha=1.0,
        sampler=sample_uniform_0_1,
    )
]
