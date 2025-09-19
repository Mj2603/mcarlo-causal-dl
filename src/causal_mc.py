from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class SCM:
    confounder_sampler: Callable[[int], np.ndarray]
    treatment_function: Callable[[np.ndarray], np.ndarray]
    outcome_function: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def sample_observational(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = self.confounder_sampler(n)
        x = self.treatment_function(z)
        y = self.outcome_function(x, z)
        return z, x, y

    def monte_carlo_ate(self, n: int = 100_000, x0: float = 0.0, x1: float = 1.0) -> float:
        z = self.confounder_sampler(n)
        y0 = self.outcome_function(np.full_like(z, x0), z)
        y1 = self.outcome_function(np.full_like(z, x1), z)
        return float(np.mean(y1 - y0))


def make_toy_scm(seed: int = 7) -> SCM:
    rng = np.random.default_rng(seed)

    def confounder_sampler(n: int) -> np.ndarray:
        return rng.normal(0.0, 1.0, size=n)

    def treatment_function(z: np.ndarray) -> np.ndarray:
        noise = rng.normal(0.0, 0.5, size=z.shape)
        return (z > 0).astype(float) + 0.1 * z + noise

    def outcome_function(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        eps = rng.normal(0.0, 0.5, size=z.shape)
        return 2.0 * x + 0.5 * z + 0.3 * x * z + eps

    return SCM(confounder_sampler, treatment_function, outcome_function)

