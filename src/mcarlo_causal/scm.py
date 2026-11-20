"""A small structural causal model (SCM) with one observed confounder.

    Z ~ N(0, 1)
    X = a * 1{Z > 0} + b * Z + U,          U ~ N(0, sigma_u^2)
    Y = c * X + d * Z + e * X * Z + eps,   eps ~ N(0, sigma_y^2)

Z opens the backdoor path X <- Z -> Y. Because E[Z] = 0, the true average
treatment effect of moving X from x0 to x1 is c * (x1 - x0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_PHI0 = 1.0 / np.sqrt(2.0 * np.pi)  # standard normal density at 0 = E[Z 1{Z>0}]


@dataclass(frozen=True)
class ConfoundedSCM:
    a: float = 1.0
    b: float = 0.1
    sigma_u: float = 0.5
    c: float = 2.0
    d: float = 0.5
    e: float = 0.3
    sigma_y: float = 0.5

    # --- sampling -------------------------------------------------------
    def sample_z(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, 1.0, size=n)

    def treatment(self, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        u = rng.normal(0.0, self.sigma_u, size=z.shape)
        return self.a * (z > 0).astype(float) + self.b * z + u

    def outcome(self, x: np.ndarray, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        eps = rng.normal(0.0, self.sigma_y, size=z.shape)
        return self.c * x + self.d * z + self.e * x * z + eps

    def sample_observational(self, n: int, rng: np.random.Generator):
        """Draw (z, x, y) from the observational distribution."""
        z = self.sample_z(n, rng)
        x = self.treatment(z, rng)
        y = self.outcome(x, z, rng)
        return z, x, y

    # --- ground truth ---------------------------------------------------
    def true_ate(self, x0: float = 0.0, x1: float = 1.0) -> float:
        """Closed-form E[Y | do(X=x1)] - E[Y | do(X=x0)]. Uses E[Z] = 0."""
        return self.c * (x1 - x0)

    def oracle_mc_ate(
        self, n: int, rng: np.random.Generator, x0: float = 0.0, x1: float = 1.0
    ) -> float:
        """Monte Carlo estimate of the interventional ATE by simulating do(X).

        This uses the true structural equations, so it is a check on the
        simulator, not an estimator that could be applied to real data.
        """
        z = self.sample_z(n, rng)
        y1 = self.outcome(np.full_like(z, x1), z, rng)
        y0 = self.outcome(np.full_like(z, x0), z, rng)
        return float(np.mean(y1 - y0))

    def population_naive_slope(self) -> float:
        """Closed-form population OLS slope of Y on X (no adjustment).

        slope = c + (d * Cov(X, Z) + e * Cov(X, XZ)) / Var(X), with
          Var(X)     = a^2/4 + b^2 + sigma_u^2 + 2ab*phi(0)
          Cov(X, Z)  = a*phi(0) + b
          Cov(X, XZ) = (a^2*phi(0) + a*b) / 2
        """
        a, b = self.a, self.b
        var_x = a**2 / 4 + b**2 + self.sigma_u**2 + 2 * a * b * _PHI0
        cov_xz = a * _PHI0 + b
        cov_x_xz = (a**2 * _PHI0 + a * b) / 2
        return self.c + (self.d * cov_xz + self.e * cov_x_xz) / var_x
