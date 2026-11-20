"""Estimators that use observational samples (z, x, y) only."""

from __future__ import annotations

from typing import Callable

import numpy as np


def naive_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of y on x with an intercept and no confounder adjustment."""
    xc = x - x.mean()
    return float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))


def _design(x: np.ndarray, z: np.ndarray, interaction: bool) -> np.ndarray:
    cols = [np.ones_like(x), x, z]
    if interaction:
        cols.append(x * z)
    return np.column_stack(cols)


def backdoor_regression_ate(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x0: float = 0.0,
    x1: float = 1.0,
    interaction: bool = True,
) -> float:
    """Backdoor adjustment (g-formula) with a linear outcome model.

    Fits E[Y | X, Z] by OLS, then integrates over the empirical distribution
    of Z (a Monte Carlo average over the observed z values):

        ATE = (1/n) * sum_i [ m(x1, z_i) - m(x0, z_i) ]
    """
    coef, *_ = np.linalg.lstsq(_design(x, z, interaction), y, rcond=None)
    m1 = _design(np.full_like(z, x1), z, interaction) @ coef
    m0 = _design(np.full_like(z, x0), z, interaction) @ coef
    return float(np.mean(m1 - m0))


def bootstrap_ci(
    estimator: Callable[..., float],
    data: tuple[np.ndarray, ...],
    rng: np.random.Generator,
    n_boot: int = 200,
    level: float = 0.95,
) -> tuple[float, float]:
    """Nonparametric percentile bootstrap interval, resampling rows of `data`."""
    n = len(data[0])
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = estimator(*(arr[idx] for arr in data))
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(stats, [alpha, 1.0 - alpha])
    return float(lo), float(hi)
