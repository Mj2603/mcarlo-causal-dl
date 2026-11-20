"""Backdoor adjustment on a synthetic structural causal model with a known effect."""

from .scm import ConfoundedSCM
from .estimators import (
    naive_slope,
    backdoor_regression_ate,
    bootstrap_ci,
)

__all__ = ["ConfoundedSCM", "naive_slope", "backdoor_regression_ate", "bootstrap_ci"]
