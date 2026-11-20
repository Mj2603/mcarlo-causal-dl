import numpy as np
import pytest

from mcarlo_causal import ConfoundedSCM, backdoor_regression_ate, bootstrap_ci, naive_slope


def test_true_ate_closed_form():
    scm = ConfoundedSCM(c=1.7)
    assert scm.true_ate(0.0, 1.0) == pytest.approx(1.7)
    assert scm.true_ate(-1.0, 2.0) == pytest.approx(5.1)


def test_oracle_mc_matches_true_ate():
    scm = ConfoundedSCM()
    est = scm.oracle_mc_ate(400_000, np.random.default_rng(0))
    # sd of (Y1 - Y0) per draw is sqrt(e^2 + 2 sigma_y^2) ~ 0.77 -> SE ~ 0.0012
    assert est == pytest.approx(scm.true_ate(), abs=0.01)


@pytest.mark.parametrize("e", [0.0, 0.3])
def test_naive_slope_matches_closed_form_bias(e):
    scm = ConfoundedSCM(e=e)
    z, x, y = scm.sample_observational(400_000, np.random.default_rng(1))
    assert naive_slope(x, y) == pytest.approx(scm.population_naive_slope(), abs=0.01)
    # and the naive slope is clearly biased away from the true effect
    assert abs(scm.population_naive_slope() - scm.true_ate()) > 0.3


def test_naive_slope_closed_form_value_no_interaction():
    # c + d*(phi0 + 0.1) / (0.25 + 0.01 + 0.25 + 0.2*phi0)
    phi0 = 1 / np.sqrt(2 * np.pi)
    expected = 2.0 + 0.5 * (phi0 + 0.1) / (0.51 + 0.2 * phi0)
    assert ConfoundedSCM(e=0.0).population_naive_slope() == pytest.approx(expected, rel=1e-12)


def test_naive_slope_equals_polyfit():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=100), rng.normal(size=100)
    assert naive_slope(x, y) == pytest.approx(np.polyfit(x, y, 1)[0])


def test_backdoor_recovers_truth_without_noise():
    scm = ConfoundedSCM(sigma_y=0.0)
    z, x, y = scm.sample_observational(1000, np.random.default_rng(3))
    # With no outcome noise the fit is exact, so the g-formula returns the
    # in-sample effect c + e * mean(z) (the population ATE is c since E[Z] = 0).
    assert backdoor_regression_ate(z, x, y) == pytest.approx(scm.c + scm.e * z.mean(), abs=1e-10)


def test_backdoor_consistent_large_sample():
    scm = ConfoundedSCM()
    z, x, y = scm.sample_observational(400_000, np.random.default_rng(4))
    assert backdoor_regression_ate(z, x, y) == pytest.approx(scm.true_ate(), abs=0.01)


def test_no_confounding_naive_equals_truth():
    scm = ConfoundedSCM(a=0.0, b=0.0, e=0.0)
    assert scm.population_naive_slope() == pytest.approx(scm.true_ate())


def test_bootstrap_ci_contains_point_estimate():
    scm = ConfoundedSCM()
    z, x, y = scm.sample_observational(2000, np.random.default_rng(5))
    lo, hi = bootstrap_ci(backdoor_regression_ate, (z, x, y), np.random.default_rng(6), n_boot=100)
    assert lo < backdoor_regression_ate(z, x, y) < hi


def test_seeded_sampling_is_reproducible():
    scm = ConfoundedSCM()
    a = scm.sample_observational(10, np.random.default_rng(7))
    b = scm.sample_observational(10, np.random.default_rng(7))
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
