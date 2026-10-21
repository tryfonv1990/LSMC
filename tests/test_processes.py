"""Tests for path simulation (lsmc.processes)."""

import numpy as np
import pytest

from lsmc.processes import simulate_gbm, simulate_gbm_exact, simulate_heston


def test_gbm_martingale_zero_drift():
    """With r=0, E[S_T] = S_0 within Monte Carlo error."""
    rng = np.random.default_rng(0)
    paths = simulate_gbm_exact(100, 0.0, 0.2, 1.0, 50, 200_000, rng=rng)
    se = paths[:, -1].std(ddof=1) / np.sqrt(paths.shape[0])
    assert abs(paths[:, -1].mean() - 100) < 4 * se


def test_gbm_terminal_moments():
    """Exact GBM terminal mean/variance match the lognormal closed form."""
    rng = np.random.default_rng(1)
    S0, r, sigma, T = 100, 0.05, 0.3, 1.0
    ST = simulate_gbm_exact(S0, r, sigma, T, 50, 400_000, rng=rng)[:, -1]
    mean_exact = S0 * np.exp(r * T)
    var_exact = S0**2 * np.exp(2 * r * T) * (np.exp(sigma**2 * T) - 1)
    assert ST.mean() == pytest.approx(mean_exact, rel=0.01)
    assert ST.var() == pytest.approx(var_exact, rel=0.03)


def test_gbm_shapes_and_initial_column():
    paths = simulate_gbm(50, 0.03, 0.2, 1.0, 40, 1000, rng=np.random.default_rng(2))
    assert paths.shape == (1000, 41)
    assert np.allclose(paths[:, 0], 50)


def test_gbm_non_negative():
    paths = simulate_gbm(10, 0.1, 0.9, 2.0, 20, 5000, rng=np.random.default_rng(3))
    assert (paths >= 0).all()


def test_heston_variance_mean_reverts():
    """CIR variance mean-reverts toward theta; full-truncation keeps S valid.

    Under the full-truncation scheme the stored variance may dip marginally
    below zero (it is floored to 0 wherever used), so I assert only small
    negative excursions and that the spot stays strictly positive.
    """
    rng = np.random.default_rng(4)
    theta = 0.04
    S, v = simulate_heston(1.0, 0.0, 0.09, 3.0, theta, 0.3, -0.5, 2.0,
                           100, 100_000, rng=rng)
    assert (S > 0).all()
    assert v.min() > -0.5 * theta   # truncation undershoots stay small vs theta
    # Starting above theta, the terminal mean variance should have fallen toward it.
    assert theta < v[:, -1].mean() < 0.09


def test_heston_martingale_zero_drift():
    """With r=0 the Heston spot is a martingale: E[S_T] ~= S_0."""
    rng = np.random.default_rng(5)
    S, _ = simulate_heston(1.0, 0.0, 0.04, 2.0, 0.04, 0.3, -0.5, 1.0,
                           50, 200_000, rng=rng)
    se = S[:, -1].std(ddof=1) / np.sqrt(S.shape[0])
    assert abs(S[:, -1].mean() - 1.0) < 4 * se
