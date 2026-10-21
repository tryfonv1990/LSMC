"""Tests for the two-factor (S, v) regression basis on the Heston American put."""

import numpy as np
import pytest

from lsmc.basis import BASES_2D, poly_2d
from lsmc.american import price_american_put_heston
from lsmc.processes import simulate_heston

# A base Heston case (Feller 2*kappa*theta > xi^2 holds).
BASE = dict(S0=1.0, K=1.0, r=0.0, v0=0.0064, kappa=2.0, theta=0.0064,
            xi=0.05, rho=-0.3, T=1.0)


def test_poly2d_columns():
    """Degree-2 two-factor basis has 6 columns {1, s, s^2, v, v^2, s*v}."""
    X = np.array([[2.0, 1.0], [0.5, 0.25]])
    design = poly_2d(X, degree=2)
    assert design.shape == (2, 6)
    # For row [s=2, v=1] the six monomials (in total-degree order) are all present.
    assert set(np.round(design[0], 6)) == {1.0, 2.0, 4.0}  # {1, s=2, s^2=4, v=1, v^2=1, sv=2}


def test_poly2d_rejects_wrong_shape():
    with pytest.raises(ValueError):
        poly_2d(np.array([1.0, 2.0, 3.0]))          # 1-D input not allowed


def test_registry_has_2d_basis():
    assert "poly2d_sv" in BASES_2D


def test_two_factor_price_is_finite_and_above_european():
    """Two-factor Heston put runs, is finite, and >= its European value."""
    rng = np.random.default_rng(7)
    res = price_american_put_heston(**BASE, n_steps=80, n_paths=40_000,
                                    two_factor=True, rng=rng)
    assert np.isfinite(res.price)
    # European (terminal-payoff) value on the same paths, r = 0.
    rng = np.random.default_rng(7)
    S, _ = simulate_heston(BASE["S0"], BASE["r"], BASE["v0"], BASE["kappa"],
                           BASE["theta"], BASE["xi"], BASE["rho"], BASE["T"],
                           80, 40_000, rng=rng)
    european = np.maximum(BASE["K"] - S[:, -1], 0.0).mean()
    assert res.price >= european - 3 * res.stderr


def test_two_factor_matches_spot_only_atm():
    """At-the-money the two bases agree within a few standard errors.

    Variance carries little extra information about the exercise decision when
    spot sits at the strike, so the two regressions should give close prices.
    """
    rng = np.random.default_rng(2024)
    spot = price_american_put_heston(**BASE, n_steps=100, n_paths=60_000,
                                     two_factor=False, rng=rng)
    rng = np.random.default_rng(2024)                # same seed => same paths
    twof = price_american_put_heston(**BASE, n_steps=100, n_paths=60_000,
                                     two_factor=True, rng=rng)
    assert abs(twof.price - spot.price) < 5 * (spot.stderr + twof.stderr)
