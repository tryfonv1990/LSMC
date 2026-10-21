"""Tests for natural-gas storage valuation (lsmc.storage)."""

import numpy as np
import pytest

from lsmc.storage import simulate_day_ahead, value_storage

# A small standard-contract bundle reused across tests (Exhibit 1 shape, scaled
# grid kept coarse for speed).
CONTRACT = dict(v_min=0, v_max=250000, v_start=100000, v_end=100000,
                max_inject=2500, max_withdraw=7500, alpha=2500, penalty=1.0,
                basis="bspline_b6")


def _paths(sigma=0.0945, n_paths=2000, seed=0, n_days=365):
    S0 = np.exp(3.0)
    rng = np.random.default_rng(seed)
    return simulate_day_ahead(S0, 0.05, sigma, 3.0, n_days, n_paths, rng=rng)


def test_price_mean_reverts():
    """Log-price reverts toward mu: terminal mean price near exp(mu) = e^3."""
    paths = _paths()
    # Started at the long-run level, so the terminal mean should stay close to it.
    assert paths[:, -1].mean() == pytest.approx(np.exp(3.0), rel=0.15)
    assert (paths > 0).all()                      # prices are strictly positive


def test_storage_value_positive():
    """A storage contract is a bundle of options, so its value is non-negative."""
    res = value_storage(_paths(), **CONTRACT)
    assert res.value > 0.0
    assert res.stderr > 0.0


def test_storage_value_increases_with_volatility():
    """Higher spot volatility -> more extrinsic value -> higher storage value."""
    low = value_storage(_paths(sigma=0.03, seed=1), **CONTRACT).value
    high = value_storage(_paths(sigma=0.15, seed=1), **CONTRACT).value
    assert high > low


def test_terminal_penalty_enforced_by_value():
    """A punitive end-volume penalty should not raise value above the free case.

    With a large penalty the strategy is forced back to v_end, costing optionality,
    so the value cannot exceed the lightly-penalised case (sanity on the penalty).
    """
    paths = _paths(seed=2)
    light = value_storage(paths, **{**CONTRACT, "penalty": 1.0}).value
    heavy = value_storage(paths, **{**CONTRACT, "penalty": 50.0}).value
    assert heavy <= light + 1e-6 * abs(light) + 1.0


def test_off_grid_alpha_rejected():
    """alpha must divide capacity, volumes and rates; otherwise fail loudly."""
    with pytest.raises(ValueError, match="multiple of alpha"):
        value_storage(_paths(n_paths=200, n_days=10), **{**CONTRACT, "alpha": 3000})
