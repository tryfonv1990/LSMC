"""Tests for Section 3: Milstein vs Euler discretisation convergence.

These check the bias behaviour the section is about: on a shared Brownian path
(so Monte Carlo noise cancels), the Milstein scheme should track the bias-free
reference at least as well as Euler, and strictly better on a coarse grid.
"""

import numpy as np
import pytest

from lsmc.american import price_american_put_on_paths
from lsmc.processes import (correlated_heston_increments, draw_normals,
                            simulate_gbm, simulate_gbm_exact, simulate_heston)


def _gbm_price_on_shared_dW(scheme, S0, K, r, sigma, T, n_steps, dW):
    """American-put price with a given scheme on caller-supplied increments dW."""
    if scheme == "exact":
        paths = simulate_gbm_exact(S0, r, sigma, T, n_steps, dW.shape[0], dW=dW)
    else:
        paths = simulate_gbm(S0, r, sigma, T, n_steps, dW.shape[0], scheme=scheme,
                             dW=dW)
    return price_american_put_on_paths(paths, K, r, T, basis="laguerre").price


def test_milstein_less_biased_than_euler_gbm():
    """On a coarse grid and shared path, Milstein sits closer to the exact GBM.

    Using the SAME Brownian increments for all three schemes removes Monte Carlo
    noise, so the remaining gap to the exact scheme is pure discretisation bias.
    """
    S0, K, r, sigma, T = 40, 40, 0.06, 0.4, 1.0
    n_steps, n_paths = 12, 120000        # deliberately coarse -> visible bias
    rng = np.random.default_rng(0)
    dW = draw_normals(n_paths, n_steps, rng, antithetic=True) * np.sqrt(T / n_steps)

    exact = _gbm_price_on_shared_dW("exact", S0, K, r, sigma, T, n_steps, dW)
    euler = _gbm_price_on_shared_dW("euler", S0, K, r, sigma, T, n_steps, dW)
    milstein = _gbm_price_on_shared_dW("milstein", S0, K, r, sigma, T, n_steps, dW)

    # Milstein's bias magnitude should not exceed Euler's (usually strictly less).
    assert abs(milstein - exact) <= abs(euler - exact) + 1e-9


def test_milstein_heston_runs_and_is_finite():
    """Milstein Heston simulates valid paths and prices a finite value."""
    rng = np.random.default_rng(1)
    S, v = simulate_heston(1.0, 0.06, 0.0064, 2.0, 0.0144, 0.10, 0.3, 1.0,
                           50, 40000, scheme="milstein", rng=rng)
    assert (S > 0).all()
    price = price_american_put_on_paths(S, 1.0, 0.06, 1.0, basis="laguerre").price
    assert np.isfinite(price) and price > 0.0


def test_heston_shared_increments_nested_grid():
    """Pre-drawn increments reproduce the rng path, and coarse sums nest the fine grid.

    Section 3c drives the coarse grid with block sums of the fine increments, so
    the fine and coarse runs share one Brownian motion. Check that (i) passing the
    increments explicitly gives the same paths as drawing them internally, and
    (ii) the summed increments keep the variance and the rho correlation.
    """
    args = (1.0, 0.06, 0.04, 2.0, 0.04, 0.6, 0.3, 1.0)
    n_paths, fine, coarse = 20000, 40, 10
    S_rng, v_rng = simulate_heston(*args, fine, n_paths,
                                   rng=np.random.default_rng(7))
    dW = correlated_heston_increments(n_paths, fine, 1.0 / fine, 0.3,
                                      np.random.default_rng(7), antithetic=True)
    S_dw, v_dw = simulate_heston(*args, fine, n_paths, dW=dW)
    np.testing.assert_array_equal(S_rng, S_dw)
    np.testing.assert_array_equal(v_rng, v_dw)

    dWv_c, dWs_c = (x.reshape(n_paths, coarse, fine // coarse).sum(axis=2) for x in dW)
    assert dWv_c.var() == pytest.approx(1.0 / coarse, rel=0.03)
    assert np.corrcoef(dWv_c.ravel(), dWs_c.ravel())[0, 1] == pytest.approx(0.3, abs=0.02)
    # Total Brownian increment over [0, T] is identical on both grids.
    np.testing.assert_allclose(dWs_c.sum(axis=1), dW[1].sum(axis=1))
