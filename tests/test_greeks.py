"""Tests for the American-put gamma profile (lsmc.greeks)."""

import numpy as np
import pytest

from lsmc.greeks import gamma_profile


def test_gamma_positive():
    """Gamma of a (convex) put payoff is non-negative at every spot."""
    pts = gamma_profile([36, 40, 44], 40, 0.0, 0.2, 1.0, h=0.5,
                        n_steps=50, n_paths=30000, n_batches=4, seed=5)
    for p in pts:
        assert p.gamma > -1e-6            # allow tiny negative MC noise


def test_gamma_peaks_near_atm():
    """Gamma is largest near the money (peak curvature) for the put.

    Uses a one-month horizon: gamma is sharply peaked at the money for short
    maturities (at long maturities it flattens and the peak drifts, which is a
    property of the option, not the estimator)."""
    pts = gamma_profile([32, 40, 48], 40, 0.0, 0.2, 1.0 / 12, h=0.5,
                        n_steps=21, n_paths=30000, n_batches=4, seed=6)
    g = {p.S0: p.gamma for p in pts}
    assert g[40] > g[32]
    assert g[40] > g[48]


def test_gamma_estimator_variance_finite_positive():
    """The reported gamma-estimator variance is finite and non-negative."""
    pts = gamma_profile([40], 40, 0.0, 0.2, 1.0, h=0.5,
                        n_steps=50, n_paths=30000, n_batches=5, seed=7)
    p = pts[0]
    assert np.isfinite(p.gamma_var) and p.gamma_var >= 0.0
    assert np.isfinite(p.gamma_stderr) and p.gamma_stderr >= 0.0
