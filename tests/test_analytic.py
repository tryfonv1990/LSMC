"""Tests for closed-form and analytic-approximation benchmarks (lsmc.analytic)."""

import numpy as np
import pytest

from lsmc.analytic import baw_american_put, bs_call, bs_put


def test_bs_put_reference_value():
    """Black-Scholes ATM put, S=K=40, r=0.06, sigma=0.2, T=1 -> ~2.066."""
    assert bs_put(40, 40, 0.06, 0.2, 1.0) == pytest.approx(2.0664, abs=1e-3)


def test_put_call_parity():
    """C - P = S e^{-qT} - K e^{-rT}."""
    import numpy as np
    S, K, r, q, sigma, T = 42, 40, 0.05, 0.01, 0.25, 1.5
    lhs = bs_call(S, K, r, sigma, T, q) - bs_put(S, K, r, sigma, T, q)
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert lhs == pytest.approx(rhs, abs=1e-9)


def test_bs_put_intrinsic_at_expiry():
    assert bs_put(30, 40, 0.06, 0.2, 0.0) == pytest.approx(10.0)
    assert bs_put(50, 40, 0.06, 0.2, 0.0) == pytest.approx(0.0)


@pytest.mark.parametrize("S", [36, 38, 40, 42, 44])
def test_baw_bounds(S):
    """American put >= European put and >= intrinsic value."""
    amer = baw_american_put(S, 40, 0.06, 0.2, 1.0)
    euro = bs_put(S, 40, 0.06, 0.2, 1.0)
    assert amer >= euro - 1e-9
    assert amer >= max(40 - S, 0) - 1e-9


def test_baw_increases_with_vol():
    """Higher volatility -> more valuable American put."""
    low = baw_american_put(40, 40, 0.06, 0.15, 1.0)
    high = baw_american_put(40, 40, 0.06, 0.35, 1.0)
    assert high > low


@pytest.mark.parametrize("S", [36, 40, 44])
def test_baw_zero_rate_equals_european(S):
    """At r = 0 there is no early-exercise premium: BAW = European put.

    Guards the r = 0 branch (Section 2), where BAW's k = 1 - exp(-rT) -> 0 would
    otherwise divide by zero. The value must be finite and equal to bs_put.
    """
    baw = baw_american_put(S, 40, 0.0, 0.2, 1.0)
    euro = bs_put(S, 40, 0.0, 0.2, 1.0)
    assert np.isfinite(baw)
    assert baw == pytest.approx(euro, abs=1e-9)
