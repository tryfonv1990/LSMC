"""Closed-form and analytic-approximation benchmarks.

* bs_put / bs_call -- Black-Scholes European values. Used to isolate the
  early-exercise premium (American - European) that LS Table 1 reports, and as
  the control-variate reference in Section 2.
* baw_american_put -- Barone-Adesi & Whaley (1987) quadratic approximation
  of the American put, the analytic American benchmark for Sections 2-3.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bs_call(S: float, K: float, r: float, sigma: float, T: float,
            q: float = 0.0) -> float:
    """Black-Scholes European call (continuous dividend yield q)."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S: float, K: float, r: float, sigma: float, T: float,
           q: float = 0.0) -> float:
    """Black-Scholes European put. Via put-call parity from bs_call."""
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def baw_american_put(S: float, K: float, r: float, sigma: float, T: float,
                     q: float = 0.0) -> float:
    """Barone-Adesi-Whaley (1987) American put approximation.

    Idea: the early-exercise premium approximately satisfies the same PDE as the
    option but with the time-decay term dropped, which reduces to an ODE with a
    power-law solution S^q2. Below a critical spot S** the option is exercised
    (value = K - S); above it the value is the European put plus a premium
    A2 (S/S**)^q2. S** solves the value-matching / smooth-pasting condition.

    Reduces to the European put as T -> 0 or sigma -> 0. With r = q = 0 the
    American put still carries a positive early-exercise premium.
    """
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K - S, max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0))

    # r = 0 (and no carry) limit: BAW's k = 1 - exp(-rT) -> 0, so the M/k term in
    # q2 blows up (0/0). Financially this is the right place to stop: with zero
    # interest and no dividend the American put has no early-exercise incentive
    # (the discounted spot is a martingale, so continuation >= intrinsic always),
    # hence American = European. Return the exact European value -- the correct
    # r -> 0 limit of the approximation. This is the regime Section 2 mandates.
    if r <= 0.0 and q <= 0.0:
        return bs_put(S, K, r, sigma, T, q)

    # Dimensionless BAW coefficients (their eqs. for M, N, k).
    sig2 = sigma * sigma
    M = 2.0 * r / sig2
    N = 2.0 * (r - q) / sig2
    k = 1.0 - np.exp(-r * T)                       # note: -> 0 as r -> 0
    # Root q2 < 0 of the characteristic quadratic (put branch).
    q2 = (-(N - 1.0) - np.sqrt((N - 1.0) ** 2 + 4.0 * M / k)) / 2.0

    def _euro(s):
        return bs_put(s, K, r, sigma, T, q)

    def smooth_pasting(Sx):
        # LHS/RHS of the value-matching condition rewritten as f(S*) = 0.
        d1 = (np.log(Sx / K) + (r - q + 0.5 * sig2) * T) / (sigma * np.sqrt(T))
        rhs = (_euro(Sx) - (K - Sx)
               - (Sx / q2) * (1.0 - np.exp(-q * T) * norm.cdf(-d1)))
        return rhs

    # Critical exercise price S** in (0, K); bracket and solve.
    try:
        S_star = brentq(smooth_pasting, 1e-8, K, maxiter=200)
    except ValueError:
        # No sign change (e.g. degenerate params) -> fall back to European value.
        return _euro(S)

    if S <= S_star:
        return K - S                               # immediate exercise region
    # Above S**: European value plus the early-exercise premium A2 (S/S**)^q2.
    d1_star = (np.log(S_star / K) + (r - q + 0.5 * sig2) * T) / (sigma * np.sqrt(T))
    A2 = -(S_star / q2) * (1.0 - np.exp(-q * T) * norm.cdf(-d1_star))
    return _euro(S) + A2 * (S / S_star) ** q2
