"""Sensitivities for the American put -- assignment Section 2 (final part).

Computes the gamma profile (gamma vs spot) for a range of moneynesses using
finite-difference bump-and-revalue with common random numbers (the same paths /
random seed at each bumped spot) to control estimator noise, and reports the
variance of the gamma estimator across independent batches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .american import price_american_put_on_paths
from .processes import draw_normals


@dataclass
class GammaPoint:
    """Gamma estimate at one spot, with its across-batch estimator variance."""
    S0: float
    gamma: float          # mean central-difference gamma over the batches
    gamma_var: float      # variance of the per-batch gamma estimates
    gamma_stderr: float   # standard error of the mean gamma


def _price_at(S0: float, K: float, r: float, sigma: float, T: float,
              dW: np.ndarray, n_steps: int, basis: str) -> float:
    """LSMC American-put PV for a given S0 on a FIXED Brownian path dW.

    Building the GBM grid from the exact log-Euler solution on shared increments
    dW is the common-random-numbers trick: every bumped spot (S0-h, S0, S0+h)
    sees the identical noise, so the finite difference cancels the Monte Carlo
    noise and leaves the (deterministic) curvature of price in spot.
    """
    dt = T / n_steps
    # Exact GBM on the shared increments: S_t = S0 exp((r - v/2)t + sigma W_t).
    increments = (r - 0.5 * sigma ** 2) * dt + sigma * dW
    log_paths = np.log(S0) + np.cumsum(increments, axis=1)
    n_paths = dW.shape[0]
    paths = np.concatenate([np.full((n_paths, 1), S0), np.exp(log_paths)], axis=1)
    return price_american_put_on_paths(paths, K, r, T, basis=basis).price


def gamma_profile(spots, K: float, r: float, sigma: float, T: float, *,
                  h: float = 0.5, n_steps: int = 252, n_paths: int = 100_000,
                  antithetic: bool = True, n_batches: int = 10,
                  seed: int = 0, basis: str = "laguerre") -> list[GammaPoint]:
    """Gamma vs spot by central finite difference with common random numbers.

    gamma(S0) = [V(S0 + h) - 2 V(S0) + V(S0 - h)] / h^2, where each V is priced
    on the SAME Brownian path within a batch (CRN). I repeat over n_batches
    independent path sets and report the mean gamma and the across-batch variance
    of the estimator -- the variance the assignment asks me to quantify.

    h is the absolute spot bump; spots is any iterable of S0 values (e.g.
    a moneyness grid around the strike).
    """
    out = []
    for S0 in spots:
        per_batch = []
        for b in range(n_batches):
            # Independent Brownian path per batch, but shared across the 3 bumps.
            rng = np.random.default_rng([seed, int(round(S0 * 1000)), b])
            dW = draw_normals(n_paths, n_steps, rng, antithetic) * np.sqrt(T / n_steps)
            v_up = _price_at(S0 + h, K, r, sigma, T, dW, n_steps, basis)
            v_0 = _price_at(S0, K, r, sigma, T, dW, n_steps, basis)
            v_dn = _price_at(S0 - h, K, r, sigma, T, dW, n_steps, basis)
            per_batch.append((v_up - 2.0 * v_0 + v_dn) / (h * h))
        g = np.asarray(per_batch)
        out.append(GammaPoint(S0=S0, gamma=g.mean(),
                              gamma_var=g.var(ddof=1),
                              gamma_stderr=g.std(ddof=1) / np.sqrt(n_batches)))
    return out
