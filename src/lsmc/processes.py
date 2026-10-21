"""Stochastic-process path simulation.

Discrete-time simulation of the price processes used throughout the assignment
on a fixed time grid, with a selectable time-stepping scheme:

* simulate_gbm    -- geometric Brownian motion, dS = r S dt + sigma S dW.
* simulate_heston -- Heston stochastic-volatility model,
                         dS = mu S dt + sqrt(v) S dW^S,
                         dv = kappa(theta - v) dt + xi sqrt(v) dW^v,
                         corr(dW^S, dW^v) = rho.

Both support scheme="euler" (Euler-Maruyama, Section 1) and
scheme="milstein" (adds the second-order correction term, Section 3), plus
antithetic variates for variance reduction.

Convention: every simulator returns the full price grid of shape
(n_paths, n_steps + 1) with column 0 equal to the initial value, so the
LSMC backward induction can access the state at every exercise date.
"""

from __future__ import annotations

import numpy as np


def draw_normals(n_paths: int, n_steps: int, rng: np.random.Generator,
                 antithetic: bool) -> np.ndarray:
    """Standard-normal increments of shape (n_paths, n_steps).

    Public helper: the simulators use it internally, and the discretisation-step
    study calls it directly to build a shared fine Brownian path for nested grids.

    With antithetic variates I draw n_paths/2 normals and append their
    negatives; pairing Z with -Z halves the variance of the mean estimator for
    payoffs that are close to linear in the driving noise.
    """
    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("antithetic sampling requires an even n_paths")
        half = rng.standard_normal((n_paths // 2, n_steps))
        return np.concatenate([half, -half], axis=0)
    return rng.standard_normal((n_paths, n_steps))


# Backwards-compatible internal alias (older call sites used the underscore name).
_draw_normals = draw_normals


def simulate_gbm(S0: float, r: float, sigma: float, T: float,
                 n_steps: int, n_paths: int, *, scheme: str = "euler",
                 antithetic: bool = True, rng: np.random.Generator | None = None,
                 dW: np.ndarray | None = None) -> np.ndarray:
    """Simulate geometric Brownian motion on a uniform grid of n_steps.

    Parameters mirror the risk-neutral SDE dS = r S dt + sigma S dW.

    scheme:
      * "euler"    -- Euler-Maruyama on S: S += r S dt + sigma S dW. Simple, but
                      order-0.5 strong / order-1 weak, and can go negative for
                      large steps.
      * "milstein" -- adds 0.5 sigma^2 S (dW^2 - dt), the Ito correction from the
                      state-dependent diffusion; raises strong order to 1.0.

    dW (optional): pre-drawn Brownian increments of shape (n_paths, n_steps),
    already scaled by sqrt(dt). When supplied, the simulator uses them instead of
    drawing its own -- this lets a caller feed the same Brownian path into two
    schemes (Euler vs exact) so their difference isolates the discretisation bias
    with no Monte Carlo noise (see the Section-1b nested-grid study). rng and
    antithetic are ignored in that case.

    Returns the (n_paths, n_steps + 1) price grid.
    """
    rng = np.random.default_rng() if rng is None else rng
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    # Use caller-supplied increments if given; otherwise draw fresh ones.
    if dW is None:
        dW = draw_normals(n_paths, n_steps, rng, antithetic) * sqrt_dt

    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    for t in range(n_steps):
        prev = S[:, t]
        drift = r * prev * dt
        diffusion = sigma * prev * dW[:, t]
        nxt = prev + drift + diffusion
        if scheme == "milstein":
            nxt = nxt + 0.5 * sigma**2 * prev * (dW[:, t] ** 2 - dt)
        elif scheme != "euler":
            raise ValueError(f"unknown scheme {scheme!r}")
        # Floor at zero: GBM is non-negative, but discrete steps can undershoot.
        S[:, t + 1] = np.maximum(nxt, 0.0)
    return S


def simulate_gbm_exact(S0: float, r: float, sigma: float, T: float,
                       n_steps: int, n_paths: int, *, antithetic: bool = True,
                       rng: np.random.Generator | None = None,
                       dW: np.ndarray | None = None) -> np.ndarray:
    """Exact log-Euler GBM: S_{t+1} = S_t exp((r - sigma^2/2) dt + sigma dW).

    Free of time-discretisation bias (the log-SDE has constant coefficients), so
    it is a useful reference for isolating the discretisation error of the
    plain Euler/Milstein schemes above from the Monte Carlo (variance) error.

    dW (optional): pre-drawn Brownian increments (n_paths, n_steps), scaled by
    sqrt(dt) -- see simulate_gbm for why this matters (shared-path bias study).
    """
    rng = np.random.default_rng() if rng is None else rng
    dt = T / n_steps
    if dW is None:
        dW = draw_normals(n_paths, n_steps, rng, antithetic) * np.sqrt(dt)
    # Exact solution of d(ln S): each step adds (r - sigma^2/2) dt + sigma dW.
    increments = (r - 0.5 * sigma**2) * dt + sigma * dW
    log_paths = np.log(S0) + np.cumsum(increments, axis=1)   # running sum along time
    # Prepend the S0 column, then exponentiate back to price space.
    return np.concatenate([np.full((n_paths, 1), S0), np.exp(log_paths)], axis=1)


def correlated_heston_increments(n_paths: int, n_steps: int, dt: float,
                                 rho: float, rng: np.random.Generator,
                                 antithetic: bool
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Correlated Brownian increments (dWv, dWs) for Heston, scaled by sqrt(dt).

    Cholesky: W^v independent, W^S = rho W^v + sqrt(1-rho^2) W^perp, so
    corr(dWs, dWv) = rho. Summing blocks of these increments gives valid coarser
    increments with the same correlation (the map is linear).
    """
    sqrt_dt = np.sqrt(dt)
    Zv = draw_normals(n_paths, n_steps, rng, antithetic)      # drives variance
    Zperp = draw_normals(n_paths, n_steps, rng, antithetic)   # independent of Zv
    dWv = Zv * sqrt_dt
    dWs = (rho * Zv + np.sqrt(1.0 - rho**2) * Zperp) * sqrt_dt
    return dWv, dWs


def simulate_heston(S0: float, r: float, v0: float, kappa: float, theta: float,
                    xi: float, rho: float, T: float, n_steps: int, n_paths: int,
                    *, scheme: str = "euler", antithetic: bool = True,
                    rng: np.random.Generator | None = None,
                    dW: tuple[np.ndarray, np.ndarray] | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the Heston model; returns (S_grid, v_grid), each (n_paths, n_steps+1).

    Variance process is CIR: dv = kappa(theta - v) dt + xi sqrt(v) dW^v. I use
    "full truncation" (max(v, 0) inside the drift/diffusion), the standard fix
    for the Euler scheme occasionally driving v below zero. The Feller condition
    2 kappa theta > xi^2 keeps the continuous process strictly positive.

    scheme="milstein" adds the CIR Milstein term 0.25 xi^2 (dW^2 - dt) to the
    variance leg, which markedly reduces its discretisation bias (Section 3).

    Antithetic variates are applied to both Brownian factors.

    dW (optional): pre-drawn (dWv, dWs) increments, each (n_paths,
    n_steps), already scaled by sqrt(dt) and correlated (corr = rho). Lets a
    caller drive a coarse and a fine grid with the same Brownian motion (sum
    blocks of fine increments) so their gap is pure discretisation bias -- see
    correlated_heston_increments and Section 3c. rng, antithetic and
    rho are not used for the noise in that case.
    """
    rng = np.random.default_rng() if rng is None else rng
    dt = T / n_steps

    if dW is None:
        dWv, dWs = correlated_heston_increments(n_paths, n_steps, dt, rho, rng,
                                                antithetic)
    else:
        dWv, dWs = dW

    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0
    for t in range(n_steps):
        vt = np.maximum(v[:, t], 0.0)          # full truncation
        sqrt_vt = np.sqrt(vt)

        v_next = v[:, t] + kappa * (theta - vt) * dt + xi * sqrt_vt * dWv[:, t]
        if scheme == "milstein":
            v_next = v_next + 0.25 * xi**2 * (dWv[:, t] ** 2 - dt)
        elif scheme != "euler":
            raise ValueError(f"unknown scheme {scheme!r}")
        v[:, t + 1] = v_next

        # Log-Euler on S using the current variance keeps S strictly positive and
        # removes the (large) discretisation error of a plain level scheme on S,
        # leaving the variance leg as the accuracy bottleneck.
        S[:, t + 1] = S[:, t] * np.exp((r - 0.5 * vt) * dt + sqrt_vt * dWs[:, t])
    return S, v
