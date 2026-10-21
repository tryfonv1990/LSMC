"""Natural-gas storage valuation - assignment Section 4 (Boogert & de Jong 2008).

Two pieces:

1. simulate_day_ahead - Monte Carlo simulation of the mean-reverting
   (Ornstein-Uhlenbeck in log-price) day-ahead gas price of the paper's eq. (22):
       dS/S = kappa[mu(t) - ln S] dt + sigma dW,
   daily steps, M = 10,000 paths, 7 July -> 30 June, sigma = 9.45%, kappa = 0.05,
   mu(t) a chosen constant, zero rates and bid-ask spreads.

2. value_storage - the storage dynamic program. Unlike the single-exercise
   American option, at each day and each discretised inventory level the holder
   chooses to inject / hold / withdraw, adding the instantaneous cash flow
   h(S, dv) to the continuation value (paper eqs. 10-12, simplified via eq. 16).
   Continuation values are regressed on the b6 B-spline basis (Exhibit 6) via OLS,
   subject to volume and injection/withdrawal-rate constraints (eq. 8) and the
   terminal inventory requirement.

Contract defaults from Exhibit 1: capacity 250,000 MWh, start = end = 100,000 MWh.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import basis as _basis
from .processes import draw_normals


@dataclass
class StorageResult:
    """Storage valuation with its Monte Carlo standard error."""
    value: float          # mean present value of the contract (EUR)
    stderr: float         # standard error of that mean
    n_paths: int
    path_values: np.ndarray | None = None   # realised value on each path (Exhibit 3 histogram)


def simulate_day_ahead(S0: float, kappa: float, sigma: float, mu: float,
                       n_days: int, n_paths: int, *, antithetic: bool = True,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """Simulate the day-ahead gas price (paper eq. 22) on a daily grid.

    I work with the log-price X = ln S, which turns the price SDE (eq. 22) into
    a linear Ornstein-Uhlenbeck process (eq. 23):

        dX = kappa[mu - X - sigma^2/(2 kappa)] dt + sigma dW,

    and step it with Euler-Maruyama at dt = 1 day (so kappa, sigma are the daily
    rates the brief gives). mu is the constant long-run log-price level the
    assignment lets me choose. Returns the price grid S of shape
    (n_paths, n_days + 1), column 0 = S0.
    """
    rng = np.random.default_rng() if rng is None else rng
    dt = 1.0                                  # one day; kappa & sigma are daily
    # Ito correction term in eq. (23): the log-price mean-reverts to mu - sigma^2/(2 kappa).
    drift_level = mu - sigma ** 2 / (2.0 * kappa)
    dW = draw_normals(n_paths, n_days, rng, antithetic) * np.sqrt(dt)

    X = np.empty((n_paths, n_days + 1))
    X[:, 0] = np.log(S0)
    for t in range(n_days):
        # OU Euler step: pull X toward drift_level at speed kappa, plus noise.
        X[:, t + 1] = X[:, t] + kappa * (drift_level - X[:, t]) * dt + sigma * dW[:, t]
    return np.exp(X)                          # back to price space


def value_storage(paths: np.ndarray, *, v_min: float, v_max: float,
                  v_start: float, v_end: float, max_inject: float,
                  max_withdraw: float, alpha: float, penalty: float = 5.0,
                  basis: str = "bspline_b6", discount: float = 1.0) -> StorageResult:
    """Value a gas-storage contract by least-squares Monte Carlo (eqs. 10-21).

    The holder faces a dynamic program: each day, at each inventory level, choose
    inject / hold / withdraw to maximise the immediate cash flow plus the
    continuation value of the resulting inventory. I solve it by backward
    induction over a (time x inventory) grid, estimating continuation values by
    regressing realised future cash flows onto B-splines of the current spot.

    Design choices (all faithful to the paper):

    * Inventory grid v(n) = v_min + n*alpha, n = 0..N-1, with the step
      alpha chosen by the user and N = (v_max - v_min)/alpha + 1. The paper's
      discretisation (alpha = 2500, N = 101; "Standard Storage Contract", p. 89).
      Capacity, rates and start/end volumes must be whole multiples of alpha,
      so every allowed action moves an INTEGER number of grid steps and lands
      exactly on a node; no interpolation (eqs. 26-28) is needed.
    * Cash flow h(S, dv) = -S*dv (eq. 2 with zero costs/spreads): injecting
      (dv > 0) costs S per MWh; withdrawing (dv < 0) earns S per MWh.
    * Dimension reduction (eq. 16): continuation depends only on the
      POST-decision inventory, so I regress once per post-decision node. The
      design matrix depends only on S_t (not inventory), so all N regressions at
      a time step share it - I solve them in ONE least-squares call with N
      right-hand sides.
    * Decision vs realised value (LS anti-bias rule): the fitted continuation
      chooses the action, but I accumulate the realised future cash flow.
    * Terminal penalty (eq. 20) q = -penalty*S_T*|v - v_end| forces the
      strategy to finish at the required end inventory.
    """
    n_paths, n_cols = paths.shape
    T = n_cols - 1                                   # last day index

    def _steps(volume: float, name: str) -> int:
        """Volume as a whole number of grid steps; reject off-grid values."""
        k = volume / alpha
        if not np.isclose(k, round(k)):
            raise ValueError(f"{name} = {volume} is not a multiple of alpha = {alpha}"
                             " (off-grid volumes would need interpolation, eqs. 26-28)")
        return int(round(k))

    n_vol = _steps(v_max - v_min, "v_max - v_min") + 1   # grid points (101 for alpha=2500)
    _steps(v_end - v_min, "v_end - v_min")           # required end volume must be a node too
    volumes = v_min + np.arange(n_vol) * alpha       # node inventories (MWh)
    build = _basis.BASES[basis]

    # Allowed inventory steps per day: inject up (+), withdraw down (-).
    up = _steps(max_inject, "max_inject")            # grid steps injectable/day
    down = _steps(max_withdraw, "max_withdraw")      # grid steps withdrawable/day
    offsets = np.arange(-down, up + 1)               # e.g. [-3,-2,-1,0,+1] for alpha=2500

    # Terminal value (eq. 20): penalty for ending away from v_end. Scaled by the
    # terminal price so the penalty is in comparable (EUR) units to cash flows.
    S_T = paths[:, T]
    Y = -penalty * S_T[:, None] * np.abs(volumes[None, :] - v_end)   # (n_paths, n_vol)

    # Work arrays reused every day (allocated once, overwritten in place).
    best_val = np.empty((n_paths, n_vol))            # best fitted objective per (path, node)
    best_real = np.empty((n_paths, n_vol))           # realised value of that best action
    val_buf = np.empty((n_paths, n_vol))             # scratch: one action's fitted objective
    real_buf = np.empty((n_paths, n_vol))            # scratch: one action's realised value
    mask = np.empty((n_paths, n_vol), dtype=bool)    # scratch: where that action wins

    # Backward induction over trading days t = T-1, ..., 1 (t=0 takes no action).
    for t in range(T - 1, 0, -1):
        S_t = paths[:, t]
        design = build(S_t)                          # (n_paths, k) B-spline design
        Y = discount * Y                             # future cash flows, discounted one day

        # --- Eqs. 15/17: OLS continuation for ALL inventory nodes at once ------
        # One OLS solve for ALL post-decision nodes: each column of Y is the
        # regression target for one node, and they all share the same design
        # matrix (it depends only on S_t).
        coef, *_ = np.linalg.lstsq(design, Y, rcond=None)   # (k, n_vol)
        cont = design @ coef                         # fitted continuation (n_paths, n_vol)

        # --- Eqs. 18-19: best action at every (path, node), all nodes at once --
        # Loop only over the few allowed actions d (5 for alpha = 2500, 9 for
        # 1250) and update every path and node with array slices, keeping a
        # running best. In-place ops (out=, copyto) avoid allocating new arrays.
        best_val.fill(-np.inf)
        for d in offsets:
            # Action d moves node n to m = n + d. Only nodes whose target stays on
            # the grid (0 <= m < n_vol) may take it, e.g. a full store cannot inject.
            n_lo, n_hi = max(0, -d), min(n_vol, n_vol - d)   # valid source nodes [n_lo, n_hi)
            k = n_hi - n_lo                                  # how many nodes allow action d
            src = slice(n_lo, n_hi)                          # nodes n taking action d
            dst = slice(n_lo + d, n_hi + d)                  # their targets m = n + d
            cash = (-S_t * (alpha * d))[:, None]             # h(S, dv): cost if inject, profit if withdraw
            # Decision uses the FITTED continuation (eq. 18) ...
            val = np.add(cash, cont[:, dst], out=val_buf[:, :k])
            # ... but the value I carry back uses the REALISED future cash flow
            # (eq. 19, LS anti-bias rule: never carry back the fitted value).
            real = np.add(cash, Y[:, dst], out=real_buf[:, :k])
            # Strict ">" keeps the first action on ties (offsets run withdraw -> inject).
            take = np.greater(val, best_val[:, src], out=mask[:, :k])
            np.copyto(best_val[:, src], val, where=take)
            np.copyto(best_real[:, src], real, where=take)
        Y = best_real.copy()                         # realised value from each node at day t

    # Value at inception: the holder starts at v_start (no action at t=0), so the
    # contract value is the average realised cash flow from the start node.
    start_node = _steps(v_start - v_min, "v_start - v_min")
    pv = discount * Y[:, start_node]
    return StorageResult(value=pv.mean(),
                         stderr=pv.std(ddof=1) / np.sqrt(n_paths), n_paths=n_paths,
                         path_values=pv)
