"""LSMC American (Bermudan) put pricer -- assignment Section 1.

Longstaff-Schwartz backward induction over a discrete exercise grid:

1. simulate price paths (processes);
2. initialise the cash flow with the terminal intrinsic value max(K - S_T, 0);
3. step backward through each exercise date: for in-the-money paths, regress the
   discounted realised future cash flow onto basis functions of spot to estimate
   the continuation value, then exercise where immediate payoff >= continuation
   (updating the realised cash flow / stopping time on those paths);
4. discount each path's first-exercise cash flow to t=0 and average.

Key LS design points, reflected below:
* regress on in-the-money paths only (section 2.2);
* compare immediate exercise against the fitted continuation, but discount the
  realised cash flow (never the fitted value) to avoid the upward bias from
  the convex max operator (section 2.3, footnote 9);
* normalise spot by the strike before building the (Laguerre) basis to avoid
  exponential underflow (section 8.3).

Benchmarks: the stylised 8-path example (~0.1144) and Table 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import basis as _basis
from .processes import simulate_gbm, simulate_gbm_exact, simulate_heston
from .regression import continuation_values


@dataclass
class LSMCResult:
    """Container for an LSMC valuation and its Monte Carlo standard error."""
    price: float
    stderr: float
    n_paths: int


def _make_basis(name: str, K: float) -> Callable[[np.ndarray], np.ndarray]:
    """Return a basis builder that normalises spot by the strike first.

    Normalising keeps the weighted-Laguerre exponential well conditioned (LS
    section 8.3) and makes the fit scale-invariant across the S in [36, 44]
    range used in Table 1.
    """
    fn = _basis.BASES[name]
    return lambda S: fn(np.asarray(S, dtype=float) / K)


def price_american_put_on_paths(paths: np.ndarray, K: float, r: float, T: float,
                                *, basis: str = "laguerre",
                                return_pv: bool = False):
    """Price an American put given a pre-simulated price grid.

    Separated from path generation so the same paths can price both the American
    option and its European control variate (Section 2), and so tests can feed
    the paper's exact 8-path matrix.

    paths has shape (n_paths, n_steps + 1); every column after column 0 is an
    exercise date (Bermudan grid). Discounting uses the per-step factor
    exp(-r dt).

    return_pv (Section 2): when True, also return the per-path discounted
    cash flow array pv (shape (n_paths,)) whose mean is the price. The
    control variate needs these per-path values -- paired with the European
    payoff on the same paths -- to estimate the optimal beta. Default False
    keeps the plain LSMCResult return for every existing caller/test.
    """
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    dt = T / n_steps
    discount = np.exp(-r * dt)
    build = _make_basis(basis, K)

    # cashflow[i] = realised cash flow on path i at its current (provisional)
    # stopping time; time_idx[i] = the step index of that stopping time. Start at
    # maturity with the terminal intrinsic value.
    cashflow = np.maximum(K - paths[:, -1], 0.0)
    stop_idx = np.full(n_paths, n_steps, dtype=int)

    # Walk backward over the interior exercise dates t = n_steps-1, ..., 1.
    # (t = 0 is inception: not an exercise date, just the date cash flows are discounted to.)
    for t in range(n_steps - 1, 0, -1):
        S_t = paths[:, t]
        intrinsic = np.maximum(K - S_t, 0.0)         # payoff if exercised now
        itm = intrinsic > 0.0                        # in-the-money paths (regress on these)

        # Discount each path's current realised cash flow back from its stopping
        # time to t; this is the regression target Y (continuation payoff).
        y = cashflow * discount ** (stop_idx - t)
        cont = continuation_values(S_t, y, itm, build)   # E[continuation | S_t] via OLS

        # Exercise where in the money and immediate payoff >= estimated continuation.
        exercise = itm & (intrinsic >= cont)
        cashflow[exercise] = intrinsic[exercise]     # lock in the exercise payoff
        stop_idx[exercise] = t                       # ...and move the stopping time to t

    # Discount every path's realised cash flow from its stopping time to t=0.
    pv = cashflow * discount ** stop_idx
    price = pv.mean()
    stderr = pv.std(ddof=1) / np.sqrt(n_paths)
    result = LSMCResult(price=price, stderr=stderr, n_paths=n_paths)
    # Section 2 control variate needs the raw per-path PVs, not just the summary.
    return (result, pv) if return_pv else result


def price_american_put_on_paths_2f(S_paths: np.ndarray, v_paths: np.ndarray,
                                   K: float, r: float, T: float, *,
                                   basis: str = "poly2d_sv", s_scale: float = 1.0,
                                   v_scale: float = 1.0) -> LSMCResult:
    """Two-factor LSMC American put: regress continuation on state (S, v).

    Same backward induction as price_american_put_on_paths, but the
    continuation-value regression uses both the spot and the instantaneous
    variance as explanatory variables (Longstaff-Schwartz multi-factor
    prescription, paper p.123). This captures the genuine dependence of the
    continuation value on current variance under Heston -- higher v means more
    time value -- which the spot-only regression ignores.

    S_paths and v_paths share shape (n_paths, n_steps + 1). Each state
    column is divided by its scale (s_scale = strike; v_scale = a
    representative variance level such as theta or v0) so both regressors are
    O(1) -- without this the OLS design matrix is badly scaled (v ~ 0.006 next to
    S ~ 1) and the fit degrades.
    """
    n_paths, n_cols = S_paths.shape
    n_steps = n_cols - 1
    dt = T / n_steps
    discount = np.exp(-r * dt)
    build = _basis.BASES_2D[basis]

    # Realised cash flow / stopping index, initialised at terminal intrinsic value.
    cashflow = np.maximum(K - S_paths[:, -1], 0.0)
    stop_idx = np.full(n_paths, n_steps, dtype=int)

    for t in range(n_steps - 1, 0, -1):
        S_t = S_paths[:, t]
        intrinsic = np.maximum(K - S_t, 0.0)         # payoff if exercised now
        itm = intrinsic > 0.0                        # in-the-money paths

        # Two-column state [S/s_scale, v/v_scale], both O(1) for conditioning.
        state = np.column_stack([S_t / s_scale, v_paths[:, t] / v_scale])
        y = cashflow * discount ** (stop_idx - t)
        cont = continuation_values(state, y, itm, build)   # regress on (S, v)

        exercise = itm & (intrinsic >= cont)
        cashflow[exercise] = intrinsic[exercise]     # lock in exercise payoff
        stop_idx[exercise] = t                       # ...and its stopping time

    pv = cashflow * discount ** stop_idx
    return LSMCResult(price=pv.mean(),
                      stderr=pv.std(ddof=1) / np.sqrt(n_paths), n_paths=n_paths)


def price_american_put_gbm(S0: float, K: float, r: float, sigma: float, T: float,
                           *, n_steps: int = 50, n_paths: int = 100_000,
                           basis: str = "laguerre", scheme: str = "euler",
                           antithetic: bool = True, exact: bool = False,
                           rng: np.random.Generator | None = None) -> LSMCResult:
    """Simulate GBM paths and price an American put by LSMC.

    n_steps is the total number of steps (= exercise dates) over [0, T]. For
    Table 1 that is 50 * T (50 exercise points per year); callers pass it in.
    Set exact=True to use the bias-free log-Euler GBM (isolates Monte Carlo
    error from discretisation error); scheme selects Euler vs Milstein for
    the plain path generator used in the discretisation-step study.
    """
    # exact log-Euler (no discretisation bias) vs plain Euler/Milstein stepper.
    sim = simulate_gbm_exact if exact else simulate_gbm
    if exact:
        paths = sim(S0, r, sigma, T, n_steps, n_paths,
                    antithetic=antithetic, rng=rng)  # exact sim has no scheme arg
    else:
        paths = sim(S0, r, sigma, T, n_steps, n_paths, scheme=scheme,
                    antithetic=antithetic, rng=rng)
    return price_american_put_on_paths(paths, K, r, T, basis=basis)


def price_american_put_heston(S0: float, K: float, r: float, v0: float,
                              kappa: float, theta: float, xi: float, rho: float,
                              T: float, *, n_steps: int = 50,
                              n_paths: int = 100_000, basis: str = "laguerre",
                              scheme: str = "euler", antithetic: bool = True,
                              two_factor: bool = False,
                              basis_2f: str = "poly2d_sv",
                              rng: np.random.Generator | None = None) -> LSMCResult:
    """Simulate Heston paths and price an American put by LSMC.

    two_factor=False (default): regress the continuation value on spot only.
    The variance path still enters through its effect on the simulated spot, and
    the put's exercise decision is dominated by the price level, so this is a
    reasonable (and much simpler) approximation.

    two_factor=True: regress on both spot and variance using basis_2f
    (the LS multi-factor prescription). The Heston state is genuinely (S, v), so
    this removes the small exercise-boundary bias from ignoring current variance.
    Variance is scaled by theta (its long-run level) for O(1) conditioning.
    """
    S, v = simulate_heston(S0, r, v0, kappa, theta, xi, rho, T, n_steps,
                           n_paths, scheme=scheme, antithetic=antithetic, rng=rng)
    if two_factor:
        # Scale variance by its long-run level theta (fall back to v0) so the
        # (S, v) regressors are comparable in magnitude.
        v_scale = theta if theta > 0 else (v0 if v0 > 0 else 1.0)
        return price_american_put_on_paths_2f(S, v, K, r, T, basis=basis_2f,
                                              s_scale=K, v_scale=v_scale)
    return price_american_put_on_paths(S, K, r, T, basis=basis)  # spot-only regression
