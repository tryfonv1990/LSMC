"""Control-variate variance reduction -- assignment Section 2.

Uses the European put (with its analytic Black-Scholes value as the known mean)
as a control variate for the American put estimated on the same paths:

    X_cv = X_american - beta * (Y_european - E[Y_european]),

with beta estimated from the sample covariance. Provides diagnostics required by
the report:

* correlation between the American and European put payoffs vs lognormal vol,
* variance of the plain estimator vs the control-variate estimator.

Time discretisation is fixed at one day for this section.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .american import price_american_put_on_paths
from .analytic import bs_put
from .processes import simulate_gbm


@dataclass
class CVResult:
    """American-put price with and without the European-put control variate.

    All prices are present values (discounted to t=0). stderr fields are the
    Monte Carlo standard errors of the respective estimators; the whole point of
    the control variate is that cv_stderr < plain_stderr.
    """
    plain_price: float          # LSMC mean, no variance reduction
    plain_stderr: float
    cv_price: float             # control-variate-adjusted mean
    cv_stderr: float
    beta: float                 # optimal regression coefficient cov(X,Y)/var(Y)
    corr: float                 # corr(American PV, European PV) -- drives the gain
    var_reduction: float        # plain variance / cv variance (>= 1)
    n_paths: int


def cv_american_put(S0: float, K: float, r: float, sigma: float, T: float, *,
                    n_steps: int = 252, n_paths: int = 100_000,
                    basis: str = "laguerre", antithetic: bool = True,
                    rng: np.random.Generator | None = None) -> CVResult:
    """Price an American put with a European-put control variate on shared paths.

    The control variate exploits that the European put's exact mean is known
    (Black-Scholes), so subtracting its centred payoff cancels the part of the
    American payoff's noise that is common to both -- which is almost all of it,
    since the two options share the same terminal spot. Steps:

    1. Simulate GBM paths ONCE (Euler, one-day grid) so the American estimator
       X and the European control Y ride the identical randomness.
    2. X = per-path discounted American PV (LSMC backward induction).
    3. Y = per-path discounted European payoff exp(-rT) max(K - S_T, 0), whose
       known mean is E[Y] = bs_put(S0, K, r, sigma, T).
    4. beta* = cov(X, Y) / var(Y) minimises the variance of the combined
       estimator X_cv = X - beta*(Y - E[Y]) (standard optimal-control-variate).
    5. Report both estimators' means/standard errors and the variance reduction.
    """
    rng = np.random.default_rng() if rng is None else rng
    # One shared set of Euler-Maruyama GBM paths for both instruments.
    paths = simulate_gbm(S0, r, sigma, T, n_steps, n_paths,
                         scheme="euler", antithetic=antithetic, rng=rng)

    # X: per-path discounted American PV from the LSMC pricer (return_pv=True).
    _, X = price_american_put_on_paths(paths, K, r, T, basis=basis,
                                       return_pv=True)

    # Y: per-path discounted European put payoff on the SAME terminal spots.
    disc_T = np.exp(-r * T)
    Y = disc_T * np.maximum(K - paths[:, -1], 0.0)
    EY = bs_put(S0, K, r, sigma, T)              # exact known mean of the control

    # Optimal beta and the variance-reduced estimator. cov/var via a 2x2 sample
    # covariance matrix (ddof=1 for the usual unbiased estimates).
    cov = np.cov(X, Y, ddof=1)                   # [[var X, cov XY], [cov XY, var Y]]
    var_X, var_Y, cov_XY = cov[0, 0], cov[1, 1], cov[0, 1]
    beta = cov_XY / var_Y if var_Y > 0 else 0.0
    corr = cov_XY / np.sqrt(var_X * var_Y) if var_X > 0 and var_Y > 0 else 0.0

    X_cv = X - beta * (Y - EY)                   # centred control leaves E[X_cv]=E[X]
    n = n_paths
    plain_price, cv_price = X.mean(), X_cv.mean()
    plain_stderr = np.sqrt(var_X / n)
    var_cv = X_cv.var(ddof=1)
    cv_stderr = np.sqrt(var_cv / n)
    var_reduction = var_X / var_cv if var_cv > 0 else np.inf

    return CVResult(plain_price=plain_price, plain_stderr=plain_stderr,
                    cv_price=cv_price, cv_stderr=cv_stderr, beta=beta, corr=corr,
                    var_reduction=var_reduction, n_paths=n)
