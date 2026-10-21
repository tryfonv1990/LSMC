"""lsmc — a minimal Least-Squares Monte Carlo engine for early-exercise pricing.

The package implements the Longstaff-Schwartz (2001) regression-based Monte
Carlo method and extends it, per the interview assignment, to:

* American puts under geometric Brownian motion and Heston stochastic vol,
* variance reduction via control variates (Black-Scholes European put) and the
  Barone-Adesi-Whaley (1987) analytic American approximation,
* higher-order (Milstein) time stepping for improved SDE convergence,
* natural-gas storage valuation following Boogert & de Jong (2008).

Sub-modules
-----------
processes        GBM & Heston path simulation (Euler-Maruyama and Milstein).
basis            Regression basis builders (powers, Laguerre, B-spline b6).
regression       OLS continuation-value estimation on in-the-money paths.
american         LSMC American put pricer (Section 1).
analytic         Black-Scholes European put and Barone-Adesi-Whaley American.
control_variate  Control-variate estimator and variance diagnostics (Section 2).
greeks           Gamma profile and estimator-variance tooling (Section 2).
storage          Gas day-ahead price simulation and storage DP (Section 4).
"""

__version__ = "0.1.0"
