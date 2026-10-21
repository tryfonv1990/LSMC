"""Ordinary-least-squares estimation of the conditional continuation value.

The heart of Longstaff-Schwartz: at each exercise date I estimate
E[discounted future cash flow | current state] by regressing realised future
cash flows onto basis functions of the state, using only in-the-money paths
(LS section 2.2 -- restricting to the region where the exercise decision is
actually relevant needs far fewer basis functions and lowers variance).

Kept dependency-light (numpy.linalg.lstsq, which uses an SVD and so tolerates
the near-collinear design matrices that some basis choices produce) so it can be
reused by the American pricer (Section 1) and the storage DP (Section 4).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def fit_predict(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit y ~ design by OLS and return the in-sample fitted values.

    lstsq solves the least-squares normal equations via SVD, which is robust
    to rank deficiency -- important because LS note (section 8.3) that some
    bases give a nearly singular cross-moment matrix. I only ever use the
    fitted values (the conditional expectation estimate), never individual
    coefficients, so collinearity is harmless.
    """
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)   # OLS coefficients (SVD)
    return design @ coef                                # fitted values = design . coef


def continuation_values(state: np.ndarray, discounted_cashflow: np.ndarray,
                        itm_mask: np.ndarray,
                        basis: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Estimate the continuation value for every path at one exercise date.

    Parameters
    ----------
    state : (n_paths,) or (n_paths, k) state variable(s) at this date -- spot
        alone (1-D) or several factors such as (spot, variance) for Heston (2-D).
        state[itm_mask] is passed straight to basis; the matching builder
        (from BASES or BASES_2D) decides how many columns to produce.
    discounted_cashflow : (n_paths,) realised future cash flow along each path,
        already discounted to this date -- the regression target Y.
    itm_mask : (n_paths,) bool, True where the option is in the money. The
        regression is fit on these paths only.
    basis : builder mapping states -> design matrix.

    Returns
    -------
    (n_paths,) array of estimated continuation values; entries for
    out-of-the-money paths are 0 (their exercise decision is never binding, so
    the value is not needed there).
    """
    cont = np.zeros_like(discounted_cashflow, dtype=float)
    if not np.any(itm_mask):
        return cont                                  # no ITM paths -> nothing to fit
    design = basis(state[itm_mask])                  # basis fns on ITM states only
    # Fitted values are the continuation estimate; leave OTM entries at 0.
    cont[itm_mask] = fit_predict(design, discounted_cashflow[itm_mask])
    return cont
