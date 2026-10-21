"""Regression basis functions for the least-squares continuation value.

The assignment flags the choice of basis as important. I expose several so the
report can show the robustness noted in Longstaff-Schwartz section 8.3 (their
American-put results are "virtually identical" across Laguerre, simple powers,
Hermite and trigonometric bases).

Each builder maps a 1-D array of state values (typically spot prices) to a
design matrix of shape (n_points, n_features) whose first column is the
constant term, ready for regression.fit_continuation.

My default for the American put is laguerre with the state normalised by
the strike (see american.py): the weighted Laguerre polynomials contain an
exp(-x/2) factor, so feeding raw prices of ~40 would underflow -- LS section 8.3
renormalises by the strike for exactly this reason.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import laguerre as _lag


def powers(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """Constant + simple monomials {1, x, x^2, ..., x^degree}."""
    x = np.asarray(x, dtype=float)
    return np.column_stack([x**k for k in range(degree + 1)])


def weighted_laguerre(x: np.ndarray, n: int = 3) -> np.ndarray:
    """Constant + first n weighted Laguerre polynomials exp(-x/2) L_k(x).

    Matches the LS American-put basis (their eqs. 2-4): a constant and three
    weighted Laguerre terms. x should be pre-scaled to O(1) (I divide by the
    strike) to keep the exponential well conditioned.
    """
    x = np.asarray(x, dtype=float)
    weight = np.exp(-x / 2.0)
    cols = [np.ones_like(x)]
    for k in range(n):
        coef = np.zeros(k + 1)
        coef[k] = 1.0                       # the k-th Laguerre polynomial L_k
        cols.append(weight * _lag.lagval(x, coef))
    return np.column_stack(cols)


def poly_2d(X: np.ndarray, degree: int = 2) -> np.ndarray:
    """Two-factor polynomial basis for state (s, v) -- the Heston case.

    X is an (n_points, 2) array whose columns are the two (pre-scaled)
    state variables, e.g. spot/strike and variance/theta. For degree=2 this
    returns {1, s, s^2, v, v^2, s*v} -- exactly the Longstaff-Schwartz
    multi-factor prescription (paper p.123): "terms in X and in Y, as well as
    cross-products". General degree returns every monomial s^i v^j with
    i + j <= degree.

    I use plain powers rather than weighted Laguerre here: the Laguerre exp(-x/2)
    weight is tuned to a single positive variable and does not transfer cleanly
    to two variables on different scales, and LS themselves use simple powers +
    cross-terms for their multi-factor examples. Callers should pre-scale each
    column to O(1) (see american.py) so the design matrix stays well conditioned.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("poly_2d expects an (n, 2) array of [s, v] states")
    s, v = X[:, 0], X[:, 1]
    cols = []
    # Every monomial s^i v^j with i + j <= degree (total-degree ordering).
    for total in range(degree + 1):
        for i in range(total + 1):          # i powers of s, (total-i) of v
            cols.append(s**i * v ** (total - i))
    return np.column_stack(cols)


def bspline_b6(x: np.ndarray, n_basis: int = 6, degree: int = 3,
               x_range: tuple[float, float] | None = None) -> np.ndarray:
    """Cubic B-spline design matrix -- the b6 basis for gas storage (Section 4).

    Boogert & de Jong (Exhibit 6) prefer B-splines for the storage regressions:
    local support makes them stable across the wide price range spanned by many
    paths, unlike global polynomials. Matches the paper's b6 exactly (footnote 4:
    Ramsay's create_bspline_basis with order 4, nbasis 6): 6 cubic B-splines on
    n_basis - degree - 1 = 2 evenly spaced interior knots. No separate constant
    column: B-splines sum to 1 on [lo, hi], so the constant is already spanned.
    """
    from scipy.interpolate import BSpline

    x = np.asarray(x, dtype=float)
    lo, hi = (x.min(), x.max()) if x_range is None else x_range
    # Clamped knot vector: repeat end knots (degree + 1) times so the basis spans
    # the whole [lo, hi] interval; interior knots evenly spaced.
    n_interior = n_basis - degree - 1                       # b6 cubic -> 2 knots
    interior = np.linspace(lo, hi, n_interior + 2)[1:-1]
    knots = np.concatenate([[lo] * (degree + 1), interior, [hi] * (degree + 1)])
    cols = []
    # Build each B-spline basis function by turning on one coefficient at a time.
    for j in range(n_basis):
        c = np.zeros(n_basis)
        c[j] = 1.0
        spline = BSpline(knots, c, degree, extrapolate=True)   # the j-th basis fn
        cols.append(spline(x))                                 # evaluate it at x
    return np.column_stack(cols)


# Registry of 1-D bases (state = spot only); callers select by name.
# "powers_deg2" is the exact {1, X, X^2} basis of the LS worked example.
BASES = {
    "powers": powers,
    "powers_deg2": lambda x: powers(x, degree=2),
    "laguerre": weighted_laguerre,
    "bspline_b6": bspline_b6,
}

# Registry of 2-D bases (state = [s, v]); kept separate because the call
# signature differs (an (n, 2) array rather than a 1-D vector).
BASES_2D = {
    "poly2d_sv": lambda X: poly_2d(X, degree=2),
}
