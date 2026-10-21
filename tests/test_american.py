"""Tests for the LSMC American put pricer (lsmc.american)."""

import numpy as np
import pytest

from lsmc.american import price_american_put_gbm, price_american_put_on_paths

# The Longstaff-Schwartz (2001) section-1 stylised 8-path stock-price matrix.
LS_PATHS = np.array([
    [1.00, 1.09, 1.08, 1.34],
    [1.00, 1.16, 1.26, 1.54],
    [1.00, 1.22, 1.07, 1.03],
    [1.00, 0.93, 0.97, 0.92],
    [1.00, 1.11, 1.56, 1.52],
    [1.00, 0.76, 0.77, 0.90],
    [1.00, 0.92, 0.84, 1.01],
    [1.00, 0.88, 1.22, 1.34],
])


def test_ls_eight_path_example():
    """Reproduce the paper's worked example: American put value = 0.1144.

    Uses the paper's exact basis (constant + X + X^2, i.e. degree-2 powers) and
    its parameters (K=1.10, r=6%/period, 3 periods).
    """
    res = price_american_put_on_paths(LS_PATHS, K=1.10, r=0.06, T=3.0,
                                      basis="powers_deg2")
    assert res.price == pytest.approx(0.1144, abs=5e-4)


# A subset of LS Table 1: (S0, sigma, T) -> (simulated American, s.e.).
# Strike 40, r = 0.06, 50 exercise dates/year, 100k (50k+50k antithetic) paths.
TABLE1 = [
    (36, 0.20, 1, 4.472, 0.010),
    (40, 0.20, 1, 2.313, 0.009),
    (40, 0.20, 2, 2.879, 0.010),
    (44, 0.40, 2, 5.622, 0.021),
]


@pytest.mark.parametrize("S0,sigma,T,ref,se", TABLE1)
def test_table1_within_tolerance(S0, sigma, T, ref, se):
    """LSMC matches Table 1 within a few Monte Carlo standard errors."""
    rng = np.random.default_rng(12345)
    res = price_american_put_gbm(S0, 40, 0.06, sigma, T, n_steps=int(50 * T),
                                 n_paths=100_000, basis="laguerre", exact=True,
                                 rng=rng)
    # Allow ~4 combined standard errors (paper s.e. + my s.e.).
    tol = 4 * (se + res.stderr)
    assert abs(res.price - ref) < tol


def test_american_ge_european_zero_rate():
    """With r=0 the American put still >= its European (terminal-only) value."""
    rng = np.random.default_rng(7)
    from lsmc.processes import simulate_gbm_exact
    paths = simulate_gbm_exact(40, 0.0, 0.3, 1.0, 50, 60_000, rng=rng)
    american = price_american_put_on_paths(paths, 40, 0.0, 1.0, basis="laguerre")
    european = np.maximum(40 - paths[:, -1], 0.0).mean()
    assert american.price >= european - 3 * american.stderr
