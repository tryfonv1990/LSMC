"""Tests for the European-put control variate (lsmc.control_variate)."""

import numpy as np
import pytest

from lsmc.analytic import baw_american_put, bs_put
from lsmc.control_variate import cv_american_put


def test_cv_reduces_variance():
    """The control-variate estimator must have smaller variance than the plain one."""
    rng = np.random.default_rng(0)
    res = cv_american_put(40, 40, 0.06, 0.2, 1.0, n_steps=50, n_paths=40000, rng=rng)
    assert res.cv_stderr < res.plain_stderr
    assert res.var_reduction > 1.0


def test_cv_correlation_bounds():
    """corr(American, European) is a genuine correlation in (0, 1]."""
    rng = np.random.default_rng(1)
    res = cv_american_put(40, 40, 0.06, 0.2, 1.0, n_steps=50, n_paths=40000, rng=rng)
    assert 0.0 < res.corr <= 1.0


def test_cv_price_matches_baw_at_positive_rate():
    """CV price agrees with the BAW American benchmark to within a few stderr."""
    rng = np.random.default_rng(2)
    res = cv_american_put(40, 40, 0.06, 0.2, 1.0, n_steps=50, n_paths=60000, rng=rng)
    baw = baw_american_put(40, 40, 0.06, 0.2, 1.0)
    assert res.cv_price == pytest.approx(baw, abs=4 * res.cv_stderr + 0.05)


def test_cv_price_matches_european_at_zero_rate():
    """At r = 0 the American put equals the European put; CV should recover it."""
    rng = np.random.default_rng(3)
    res = cv_american_put(40, 40, 0.0, 0.2, 1.0, n_steps=50, n_paths=60000, rng=rng)
    euro = bs_put(40, 40, 0.0, 0.2, 1.0)
    assert res.cv_price == pytest.approx(euro, abs=4 * res.cv_stderr + 0.02)
    # And the correlation is even higher than at r > 0 (the payoffs coincide).
    assert res.corr > 0.8
