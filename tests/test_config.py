"""Tests for the YAML config loader (lsmc.config)."""

import pytest

from lsmc.config import AttrDict, load_config
from lsmc.basis import BASES


def test_section1_loads_and_has_expected_fields():
    cfg = load_config("section1")
    assert cfg.common.basis in BASES
    assert cfg.common.n_paths > 0
    assert len(cfg.table1.reference) == 16
    assert set(cfg.heston.sweeps.keys()) == {"kappa", "theta", "xi", "rho"}


def test_yaml_suffix_optional():
    """Passing the name with or without .yaml resolves to the same file."""
    assert load_config("section1") == load_config("section1.yaml")


def test_unknown_basis_rejected():
    from lsmc.config import _validate
    with pytest.raises(ValueError, match="unknown basis"):
        _validate({"common": {"basis": "chebyshev"}})


def test_unknown_scheme_rejected():
    from lsmc.config import _validate
    with pytest.raises(ValueError, match="unknown scheme"):
        _validate({"discretisation": {"scheme": "runge_kutta"}})


def test_unknown_basis_2f_rejected():
    from lsmc.config import _validate
    with pytest.raises(ValueError, match="unknown basis_2f"):
        _validate({"heston": {"basis_2f": "poly3d"}})


def test_section1_has_two_factor_settings():
    cfg = load_config("section1")
    assert isinstance(cfg.heston.two_factor, bool)
    assert cfg.heston.basis_2f in {"poly2d_sv"}


def test_discretisation_fine_grid_divisible():
    """Nested-grid study needs fine_steps divisible by every step size."""
    cfg = load_config("section1")
    dc = cfg.discretisation
    for d in dc.step_days:
        assert dc.fine_steps % d == 0, f"fine_steps {dc.fine_steps} not divisible by {d}"


def test_heston_moneyness_axis_and_feller():
    """Moneyness axis present; base case and every sweep point satisfy Feller."""
    cfg = load_config("section1")
    assert len(cfg.heston.moneyness) >= 3
    b = cfg.heston.base
    # Feller must hold as each parameter is swept (others held at base).
    for kappa in cfg.heston.sweeps["kappa"]:
        assert 2 * kappa * b["theta"] > b["xi"] ** 2
    for theta in cfg.heston.sweeps["theta"]:
        assert 2 * b["kappa"] * theta > b["xi"] ** 2
    for xi in cfg.heston.sweeps["xi"]:
        assert 2 * b["kappa"] * b["theta"] > xi ** 2
    # Assignment requires positive spot-vol correlation.
    assert b["rho"] > 0 and all(r >= 0 for r in cfg.heston.sweeps["rho"])


def test_feller_violation_warns():
    from lsmc.config import _validate
    bad = {"heston": {"base": {"kappa": 0.1, "theta": 0.01, "xi": 0.5}}}
    with pytest.warns(UserWarning, match="Feller"):
        _validate(bad)


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist")


def test_attrdict_nested_access():
    d = AttrDict({"a": {"b": 1}})
    assert d.a.b == 1
