"""Load and validate experiment configuration from YAML.

All settings live in configs/*.yaml, not in the experiment scripts, so they
can be seen and changed in one place. The loader checks the few fields that must
match the code (mainly the regression basis name). A typo then gives a clear
error straight away, instead of a confusing one later in the run.

The config is returned as an AttrDict, so settings can be read as
cfg.common.basis; it is still a plain dict underneath.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .basis import BASES, BASES_2D

# Repository root = two levels up from this file (src/lsmc/config.py -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = _REPO_ROOT / "configs"

VALID_SCHEMES = {"euler", "milstein"}


class AttrDict(dict):
    """dict whose keys are also attributes; nested dicts are wrapped lazily."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return AttrDict(value) if isinstance(value, dict) else value


def _validate(cfg: dict) -> None:
    """Check the fields that must agree with code; raise ValueError on mismatch."""
    basis = cfg.get("common", {}).get("basis")
    if basis is not None and basis not in BASES:
        raise ValueError(
            f"unknown basis {basis!r}; choose one of {sorted(BASES)}"
        )
    for block in ("discretisation", "table1"):
        scheme = cfg.get(block, {}).get("scheme")
        if scheme is not None and scheme not in VALID_SCHEMES:
            raise ValueError(
                f"unknown scheme {scheme!r} in {block}; choose one of "
                f"{sorted(VALID_SCHEMES)}"
            )
    basis_2f = cfg.get("heston", {}).get("basis_2f")
    if basis_2f is not None and basis_2f not in BASES_2D:
        raise ValueError(
            f"unknown basis_2f {basis_2f!r}; choose one of {sorted(BASES_2D)}"
        )
    # Heston Feller sanity check (warn-level: the sim still runs if violated).
    base = cfg.get("heston", {}).get("base")
    if base and 2 * base["kappa"] * base["theta"] <= base["xi"] ** 2:
        import warnings
        warnings.warn(
            "Heston base case violates the Feller condition 2*kappa*theta > xi^2;"
            " the variance process may hit zero frequently.", stacklevel=2,
        )


def load_config(name: str) -> AttrDict:
    """Load configs/<name>.yaml (.yaml optional), validate, and return it.

    Example
    -------
    >>> cfg = load_config("section1")
    >>> cfg.common.basis
    'laguerre'
    """
    path = CONFIG_DIR / name
    if path.suffix != ".yaml":
        path = path.with_suffix(".yaml")
    if not path.exists():
        raise FileNotFoundError(f"no config at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    _validate(cfg)
    return AttrDict(cfg)
