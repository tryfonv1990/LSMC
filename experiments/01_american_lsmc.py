# %%
# Section 1 — American Options and Least Squares Monte Carlo
#
# LSMC American put under geometric Brownian motion (Euler-Maruyama):
# reproduce Longstaff-Schwartz Table 1, study the discretisation step
# (1 day -> 1 week) across volatilities, then repeat under the Heston
# stochastic-volatility model and examine the effect of kappa, theta, xi, rho.
#
# All parameters (basis, path counts, seeds, sweeps, ...) live in
# configs/section1.yaml -- edit that file, not this script. Run as
# interactive cells (PyCharm / VS Code # %%) or top-to-bottom as a script.
# Figures are written to ../results/figures/ and tables to
# ../results/tables/ for inclusion in ../report/report.md.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsmc.american import (price_american_put_gbm, price_american_put_heston,
                           price_american_put_on_paths)
from lsmc.analytic import bs_put
from lsmc.processes import draw_normals, simulate_gbm, simulate_gbm_exact
from lsmc.config import load_config

# Load every tunable parameter from configs/section1.yaml (see lsmc.config).
CFG = load_config("section1")
COMMON = CFG.common          # the shared block: strike, rate, n_paths, basis, ...
BASIS = COMMON.basis         # config common.basis -- one basis for the whole run

# Output locations, relative to the repo root (this file lives in experiments/).
FIG = Path(__file__).resolve().parents[1] / "results" / "figures"
TAB = Path(__file__).resolve().parents[1] / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

print(f"basis = {BASIS!r}, n_paths = {COMMON.n_paths}, strike = {COMMON.strike}")

# %%
# 1a. Reproduce Longstaff-Schwartz Table 1
#
# For each (S0, sigma, T) I price the American put by LSMC and compare with the
# paper's reported simulated American value. I also report the early-exercise
# premium (American - Black-Scholes European), the quantity the paper focuses on.
#
# Paths use Euler-Maruyama, as the brief requires for Section 1. As a check I
# also price every point with the bias-free exact log-Euler GBM driven by the
# same Brownian increments: on shared noise the difference euler_minus_exact
# is the pure Euler discretisation bias, so I can show it is small next to the
# Monte Carlo error at 50 steps per year.

# %%
t1 = CFG.table1                       # config table1 block: seed, reference, ...
K, R = COMMON.strike, COMMON.rate     # strike & rate shared with every experiment


def table1_paths(S0, sigma, T, i, scheme):
    """Paths for Table 1 point i: scheme in {'euler', 'milstein', 'exact'}.

    The Brownian increments come from the per-point child seed [seed, i], so every
    scheme (and the basis comparison below) runs on the identical noise.
    """
    n_steps = int(COMMON.exercise_per_year * T)   # 50 dates/yr (config) x maturity
    rng = np.random.default_rng([t1.seed, i])
    dW = draw_normals(COMMON.n_paths, n_steps, rng, COMMON.antithetic) \
        * np.sqrt(T / n_steps)
    if scheme == "exact":
        return simulate_gbm_exact(S0, R, sigma, T, n_steps, COMMON.n_paths, dW=dW)
    return simulate_gbm(S0, R, sigma, T, n_steps, COMMON.n_paths, scheme=scheme,
                        dW=dW)


rows = []
# One row per (S0, sigma, T) in config table1.reference (the paper's Table 1).
# Each point gets its own child seed [base_seed, i]: points are decorrelated (a
# fair Table 1 reproduction) yet the run is reproducible.
for i, ref in enumerate(t1.reference):
    S0, sigma, T = ref["S0"], ref["sigma"], ref["T"]
    res = price_american_put_on_paths(table1_paths(S0, sigma, T, i, t1.scheme),
                                      K, R, T, basis=BASIS)
    euro = bs_put(S0, K, R, sigma, T)               # BS European put (r from config)
    d = t1.round_decimals
    row = {
        "S0": S0, "sigma": sigma, "T": T,
        "LSMC_American": round(res.price, d), "s.e.": round(res.stderr, d),
        "paper_American": ref["value"], "paper_s.e.": ref["se"],
        "diff": round(res.price - ref["value"], d),
        # American - European = value of the right to exercise early.
        "LSMC_early_exercise_premium": round(res.price - euro, d),
    }
    if t1.compare_exact:
        exact = price_american_put_on_paths(table1_paths(S0, sigma, T, i, "exact"),
                                            K, R, T, basis=BASIS).price
        row["exact_American"] = round(exact, d)
        row["euler_minus_exact"] = round(res.price - exact, 4)
    rows.append(row)

table1 = pd.DataFrame(rows)
table1.to_csv(TAB / "1a_table1_reproduction.csv", index=False)
print(table1.to_string(index=False))
print(f"\nMax |diff| vs paper: {table1['diff'].abs().max():.3f}")
if t1.compare_exact:
    print(f"Max |Euler - exact| (pure discretisation bias): "
          f"{table1['euler_minus_exact'].abs().max():.4f}")

# %%
# Basis robustness (LS §8.3)
#
# The LSMC price should be insensitive to the choice of basis. I verify this by
# pricing a few Table 1 points on common random paths while varying only the
# regression basis. Crucially I reuse each point's exact Table 1 seed
# ([seed, i], i = its index in the reference list), so the laguerre column
# here is identical to that point's LSMC_American in table1_reproduction.csv --
# same paths, same basis, same number. All three bases should agree to within
# Monte Carlo error. This block writes basis_comparison.csv.

# %%
BASIS_CMP_POINTS = [(36, 0.20, 1), (40, 0.20, 1), (44, 0.40, 2)]
BASIS_CHOICES = ["powers_deg2", "powers", "laguerre"]
# Map each point to (paper value, its index in the reference list) so I can
# reuse the identical per-point seed Table 1 used above.
ref_lookup = {(r["S0"], r["sigma"], r["T"]): (r["value"], i)
              for i, r in enumerate(t1.reference)}

cmp_rows = []
for S0, sigma, T in BASIS_CMP_POINTS:
    paper_val, idx = ref_lookup[(S0, sigma, T)]
    # Same seed as Table 1's point idx => byte-identical paths, so the laguerre
    # price below matches table1_reproduction.csv exactly. Simulate ONCE, then
    # reuse the paths for every basis (common random numbers across bases).
    paths = table1_paths(S0, sigma, T, idx, t1.scheme)
    row = {"S0": S0, "sigma": sigma, "T": T}
    for b in BASIS_CHOICES:
        row[b] = round(price_american_put_on_paths(paths, K, R, T, basis=b).price,
                       t1.round_decimals)
    row["paper"] = paper_val
    cmp_rows.append(row)

basis_comparison = pd.DataFrame(cmp_rows)
basis_comparison.to_csv(TAB / "1a_basis_comparison.csv", index=False)
print("\nBasis robustness on common paths (LS §8.3):")
print(basis_comparison.to_string(index=False))

# %%
# 1b. Impact of the discretisation step (1 day -> 1 week)
#
# The plain Euler-Maruyama scheme carries a time-discretisation bias on top of
# Monte Carlo noise. To see the bias cleanly I must remove the noise -- and
# the correct way is a nested-grid design:
#
# 1. Draw ONE fine Brownian path (fine_steps increments) per (sigma) case.
# 2. For each coarse step size d, build the coarse Brownian increments by
#    summing d consecutive fine increments. Because Brownian increments are
#    additive (W over a big interval = sum of W over its sub-intervals), the
#    coarse path is an EXACT sub-sample of the same underlying Brownian motion.
# 3. Price with Euler on that coarse path, and with the exact scheme on the SAME
#    coarse path. Their difference is the pure discretisation bias at step d.
#
# Since every coarse grid (and its exact reference) rides the same randomness,
# Monte Carlo noise cancels across the whole curve -- not just point by point --
# so the bias is monotonic in step size, as theory predicts. (An earlier version
# drew independent paths per step size; the leftover noise then swamped the small
# low-volatility bias and scrambled the ordering.)

# %%
dc = CFG.discretisation               # config discretisation block
S0, T = dc.S0, dc.T                   # fixed spot & maturity for this study
fine = dc.fine_steps                  # fine grid (divisible by every step_days)
n_paths = dc.n_paths                  # dedicated (larger) path count for this study

dt_rows = []
for sigma in dc.sigmas:               # config discretisation.sigmas
    # One fine Brownian path for this volatility case, shared by all step sizes.
    # Shape (n_paths, fine); scaled by sqrt(dt_fine) so increments ~ N(0, dt_fine).
    rng = np.random.default_rng([dc.seed, int(sigma * 1000)])
    dt_fine = T / fine
    dW_fine = draw_normals(n_paths, fine, rng, COMMON.antithetic) * np.sqrt(dt_fine)

    for d in dc.step_days:            # step sizes in trading days (config)
        n_steps = fine // d           # exact: fine is divisible by d (see config)
        # Coarse increment k = sum of the d fine increments inside it. Reshape to
        # (n_paths, n_steps, d) and sum the last axis -> (n_paths, n_steps).
        dW_coarse = dW_fine[:, :n_steps * d].reshape(n_paths, n_steps, d).sum(axis=2)

        # Same coarse Brownian path fed to BOTH schemes: the only difference is
        # Euler stepping vs the exact log-Euler solution => the pure bias.
        euler_paths = simulate_gbm(S0, R, sigma, T, n_steps, n_paths,
                                   scheme=dc.scheme, dW=dW_coarse)
        exact_paths = simulate_gbm_exact(S0, R, sigma, T, n_steps, n_paths,
                                         dW=dW_coarse)
        euler = price_american_put_on_paths(euler_paths, K, R, T, basis=BASIS).price
        exact = price_american_put_on_paths(exact_paths, K, R, T, basis=BASIS).price
        dt_rows.append({"sigma": sigma, "step_days": d, "n_steps": n_steps,
                        "price": round(euler, dc.round_decimals),
                        "bias_vs_exact": round(euler - exact, dc.round_decimals)})

dt_table = pd.DataFrame(dt_rows)
dt_table.to_csv(TAB / "1b_discretisation_step_study.csv", index=False)
print(dt_table.to_string(index=False))

# Plot: absolute discretisation bias vs step size, one line per volatility.
fig, ax = plt.subplots(figsize=(7, 4.5))
for sigma in dc.sigmas:
    sub = dt_table[dt_table["sigma"] == sigma]
    ax.plot(sub["step_days"], sub["bias_vs_exact"].abs(), marker="o",
            label=f"$\\sigma$ = {sigma:.2f}")
ax.set_xlabel("time step (trading days)")
ax.set_ylabel("|Euler bias| vs exact (shared Brownian path)")
ax.set_title("American put: discretisation bias grows with step size and volatility")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "1b_discretisation_bias.png", dpi=130)
print(f"saved {FIG / '1b_discretisation_bias.png'}")

# %%
# 1c. Heston stochastic volatility: parameter sensitivities
#
# I price the American put under Heston and vary each parameter one at a time
# around a base case, at three moneyness levels (ITM / ATM / OTM for the
# put). Reporting across moneyness is essential: xi and rho act on the skew and
# tails of the terminal distribution, which barely move the ATM price but shift
# ITM/OTM prices. The base case is chosen to respect the assignment's regime
# (v0 = 0.08^2 fixed, rho positive) while leaving room for each effect to show:
# theta (0.12^2) > v0 gives kappa a gap to mean-revert across.
#
# Qualitative expectations:
# * higher theta -> higher average variance -> more valuable put (all moneyness);
# * higher xi (vol of variance) -> fatter variance tails; net price effect is
#   small and non-monotonic at r=0 (lifts deep-ITM slightly, nudges ATM/OTM down);
# * higher kappa -> variance pulled from v0 toward theta faster -> since
#   theta > v0 here, faster reversion raises the average variance and the price;
# * rho (positive here) tilts the skew; its effect is largest away from ATM.
# Base kappa is 5.0 so the Feller bound 2*kappa*theta=0.144 lets xi sweep to 0.35;
# the condition 2*kappa*theta > xi^2 holds at every sweep point (checked above).

# %%
hc = CFG.heston                       # config heston block
BASE = dict(hc.base)                  # base-case parameters (config heston.base)
MONEYNESS = list(hc.moneyness)        # ITM/ATM/OTM spots (config heston.moneyness)


def heston_price(*, two_factor=None, **overrides):
    """Price the Heston put; **overrides replace individual BASE params for a sweep.

    e.g. heston_price(kappa=4.0) prices the base case with kappa swapped out. The
    two_factor flag defaults to config heston.two_factor (spot-only vs (S, v)).
    """
    tf = hc.two_factor if two_factor is None else two_factor
    p = {**BASE, **overrides}                           # base case, then overrides
    rng = np.random.default_rng(hc.seed)                # common seed => same paths
    return price_american_put_heston(
        p["S0"], p["K"], p["r"], p["v0"], p["kappa"], p["theta"], p["xi"],
        p["rho"], p["T"], n_steps=hc.n_steps, n_paths=hc.n_paths,
        basis=BASIS, two_factor=tf, basis_2f=hc.basis_2f,
        antithetic=COMMON.antithetic, rng=rng).price


# --- Effect of adding variance to the regression basis, across moneyness. ------
# Spot-only vs two-factor (S, v) on common random numbers (same seed => same
# paths). Expectation: negligible ATM, largest ITM where early exercise is live.
basis_rows = []
for S0 in MONEYNESS:
    so = heston_price(S0=S0, two_factor=False)
    tf = heston_price(S0=S0, two_factor=True)
    basis_rows.append({"S0": S0, "spot_only": round(so, hc.round_decimals),
                       "two_factor": round(tf, hc.round_decimals),
                       "diff": round(tf - so, hc.round_decimals)})
basis_cmp = pd.DataFrame(basis_rows)
basis_cmp.to_csv(TAB / "1c_heston_basis_comparison.csv", index=False)
print("\nHeston regression basis: spot-only vs two-factor (S, v), by moneyness:")
print(basis_cmp.to_string(index=False))

# --- Parameter sensitivities: one subplot per parameter, one line per moneyness.
# The three moneyness prices differ ~50x (ITM ~0.15 vs OTM ~0.003), so plotting
# raw prices on one shared axis crushes the small curves. I instead plot the
# RELATIVE change price/price(base): every line passes through 1.0 at the base
# value (each sweep includes it), so each curve's
# slope is directly comparable regardless of its absolute level. The CSV still
# stores absolute prices (plus the ratio) for exact figures.
sweeps = dict(hc.sweeps)              # {param: [values]} from config heston.sweeps
# Base-case price per moneyness (all params at BASE) = the normaliser for its line.
base_price = {S0: heston_price(S0=S0) for S0 in MONEYNESS}
fig, axes = plt.subplots(2, 2, figsize=(11, 8))   # 4 params -> 2x2 grid
hest_rows = []
for ax, (param, values) in zip(axes.ravel(), sweeps.items()):
    for S0 in MONEYNESS:
        # Sweep one param over its config values, holding the rest at BASE.
        prices = [heston_price(S0=S0, **{param: v}) for v in values]
        rel = [pr / base_price[S0] for pr in prices]     # normalise to base-case
        for v, pr, rl in zip(values, prices, rel):
            hest_rows.append({"param": param, "S0": S0, "value": v,
                              "price": round(pr, hc.round_decimals),
                              "rel_to_base": round(rl, 3)})
        label = f"S0={S0} ({'ITM' if S0 < 1 else 'OTM' if S0 > 1 else 'ATM'})"
        ax.plot(values, rel, marker="o", label=label)   # plot the ratio, not price
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")       # base-case reference (rel=1)
    # Vertical marker at the base value of this param: every line crosses 1.0 here,
    # since price/base-price = 1 when the swept param equals its base-case setting.
    ax.axvline(BASE[param], color="grey", lw=0.8, ls=":",
               label=f"base {param}={BASE[param]:g}")
    ax.set_xlabel(param)
    ax.set_ylabel("price / base-case price")
    ax.set_title(f"Sensitivity to {param}")
    ax.legend(fontsize=8)
fig.suptitle("Heston American put: relative parameter sensitivities across moneyness", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "1c_heston_sensitivities.png", dpi=130, bbox_inches="tight")

hest_table = pd.DataFrame(hest_rows)
hest_table.to_csv(TAB / "1c_heston_sensitivities.csv", index=False)
print(hest_table.to_string(index=False))
print(f"saved {FIG / '1c_heston_sensitivities.png'}")
