# %%
# Section 2 — Control Variates for American Options
#
# Variance reduction for the Section-1 American-put LSMC pricer under geometric
# Brownian motion (Euler-Maruyama, one-day steps). Three studies:
#   2a. European-put control variate: correlation with the American put and the
#       variance of the plain vs control-variate estimator across volatilities;
#   2b. Barone-Adesi-Whaley (1987) analytic benchmark: error of the LSMC price
#       with and without the control variate for one-month puts;
#   2c. Gamma profile (gamma vs spot) by finite differences with common random
#       numbers, plus the variance of the gamma estimator.
#
# The assignment fixes ALL rates to zero here. At r = 0 the American put has no
# early-exercise premium (American = European), so I ALSO run a supplementary
# r = 0.06 case for a non-degenerate picture -- see the report for the why.
#
# All parameters live in configs/section2.yaml -- edit that file, not this
# script. Run as interactive cells (# %%) or top-to-bottom. Figures go to
# ../results/figures/ and tables to ../results/tables/.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsmc.american import price_american_put_gbm
from lsmc.analytic import baw_american_put, bs_put
from lsmc.control_variate import cv_american_put
from lsmc.greeks import gamma_profile
from lsmc.config import load_config

# Load every tunable parameter from configs/section2.yaml (see lsmc.config).
CFG = load_config("section2")
COMMON = CFG.common          # shared block: strike, n_paths, rates, basis, ...
K = COMMON.strike            # strike, shared across all Section-2 studies
BASIS = COMMON.basis         # continuation-value regression basis
RATES = list(COMMON.rates)   # [0.0 (mandated), 0.06 (supplementary)]
DPY = COMMON.days_per_year   # one-day step => n_steps = round(T * DPY)

# Output locations, relative to the repo root (this file lives in experiments/).
FIG = Path(__file__).resolve().parents[1] / "results" / "figures"
TAB = Path(__file__).resolve().parents[1] / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

print(f"basis = {BASIS!r}, strike = {K}, rates = {RATES}, days/yr = {DPY}")

# %%
# 2a. European-put control variate: correlation and variance reduction
#
# The European put shares the American put's terminal spot, so their payoffs are
# highly correlated, yet the European mean is known exactly (Black-Scholes). The
# optimal control variate X_cv = X - beta(Y - E[Y]) with beta = cov(X,Y)/var(Y)
# therefore removes most of the estimator's variance. I sweep lognormal vol and
# plot (i) corr(American, European) and (ii) the variance-reduction ratio, for
# both r = 0 and r = 0.06. Expectation: corr near 1 (near-total reduction) at
# r = 0 where the two options coincide; slightly lower, but still large, at
# r = 0.06 where the American carries an early-exercise premium the control cannot
# track.

# %%
cc = CFG.correlation                  # config correlation block: seed, sigmas, ...
n_steps_1y = int(round(cc.T * DPY))   # one-day grid over the 1-year horizon
corr_rows = []
for r in RATES:
    for sigma in cc.sigmas:
        # Fresh, reproducible paths per (rate, vol); seed folds in both so the
        # two rate curves are independent but the run is repeatable.
        rng = np.random.default_rng([cc.seed, int(r * 1000), int(sigma * 1000)])
        res = cv_american_put(cc.S0, K, r, sigma, cc.T, n_steps=n_steps_1y,
                              n_paths=cc.n_paths, basis=BASIS,
                              antithetic=COMMON.antithetic, rng=rng)
        corr_rows.append({
            "rate": r, "sigma": sigma,
            "corr": round(res.corr, cc.round_decimals),
            "var_plain": res.plain_stderr ** 2 * res.n_paths,   # sample variance
            "var_cv": res.cv_stderr ** 2 * res.n_paths,
            "var_reduction": round(res.var_reduction, 2),
            "plain_price": round(res.plain_price, cc.round_decimals),
            "cv_price": round(res.cv_price, cc.round_decimals),
        })
corr_table = pd.DataFrame(corr_rows)
corr_table.to_csv(TAB / "2a_cv_correlation.csv", index=False)
print("\nControl-variate correlation & variance reduction vs vol:")
print(corr_table.to_string(index=False))

# Two-panel figure: correlation (left) and variance-reduction ratio (right).
fig, (axc, axv) = plt.subplots(1, 2, figsize=(11, 4.5))
for r in RATES:
    sub = corr_table[corr_table["rate"] == r]
    axc.plot(sub["sigma"], sub["corr"], marker="o", label=f"r = {r:g}")
    axv.plot(sub["sigma"], sub["var_reduction"], marker="o", label=f"r = {r:g}")
axc.set_xlabel("lognormal volatility σ")
axc.set_ylabel("corr(American, European)")
axc.set_title("Correlation of the two put payoffs")
axc.legend()
axv.set_xlabel("lognormal volatility σ")
axv.set_ylabel("variance reduction (plain / CV)")
axv.set_title("Estimator variance reduction")
axv.legend()
fig.suptitle("Section 2a: European-put control variate for the American put", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "2a_cv_variance_reduction.png", dpi=130, bbox_inches="tight")
print(f"saved {FIG / '2a_cv_variance_reduction.png'}")

# %%
# 2b. Barone-Adesi-Whaley benchmark: error with vs without control variate
#
# BAW (1987) is a fast analytic approximation of the American put; I use it as
# the reference "true" value. For one-month puts at σ ∈ {0.05, 0.1, 0.2, 0.3} I
# compare the LSMC price error |LSMC − BAW| with and without the control variate.
# At r = 0 BAW returns the European value (no early-exercise premium — see the
# r = 0 guard in lsmc.analytic.baw_american_put); at r = 0.06 it carries a
# genuine premium. The control-variate error should be markedly smaller than the
# plain error, since both target the same mean but the CV estimator has far lower
# variance.

# %%
bw = CFG.baw_error                     # config baw_error block
n_steps_1m = int(round(bw.T * DPY))    # one-day grid over the one-month horizon
baw_rows = []
for r in RATES:
    for sigma in bw.sigmas:
        rng = np.random.default_rng([bw.seed, int(r * 1000), int(sigma * 1000)])
        res = cv_american_put(bw.S0, K, r, sigma, bw.T, n_steps=n_steps_1m,
                              n_paths=bw.n_paths, basis=BASIS,
                              antithetic=COMMON.antithetic, rng=rng)
        baw = baw_american_put(bw.S0, K, r, sigma, bw.T)   # analytic reference
        baw_rows.append({
            "rate": r, "sigma": sigma,
            "baw": round(baw, bw.round_decimals),
            "plain_price": round(res.plain_price, bw.round_decimals),
            "cv_price": round(res.cv_price, bw.round_decimals),
            "err_plain": round(abs(res.plain_price - baw), bw.round_decimals),
            "err_cv": round(abs(res.cv_price - baw), bw.round_decimals),
        })
baw_table = pd.DataFrame(baw_rows)
baw_table.to_csv(TAB / "2b_baw_error.csv", index=False)
print("\nBAW benchmark: |LSMC - BAW| with and without the control variate:")
print(baw_table.to_string(index=False))

# Plot absolute error vs vol, plain vs CV, one panel per rate.
fig, axes = plt.subplots(1, len(RATES), figsize=(11, 4.5), sharey=False)
for ax, r in zip(np.atleast_1d(axes), RATES):
    sub = baw_table[baw_table["rate"] == r]
    ax.plot(sub["sigma"], sub["err_plain"], marker="o", label="plain LSMC")
    ax.plot(sub["sigma"], sub["err_cv"], marker="s", label="with control variate")
    ax.set_xlabel("lognormal volatility σ")
    ax.set_ylabel("|LSMC − BAW|")
    ax.set_title(f"r = {r:g}")
    ax.legend()
fig.suptitle("Section 2b: one-month American put error vs BAW approximation", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "2b_cv_baw_error.png", dpi=130, bbox_inches="tight")
print(f"saved {FIG / '2b_cv_baw_error.png'}")

# %%
# 2c. Gamma profile and gamma-estimator variance
#
# Gamma (∂²V/∂S²) is computed by a central finite difference with common random
# numbers: the three bumped spots (S₀−h, S₀, S₀+h) are priced on the SAME
# Brownian path, so the differencing cancels the Monte Carlo noise and leaves the
# curvature. I repeat over independent batches to quantify the gamma-estimator
# variance the assignment asks for. Profiles are shown per volatility; gamma is
# positive (the put is convex in spot) and peaks near the money, where curvature
# is greatest. I use the mandated r = 0 here (Section 2's base regime).

# %%
gc = CFG.gamma                         # config gamma block
n_steps_g = int(round(gc.T * DPY))     # one-day grid over the option horizon
R_GAMMA = 0.0                          # mandated zero-rate regime for the greek
gamma_rows = []
for sigma in gc.sigmas:
    pts = gamma_profile(gc.spots, K, R_GAMMA, sigma, gc.T, h=gc.bump_h,
                        n_steps=n_steps_g, n_paths=gc.n_paths,
                        antithetic=COMMON.antithetic, n_batches=gc.n_batches,
                        seed=gc.seed, basis=BASIS)
    for p in pts:
        gamma_rows.append({
            "sigma": sigma, "S0": p.S0,
            "gamma": round(p.gamma, gc.round_decimals),
            "gamma_stderr": round(p.gamma_stderr, gc.round_decimals),
            "gamma_var": p.gamma_var,
        })
gamma_table = pd.DataFrame(gamma_rows)
gamma_table.to_csv(TAB / "2c_gamma_profile.csv", index=False)
print("\nGamma profile (gamma vs spot) and estimator variance, r = 0:")
print(gamma_table.to_string(index=False))

# Plot gamma vs spot, one line per vol, with the estimator standard-error band.
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for sigma in gc.sigmas:
    sub = gamma_table[gamma_table["sigma"] == sigma]
    ax.errorbar(sub["S0"], sub["gamma"], yerr=sub["gamma_stderr"],
                marker="o", capsize=3, label=f"σ = {sigma:g}")
ax.axvline(K, color="grey", lw=0.8, ls=":", label=f"strike K = {K:g}")
ax.set_xlabel("spot S₀")
ax.set_ylabel("gamma  (∂²V/∂S²)")
ax.set_title("Section 2c: American-put gamma profile (CRN finite difference, r = 0)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "2c_gamma_profile.png", dpi=130, bbox_inches="tight")
print(f"saved {FIG / '2c_gamma_profile.png'}")

# %%
# Appendix: numerical proof that American = European at r = 0
#
# The r = 0 mandate makes early exercise never optimal, so the American put must
# equal the European put. I check this directly: price the American put by LSMC
# at r = 0 for several (σ, T) and compare with the closed-form Black-Scholes
# European value and BAW. If the theory holds, the difference American − European
# is pure Monte Carlo noise — i.e. within a fraction of the LSMC standard error —
# and BAW (via its r = 0 guard) equals the European value exactly. Written to
# results/tables/appendix/ since it backs a claim in the report rather than
# being one of the three main Section-2 deliverables.

# %%
ap = CFG.appendix                      # config appendix block
APP = TAB / "appendix"                 # results/tables/appendix/
APP.mkdir(parents=True, exist_ok=True)
app_rows = []
for case in ap.cases:
    sigma, T = case["sigma"], case["T"]
    # Reproducible paths per case; r = 0 is the whole point of this check.
    rng = np.random.default_rng([ap.seed, int(sigma * 1000), int(T * 100)])
    res = price_american_put_gbm(ap.S0, K, 0.0, sigma, T,
                                 n_steps=int(round(T * DPY)), n_paths=ap.n_paths,
                                 basis=BASIS, antithetic=COMMON.antithetic, rng=rng)
    euro = bs_put(ap.S0, K, 0.0, sigma, T)          # closed-form European
    baw = baw_american_put(ap.S0, K, 0.0, sigma, T)  # American analytic (r=0 guard)
    diff = res.price - euro
    app_rows.append({
        "r": 0.0, "sigma": sigma, "T": T,
        "lsmc_american": round(res.price, ap.round_decimals),
        "lsmc_stderr": round(res.stderr, ap.round_decimals),
        "bs_european": round(euro, ap.round_decimals),
        "baw_american": round(baw, ap.round_decimals),
        "amer_minus_euro": round(diff, ap.round_decimals),
        # How many standard errors the gap is from zero (|.| should be small).
        "diff_in_stderrs": round(diff / res.stderr, 2) if res.stderr > 0 else 0.0,
    })
app_table = pd.DataFrame(app_rows)
app_table.to_csv(APP / "2d_r0_american_equals_european.csv", index=False)
print("\nAppendix: American (LSMC) vs European (BS) vs BAW at r = 0:")
print(app_table.to_string(index=False))
