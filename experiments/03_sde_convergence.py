# %%
# Section 3 — Convergence of the Stochastic Differential Equation
#
# The total error of an LSMC price has two independent parts:
#
#   * variance — Monte Carlo sampling noise; shrinks like 1/sqrt(n_paths);
#   * bias     — time-discretisation error from solving the SDE on a finite
#                    grid; independent of the path count, set by the step size
#                    and the scheme.
#
# Adding paths (Section 2's control variate also targeted variance) drives the
# noise down but leaves bias. This section attacks that bias by
# replacing Euler-Maruyama with the higher-order Milstein scheme, for both
# GBM and Heston, and shows the error against the Barone-Adesi-Whaley (1987)
# analytic value ("subtract LSMC Euler from the BAW value", per the assignment).
#
# Three studies:
#   3a. bias-variance decomposition on GBM (fixed grid, vary paths);
#   3b. GBM Euler vs Milstein error vs BAW across path counts;
#   3c. Heston Euler vs Milstein vs a fine-grid reference (no closed form).
#
# All parameters live in configs/section3.yaml -- edit that file, not this
# script. Run as interactive cells (# %%) or top-to-bottom. Figures go to
# ../results/figures/ and tables to ../results/tables/.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsmc.american import price_american_put_gbm, price_american_put_on_paths
from lsmc.analytic import baw_american_put
from lsmc.processes import (correlated_heston_increments, draw_normals,
                            simulate_gbm, simulate_gbm_exact, simulate_heston)
from lsmc.config import load_config

# Load every editable parameter from configs/section3.yaml (see lsmc.config).
CFG = load_config("section3")
COMMON = CFG.common          # shared block: strike, rate, basis, antithetic
K, R = COMMON.strike, COMMON.rate
BASIS = COMMON.basis

# Output locations, relative to the repo root (this file lives in experiments/).
FIG = Path(__file__).resolve().parents[1] / "results" / "figures"
TAB = Path(__file__).resolve().parents[1] / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

print(f"basis = {BASIS!r}, strike = {K}, rate = {R}")

# %%
# 3a. Bias-variance decomposition (GBM)
#
# I hold the time grid FIXED (coarse) and vary only the number of paths. At each
# path count I run several independent repeats and split the error into:
#
#   * variance — the spread (standard deviation) of the repeated estimates;
#   * bias     — the mean estimate minus the bias-free reference.
#
# The reference is the exact log-Euler GBM price (no
# # discretisation error) on the same coarse grid
# estimated with many paths. Expectation: variance falls
# like 1/sqrt(n_paths) while the Euler bias is flat -- the estimator converges to
# a biased limit, not the true value, as paths -> infinity.

# %%
bv = CFG.bias_variance                 # config bias_variance block
# Bias-free reference: exact-scheme GBM on the same coarse grid (same exercise
# dates, same LSMC regression), so what separates Euler from it is the
# discretisation bias. One run with 200k paths and its own seed: its Monte Carlo
# noise is small but not zero, and shifts every row's bias equally.
ref_rng = np.random.default_rng([bv.seed, 999])
ref = price_american_put_gbm(bv.S0, K, R, bv.sigma, bv.T, n_steps=bv.n_steps,
                             n_paths=200000, basis=BASIS, exact=True,
                             antithetic=COMMON.antithetic, rng=ref_rng).price

bv_rows = []
for n_paths in bv.path_counts:
    # n_repeats independent Euler estimates at this path count.
    ests = []
    for rep in range(bv.n_repeats):
        rng = np.random.default_rng([bv.seed, n_paths, rep])
        est = price_american_put_gbm(bv.S0, K, R, bv.sigma, bv.T,
                                     n_steps=bv.n_steps, n_paths=n_paths,
                                     basis=BASIS, scheme="euler",
                                     antithetic=COMMON.antithetic, rng=rng).price
        ests.append(est)
    ests = np.asarray(ests)
    variance_sd = ests.std(ddof=1)         # spread of estimates = MC noise
    bias = ests.mean() - ref               # systematic offset vs bias-free ref
    bv_rows.append({
        "n_paths": n_paths,
        "mean_price": round(ests.mean(), bv.round_decimals),
        "bias": round(bias, bv.round_decimals),
        "std_dev": round(variance_sd, bv.round_decimals),      # ~ variance^0.5
        "rmse": round(np.sqrt(bias**2 + variance_sd**2), bv.round_decimals),
    })
bv_table = pd.DataFrame(bv_rows)
bv_table.to_csv(TAB / "3a_bias_variance_decomposition.csv", index=False)
print(f"\nBias-variance decomposition (Euler, fixed {bv.n_steps}-step grid; "
      f"exact-scheme reference = {ref:.4f}):")
print(bv_table.to_string(index=False))

# Plot |bias| (flat floor) and standard deviation (falls with paths) together.
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(bv_table["n_paths"], bv_table["bias"].abs(), marker="o",
        label="|bias| (discretisation)")
ax.plot(bv_table["n_paths"], bv_table["std_dev"], marker="s",
        label="std dev (Monte Carlo noise)")
ax.plot(bv_table["n_paths"], bv_table["rmse"], marker="^", ls="--",
        label="RMSE (total error)")
ax.set_xlabel("number of paths")
ax.set_ylabel("error component")
ax.set_title("Section 3a: bias-variance decomposition (Euler GBM, fixed grid)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "3a_bias_variance_decomposition.png", dpi=130)
print(f"saved {FIG / '3a_bias_variance_decomposition.png'}")

# %%
# 3b. GBM: Euler vs Milstein error, vs BAW and vs the exact scheme
#
# Same coarse grid, both schemes, across the 10k -> 100k path range. I report the
# error against TWO references:
#
#   * BAW — the analytic approximation the assignment names ("subtract LSMC
#     Euler from the BAW value"). Caveat: BAW is itself only an approximation,
#     so it does not sit exactly on the true price — it carries its own error.
#   * exact GBM — the bias-free log-Euler price on the same Brownian paths.
#     This isolates pure discretisation bias: any gap to the exact scheme is
#     the scheme's own error, with no BAW-approximation error mixed in.
#
# The exact-scheme column is the fair measure, and it shows the expected result:
# the Milstein correction 0.5 sigma^2 S (dW^2 - dt) raises the strong order from
# 0.5 to 1.0, roughly halving the bias vs Euler. (Against BAW the picture is
# not clear precisely because BAW's ~0.02 approximation error is comparable to the
# scheme bias at this grid — a good illustration of why the choice of reference
# matters.)
#
# Crucially I use common random numbers: at each path count all three
# schemes (exact, Euler, Milstein) are priced on the SAME Brownian increments,
# so the exact-scheme comparison is noise-free and the tiny bias difference
# between Euler and Milstein is not swamped by Monte Carlo scatter.

# %%
gc = CFG.gbm_convergence               # config gbm_convergence block
baw_gbm = baw_american_put(gc.S0, K, R, gc.sigma, gc.T)   # analytic approximation
dt_gbm = gc.T / gc.n_steps
gbm_rows = []
for n_paths in gc.path_counts:
    # One shared Brownian path for this path count; every scheme reuses it (CRN),
    # so exact/Euler/Milstein differ only by the stepping rule, not the noise.
    rng = np.random.default_rng([gc.seed, n_paths])
    dW = draw_normals(n_paths, gc.n_steps, rng, COMMON.antithetic) * np.sqrt(dt_gbm)
    # Bias-free reference on these very paths (exact log-Euler GBM).
    exact_paths = simulate_gbm_exact(gc.S0, R, gc.sigma, gc.T, gc.n_steps, n_paths,
                                     dW=dW)
    exact_price = price_american_put_on_paths(exact_paths, K, R, gc.T,
                                              basis=BASIS).price
    for scheme in gc.schemes:
        paths = simulate_gbm(gc.S0, R, gc.sigma, gc.T, gc.n_steps, n_paths,
                             scheme=scheme, dW=dW)      # same dW -> CRN
        res = price_american_put_on_paths(paths, K, R, gc.T, basis=BASIS,
                                          return_pv=True)
        price, pv = res[0].price, res[1]
        stderr = pv.std(ddof=1) / np.sqrt(n_paths)
        gbm_rows.append({
            "scheme": scheme, "n_paths": n_paths,
            "price": round(price, gc.round_decimals),
            "stderr": round(stderr, gc.round_decimals),
            "err_vs_baw": round(price - baw_gbm, gc.round_decimals),
            "err_vs_exact": round(price - exact_price, gc.round_decimals),
        })
gbm_table = pd.DataFrame(gbm_rows)
gbm_table.to_csv(TAB / "3b_gbm_scheme_convergence.csv", index=False)
print(f"\nGBM Euler vs Milstein (coarse {gc.n_steps}-step grid); "
      f"BAW = {baw_gbm:.4f} (references shared per path count via CRN):")
print(gbm_table.to_string(index=False))

# Two panels: error vs BAW (as asked) and error vs the exact scheme (honest bias).
fig, (axb, axe) = plt.subplots(1, 2, figsize=(11, 4.5))
for scheme in gc.schemes:
    sub = gbm_table[gbm_table["scheme"] == scheme]
    axb.plot(sub["n_paths"], sub["err_vs_baw"], marker="o", label=scheme)
    axe.plot(sub["n_paths"], sub["err_vs_exact"], marker="o", label=scheme)
axb.axhline(0.0, color="grey", lw=0.8, ls="--")
axb.set_xlabel("number of paths"); axb.set_ylabel("LSMC − BAW")
axb.set_title("vs BAW (approximation, assignment metric)")
axb.legend()
axe.axhline(0.0, color="grey", lw=0.8, ls="--", label="exact reference")
axe.set_xlabel("number of paths"); axe.set_ylabel("LSMC − exact")
axe.set_title("vs exact GBM (pure discretisation bias)")
axe.legend()
fig.suptitle(f"Section 3b: GBM Euler vs Milstein error (coarse {gc.n_steps}-step grid)",
             y=1.02)
fig.tight_layout()
fig.savefig(FIG / "3b_gbm_scheme_convergence.png", dpi=130, bbox_inches="tight")
print(f"saved {FIG / '3b_gbm_scheme_convergence.png'}")

# %%
# 3c. Heston: Euler vs Milstein vs a fine-grid reference
#
# The American put under Heston has no closed form, so the reference is a
# high-resolution simulation (fine_steps), where both schemes essentially
# agree. I price on a coarse grid with each scheme and measure the bias against
# that reference. The Milstein term on the variance leg
# (0.25 xi^2 (dW^2 - dt)) is where any gain comes from — but note my simulator
# uses exact log-Euler on S, so only the variance part is discretised. That
# correction only hits when the variance process is rough, i.e. high
# vol-of-variance xi. I therefore run TWO regimes (config heston.regimes):
#
#   * assignment — the brief's mild xi = 0.10: Euler and Milstein nearly
#     coincide (the variance leg is already smooth), so Milstein adds little;
#   * stress — large xi = 0.60: the variance leg is rough and Milstein
#     visibly reduces the bias.
#
# This makes the fair point: Milstein is not free everywhere — it pays off
# precisely where the discretised part is harder to integrate.
#
# Clean reference (same design as 3b and the Section-1b nested grid). A
# reference built from independent paths would put its own Monte Carlo noise
# (plus the coarse run's) into the gap, and one that may exercise on every fine
# step would also carry the extra value of 399 vs 9 exercise dates. So, at each
# path count:
#
#   1. draw ONE set of correlated fine-grid increments (dW^v, dW^S);
#   2. the coarse 10-step increments are the sums of each block of
#      fine_steps / coarse_steps fine increments -- the same Brownian motion;
#   3. the fine reference is simulated on the fine grid but may only be
#      exercised on the 10 coarse dates (I keep every block-th column).
#
# Both runs then share the noise, the path count and the exercise dates; only the
# step size differs, so bias_vs_fine is the discretisation bias of the coarse
# scheme. For transparency I also report ref_all_dates (400 exercise dates), the fine reference
# exercisable on every fine step; ref_all_dates − fine_reference is the
# Bermudan "more exercise dates" premium that the old design mixed into the bias.

# %%
hc = CFG.heston_convergence            # config heston_convergence block
if hc.fine_steps % hc.coarse_steps:
    raise ValueError("heston_convergence.fine_steps must be a multiple of "
                     "coarse_steps (nested grids)")
BLOCK = hc.fine_steps // hc.coarse_steps    # fine steps per coarse step


def heston_paths(rp, scheme, n_steps, dW):
    """Heston (S, v) paths for regime rp driven by increments dW."""
    return simulate_heston(hc.S0, hc.r, rp["v0"], rp["kappa"], rp["theta"],
                           rp["xi"], hc.rho, hc.T, n_steps, dW[0].shape[0],
                           scheme=scheme, dW=dW)


def lsmc_price(S):
    """Spot-only LSMC American put; every column after 0 is an exercise date."""
    return price_american_put_on_paths(S, hc.K, hc.r, hc.T, basis=BASIS).price


def coarsen(x):
    """Sum each block of BLOCK fine increments -> coarse-grid increments."""
    return x.reshape(x.shape[0], hc.coarse_steps, BLOCK).sum(axis=2)


hes_rows = []
for ri, rp in enumerate(hc.regimes):        # ri: deterministic regime index for seeding
    # Sweep over path counts (assignment: 10k -> 100k); coarse and fine share paths.
    for n_paths in hc.path_counts:
        rng = np.random.default_rng([hc.seed, ri, n_paths])
        dW_fine = correlated_heston_increments(n_paths, hc.fine_steps,
                                               hc.T / hc.fine_steps, hc.rho, rng,
                                               COMMON.antithetic)
        dW_coarse = tuple(coarsen(x) for x in dW_fine)

        # Fine reference: average the two schemes at high resolution (they agree
        # there). Exercise restricted to the coarse dates (every BLOCK-th column).
        fine_S = [heston_paths(rp, s, hc.fine_steps, dW_fine)[0] for s in hc.schemes]
        ref = float(np.mean([lsmc_price(S[:, ::BLOCK]) for S in fine_S]))
        ref_all = float(np.mean([lsmc_price(S) for S in fine_S]))
        del fine_S

        for scheme in hc.schemes:
            price = lsmc_price(heston_paths(rp, scheme, hc.coarse_steps, dW_coarse)[0])
            hes_rows.append({
                "regime": rp["name"], "xi": rp["xi"], "scheme": scheme,
                "n_paths": n_paths, "coarse_steps": hc.coarse_steps,
                "coarse_price": round(price, hc.round_decimals),
                "fine_reference": round(ref, hc.round_decimals),
                "bias_vs_fine": round(price - ref, hc.round_decimals),
                "ref_all_dates": round(ref_all, hc.round_decimals),
            })
hes_table = pd.DataFrame(hes_rows)
hes_table.to_csv(TAB / "3c_heston_scheme_convergence.csv", index=False)
print(f"\nHeston Euler vs Milstein vs fine-grid reference "
      f"(coarse {hc.coarse_steps} vs fine {hc.fine_steps} steps):")
print(hes_table.to_string(index=False))

# One panel per regime: |bias vs fine| against path count, Euler vs Milstein.
fig, axes = plt.subplots(1, len(hc.regimes), figsize=(11, 4.5), sharex=True)
for ax, rp in zip(np.atleast_1d(axes), hc.regimes):
    for scheme in hc.schemes:
        sub = hes_table[(hes_table["regime"] == rp["name"]) &
                        (hes_table["scheme"] == scheme)]
        ax.plot(sub["n_paths"], sub["bias_vs_fine"].abs(), marker="o", label=scheme)
    ax.set_xlabel("number of paths")
    ax.set_ylabel("|coarse − fine| bias")
    ax.set_title(f"{rp['name']} (ξ={rp['xi']:g})")
    ax.legend()
fig.suptitle(f"Section 3c: Heston discretisation bias vs paths "
             f"({hc.coarse_steps}-step grid vs {hc.fine_steps}-step reference)", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "3c_heston_scheme_convergence.png", dpi=130, bbox_inches="tight")
print(f"saved {FIG / '3c_heston_scheme_convergence.png'}")
