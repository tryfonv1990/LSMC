# %%
# Section 4 - Natural Gas Storage Valuation
#
# Valuing a natural-gas storage contract by least-squares Monte Carlo, following
# Boogert & de Jong (2008). The storage generalises the American option: instead
# of a single exercise, the holder chooses inject / hold / withdraw every day,
# subject to capacity and daily-rate limits, so the problem becomes a dynamic
# program over a (time x inventory) grid.
#
# Three studies:
#   4a. Day-ahead price simulation (mean-reverting log-price, paper eq. 22);
#   4b. Storage valuation for the standard contract (Exhibit 1);
#   4c. Convergence of the value as the path count grows, with an
#       Exhibit-3-style value distribution.
#
# All parameters live in configs/section4.yaml - edit that file, not this
# script. Run as interactive cells (# %%) or top-to-bottom. Figures go to
# ../results/figures/ and tables to ../results/tables/.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lsmc.storage import simulate_day_ahead, value_storage
from lsmc.config import load_config

# Load every tunable parameter from configs/section4.yaml (see lsmc.config).
CFG = load_config("section4")
PR = CFG.price               # day-ahead price-process block
ST = CFG.storage             # storage-contract + dynamic-program block
S0 = float(np.exp(PR.mu))    # start price exp(mu); the long-run mean is a bit lower

# Output locations, relative to the repo root (this file lives in experiments/).
FIG = Path(__file__).resolve().parents[1] / "results" / "figures"
TAB = Path(__file__).resolve().parents[1] / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# Shared keyword bundle for value_storage (the Exhibit-1 contract constraints).
CONTRACT = dict(v_min=ST.v_min, v_max=ST.v_max, v_start=ST.v_start, v_end=ST.v_end,
                max_inject=ST.max_inject, max_withdraw=ST.max_withdraw,
                alpha=ST.alpha, penalty=ST.penalty, basis=ST.basis)

print(f"S0 = {S0:.2f} EUR/MWh, capacity = {ST.v_max:,} MWh, basis = {ST.basis!r}")

# %%
# 4a. Day-ahead price simulation
#
# The spot price follows the one-factor Schwartz mean-reverting log-price process
# (paper eq. 22), simulated in daily steps over the one-year contract. With a
# constant mu (my choice, as the assignment allows), prices revert to a flat level
# (long-run mean exp(mu - sigma^2/(4 kappa)) ~ 19.2, slightly below exp(mu) ~ 20.1)
# rather than tracking a seasonal forward curve - so I capture the
# volatility that drives extrinsic storage value, but not the seasonal spread.
# I plot a few sample paths and the terminal-price distribution.

# %%
rng = np.random.default_rng(PR.seed)
paths = simulate_day_ahead(S0, PR.kappa, PR.sigma, PR.mu, PR.n_days, PR.n_paths,
                           antithetic=PR.antithetic, rng=rng)
print(f"simulated {paths.shape[0]:,} paths x {paths.shape[1]} days; "
      f"mean S_T = {paths[:, -1].mean():.2f}, "
      f"terminal std = {paths[:, -1].std():.2f}")

# Save summary statistics of the price distribution over time (not every path).
days = np.arange(paths.shape[1])
price_summary = pd.DataFrame({
    "day": days,
    "mean": paths.mean(axis=0).round(3),
    "p05": np.percentile(paths, 5, axis=0).round(3),
    "p95": np.percentile(paths, 95, axis=0).round(3),
})
price_summary.to_csv(TAB / "4a_price_paths_summary.csv", index=False)

# Figure: sample paths (left) and terminal-price histogram (right).
# The stationary log-price is N(mu - sigma^2/(2 kappa), sigma^2/(2 kappa)), so the
# long-run mean price is exp(mu - sigma^2/(4 kappa)), below the start level exp(mu).
LR_MEAN = float(np.exp(PR.mu - PR.sigma**2 / (4 * PR.kappa)))
fig, (axp, axh) = plt.subplots(1, 2, figsize=(11, 4.5))
for i in range(20):                                 # 20 illustrative paths
    axp.plot(days, paths[i], lw=0.7, alpha=0.6)
axp.plot(days, paths.mean(axis=0), color="tab:red", lw=1.5, label="cross-path mean")
axp.axhline(LR_MEAN, color="black", lw=1.0, ls="--",
            label=f"long-run mean {LR_MEAN:.1f}")
axp.axhline(S0, color="grey", lw=1.0, ls=":", label=f"start level exp(μ) = {S0:.1f}")
axp.set_xlabel("day")
axp.set_ylabel("spot price (EUR/MWh)")
axp.set_title("Sample day-ahead price paths")
axp.legend()
axh.hist(paths[:, -1], bins=50, color="tab:blue", alpha=0.8)
axh.axvline(LR_MEAN, color="black", lw=1.0, ls="--", label=f"long-run mean {LR_MEAN:.1f}")
axh.legend()
axh.set_xlabel("terminal spot price (EUR/MWh)")
axh.set_ylabel("frequency")
axh.set_title("Terminal-price distribution")
fig.suptitle("Section 4a: mean-reverting day-ahead gas price (eq. 22)", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "4a_price_simulation.png", dpi=130, bbox_inches="tight")
print(f"saved {FIG / '4a_price_simulation.png'}")

# %%
# 4b. Storage valuation for the standard contract
#
# I value the Exhibit-1 contract by backward induction over the time-inventory
# grid: at each day and inventory node the holder picks the inject/hold/withdraw
# action maximising immediate cash flow plus the regressed continuation value
# (b6 B-splines of spot). With a flat long-run level there is almost no seasonal
# (intrinsic) value, so what I price here is essentially the extrinsic value -
# the worth of reacting to spot volatility, which is exactly what the LSMC
# strategy monetises.

# %%
res = value_storage(paths, **CONTRACT)
value_rows = [{
    "contract": "standard (Exhibit 1)",
    "n_paths": res.n_paths,
    "storage_value_eur": round(res.value, ST.round_decimals),
    "stderr_eur": round(res.stderr, ST.round_decimals),
    "value_per_mwh_capacity": round(res.value / ST.v_max, 3),
}]
value_table = pd.DataFrame(value_rows)
value_table.to_csv(TAB / "4b_storage_value.csv", index=False)
print("\nStandard storage contract valuation:")
print(value_table.to_string(index=False))
print(f"=> {res.value:,.0f} EUR (+/- {res.stderr:,.0f})")

# %%
# 4c. Convergence vs number of paths + value distribution
#
# The paper reports (Exhibit 6) that the storage value is stable even
# at very few paths, thanks to strong mean reversion and the dimension reduction.
# I check this by valuing the contract over several path counts, each repeated
# n_runs times, and reporting the mean value and its across-run standard
# error (which should shrink with more paths). I also plot an Exhibit-3-style
# value distribution: like the paper's Exhibit 3, it is the realised value on
# each individual path from a SINGLE valuation (the 4b run on 10,000 paths),
# not the spread of the run averages.

# %%
conv_rows = []
for n_paths in ST.path_counts:
    run_values = []
    for run in range(ST.n_runs):
        # Independent paths per (path count, run); seed folds in both for repeatability.
        rng = np.random.default_rng([PR.seed, n_paths, run])
        p = simulate_day_ahead(S0, PR.kappa, PR.sigma, PR.mu, PR.n_days, n_paths,
                               antithetic=PR.antithetic, rng=rng)
        run_values.append(value_storage(p, **CONTRACT).value)
    run_values = np.asarray(run_values)
    conv_rows.append({
        "n_paths": n_paths,
        "mean_value_eur": round(run_values.mean(), ST.round_decimals),
        "std_eur": round(run_values.std(ddof=1), ST.round_decimals),
        "stderr_eur": round(run_values.std(ddof=1) / np.sqrt(ST.n_runs),
                            ST.round_decimals),
    })
conv_table = pd.DataFrame(conv_rows)
conv_table.to_csv(TAB / "4c_convergence.csv", index=False)
print("\nConvergence of storage value vs number of paths "
      f"({ST.n_runs} runs each):")
print(conv_table.to_string(index=False))

# Figure: convergence (left, value +/- std vs paths) and value distribution
# (right, Exhibit-3 style): the realised value on each path of the single 4b
# valuation. The storage value is the mean of this distribution.
fig, (axc, axd) = plt.subplots(1, 2, figsize=(11, 4.5))
axc.errorbar(conv_table["n_paths"], conv_table["mean_value_eur"],
             yerr=conv_table["std_eur"], marker="o", capsize=4)
axc.set_xscale("log")
axc.set_xlabel("number of paths (log scale)")
axc.set_ylabel("storage value (EUR)")
axc.set_title("Convergence of value vs paths")
dist_values = res.path_values                       # one value per path (4b run)
axd.hist(dist_values, bins=ST.hist_bins, color="tab:green", alpha=0.8,
         edgecolor="white", linewidth=0.5)           # white edges separate the bars
axd.axvline(dist_values.mean(), color="black", lw=1.2, ls="--",
            label=f"mean {dist_values.mean():,.0f}")
axd.set_xlabel("realised value per path (EUR)")
axd.set_ylabel("number of paths")
axd.set_title(f"Value distribution (single run, {res.n_paths:,} paths)")
axd.legend()
fig.suptitle("Section 4c: storage-value convergence and distribution", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "4c_convergence.png", dpi=130, bbox_inches="tight")
print(f"saved {FIG / '4c_convergence.png'}")
