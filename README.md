# Least Squares Monte Carlo in Natural Gas Storage Valuation

A small Monte Carlo engine for pricing options with early exercise, built on the
Longstaff-Schwartz (2001) least-squares Monte Carlo (LSMC) method and extended to
natural gas storage following Boogert & de Jong (2008). This repository is my
answer to an interview assignment (the assignment brief and reference papers are
not included in this repository).

The written answers are in [`report/report.md`](report/report.md).

## What it covers

1. **American options and LSMC:** LSMC American put under geometric Brownian
   motion (Euler-Maruyama); reproduction of Longstaff-Schwartz Table 1; effect of
   the time step (1 day to 1 week) at several volatilities; the same put under
   the Heston stochastic-volatility model, with the effect of κ, θ, ξ and ρ.
2. **Control variates:** the European put (Black-Scholes) as a control variate;
   correlation and variance reduction; the Barone-Adesi-Whaley (1987) analytic
   approximation; pricing error with and without the control variate; gamma
   profile and the variance of the gamma estimate.
3. **SDE convergence:** bias-variance decomposition of the error; Euler vs
   Milstein for GBM and Heston, with errors for 10,000 to 100,000 paths.
4. **Natural gas storage:** day-ahead price simulation; storage continuation
   value by OLS regression on the b6 B-spline basis, with injection and
   withdrawal limits; convergence in the number of paths; an Exhibit-3-style plot.

## Layout

```
src/lsmc/      the engine: processes, basis, regression, pricers, storage, config
configs/       one YAML settings file per section
experiments/   one script per section (# %% cells): runs the experiments, makes plots
tests/         pytest suite (checks against the papers' values)
results/       figures and tables written by the scripts
report/        report.md (written answers)
```

## Getting started

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash; use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
pip install -e .                   # install the lsmc package (editable)

pytest                             # run the tests (55, about 1 minute)
python experiments/01_american_lsmc.py   # run one section end-to-end
```

The scripts in `experiments/` use the `# %%` cell format, so they run cell by
cell in PyCharm or VS Code, or top to bottom as normal scripts. Each one writes
its figures to `results/figures/` and its tables to `results/tables/`, named by
sub-task (e.g. `2b_cv_baw_error.png`). `pyproject.toml` puts `src/` on the path,
so scripts and tests can `import lsmc` directly.

Approximate run times: Section 1 about 3 min, Section 2 about 2 min,
Section 3 about 2 min, Section 4 about 5 min. The results/ folder already holds
the outputs, so the report can be read without running anything. The seeds are
fixed and `requirements.txt` pins the library versions used (on Python 3.14), so
a rerun reproduces every table and figure exactly.

## Settings

All settings are in `configs/section1.yaml` to `configs/section4.yaml`; change
them there, not in the scripts. Each file has a `common` block (strike, rate,
antithetic flag, regression basis) and one block per experiment. For example, to
change the regression basis for Section 1:

```yaml
common:
  basis: laguerre   # laguerre | powers | powers_deg2 | bspline_b6
```

For Heston, `heston.two_factor` sets what the regression uses: `false` uses spot
only; `true` uses spot and variance with the basis `heston.basis_2f`
(`poly2d_sv` = `{1, S, S², v, v², S·v}`).

`lsmc.config.load_config("section1")` loads a file and checks it: a misspelt
basis or scheme name gives an error, and a Heston base case that breaks the
Feller condition gives a warning.

## Report

`report/report.md` holds the written answers, with links to the figures and
tables in `results/`. Sections are collapsible and best viewed on GitHub or in
the VS Code / PyCharm Markdown preview. To export to PDF:

```bash
pandoc report/report.md -o report/report.pdf
```

## References

Full details are in section 5 of the report.

- Longstaff & Schwartz (2001), *Valuing American Options by Simulation: A Simple
  Least-Squares Approach*, Review of Financial Studies.
- Barone-Adesi & Whaley (1987), *Efficient Analytic Approximation of American
  Option Values*, Journal of Finance.
- Milstein (1975), *Approximate Integration of Stochastic Differential
  Equations*, Theory of Probability and Its Applications.
- Boogert & de Jong (2008), *Gas Storage Valuation Using a Monte Carlo Method*,
  Journal of Derivatives.

Also cited: Heston (1993) for the stochastic-volatility model and Schwartz (1997)
for the one-factor gas price model.

## Highlevel Results


- **Section 1:** the LS 8-path example gives 0.1144 exactly; all 16 Table 1
  points match the paper within Monte Carlo error (max difference 0.047, about
  1.6 combined standard errors).
- **Section 2:** the European-put control variate cuts variance 4-6x at r = 0.
- **Section 3:** Milstein roughly halves Euler's bias for GBM; under Heston it
  helps only when the vol of variance ξ is large.
- **Section 4:** storage value about EUR 4.65m on 10,000 paths (paper: EUR 5.4m
  with a seasonal forward curve; I use a constant long-run level).
