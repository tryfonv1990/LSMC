# Experiments

One script per assignment section. Each is a plain `.py` file using the
`# %%` cell format, so it runs three ways:

* **Interactively** in PyCharm ("Run Cell") or VS Code — inline plots, like a
  notebook but diff-friendly and refactorable.
* **Top-to-bottom as a script**: `python experiments/01_american_lsmc.py`.
* Imported piecemeal for debugging.

Scripts import the reusable engine from `lsmc` (installed via `pip install -e .`),
run the section's experiments, save figures to `../results/figures/` and tables
to `../results/tables/`, and produce the numbers/plots referenced by
`../report/report.md`.

| Script | Section | Contents |
|--------|---------|----------|
| `01_american_lsmc.py`    | 1 | LSMC American put; Table 1 reproduction; discretisation-step (1 day → 1 week) study; Heston pricing and κ/θ/ξ/ρ sensitivities. |
| `02_control_variates.py` | 2 | European-put control variate; correlation & variance-reduction; Barone-Adesi-Whaley; error with/without CV for σ∈{0.05,0.1,0.2,0.3}; gamma profile and estimator variance. |
| `03_sde_convergence.py`  | 3 | Bias-variance decomposition; Milstein vs Euler-Maruyama; error vs #paths (10k→100k) for GBM and Heston. |
| `04_gas_storage.py`      | 4 | Day-ahead price simulation; storage continuation value via b6 B-spline OLS; volume/rate constraints; convergence vs #paths; Exhibit-3-style plot. |
