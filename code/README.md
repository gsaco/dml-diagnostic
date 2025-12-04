# Code Directory

This directory contains the simulation code for studying the DML condition number $\kappa_{\mathrm{DML}}$.

## Files

### Main Notebook

**`simulation_kappa_dml.ipynb`** — Main simulation notebook that reproduces all paper results.

Sections:
1. **Introduction** — Theoretical background and motivation
2. **Data-Generating Process** — PLR model specification and overlap calibration
3. **DML Estimator** — Cross-fitted estimator implementation
4. **Monte Carlo Simulation** — Run 9 DGPs × 500 replications
5. **Results** — Summary tables and figures

To run:
```bash
jupyter notebook simulation_kappa_dml.ipynb
```

By default, the notebook loads pre-computed results from `../results/simulation_results_full.csv`. Set `RUN_NEW = True` to re-run simulations.

### Simulation Module

**`simulations/`** — Reusable Python package with core functions.

```
simulations/
├── __init__.py     # Package exports
└── core.py         # All simulation functions
```

Key functions:
- `generate_plr_data()` — Generate PLR data with calibrated overlap
- `dml_plr_estimator()` — Cross-fitted DML with $\kappa_{\mathrm{DML}}$ computation
- `run_simulation_grid()` — Run Monte Carlo across DGP configurations
- `compute_summary_statistics()` — Aggregate replication results
- `format_summary_for_latex()` — Generate publication-ready tables

Example usage:
```python
from simulations import generate_plr_data, dml_plr_estimator, DGP_CONFIGS

# Generate one dataset
Y, D, X, info = generate_plr_data(n=1000, rho=0.5, overlap='moderate')

# Fit DML
result = dml_plr_estimator(Y, D, X, K=5, learner='rf')
print(f"θ̂ = {result['theta_hat']:.3f}, κ_DML = {result['kappa_dml']:.2f}")
```

## Dependencies

See `../requirements.txt`. Key packages:
- `numpy`, `pandas` — Data manipulation
- `scikit-learn` — Random Forest, cross-validation
- `matplotlib` — Visualization
