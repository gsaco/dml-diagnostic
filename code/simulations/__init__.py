"""
Simulation Module for DML Finite-Sample Conditioning Study
===========================================================

This module provides reusable functions for Monte Carlo simulations
studying the condition number κ_DML in Double Machine Learning.

Main Components
---------------
Data Generation:
    - generate_plr_data : Generate data from the PLR model
    - get_beta_D, get_gamma : Coefficient vectors
    - get_sigma_U_sq : Calibrated residual variance for overlap levels

DML Estimation:
    - dml_plr_estimator : Cross-fitted DML estimator with κ_DML computation

Monte Carlo:
    - run_simulation_grid : Run MC simulations across DGP configurations
    - compute_summary_statistics : Aggregate results by DGP
    - format_summary_for_latex : Generate LaTeX tables

Configuration:
    - DGP_CONFIGS : Pre-defined DGP configurations (9 designs)
    - OVERLAP_CONFIG : Overlap level definitions

Theory
------
The paper's main expansion shows:
    θ̂ − θ₀ = κ_DML · (Sₙ + Bₙ) + Rₙ

where κ_DML = n / Σᵢ Ûᵢ² measures ill-conditioning. Large κ_DML
amplifies both variance and bias, leading to poor finite-sample coverage.

References
----------
- Robinson, P.M. (1988). Root-N-consistent semiparametric regression.
  Econometrica, 56(4), 931–954.
- Chernozhukov, V. et al. (2018). Double/debiased machine learning for
  treatment and structural parameters. Econometrics Journal, 21(1), C1–C68.
- Chernozhukov, V., Newey, W.K., & Singh, R. (2023). A simple and general
  debiased machine learning theorem. Biometrika, 110(1), 257–264.
- Bach, P. et al. (2022). DoubleML: An object-oriented implementation.
  Journal of Machine Learning Research, 23(53), 1–6.
"""

from .core import (
    # DGP functions
    generate_plr_data,
    get_beta_D,
    get_gamma,
    get_sigma_U_sq,
    compute_theoretical_r2,
    calibrate_sigma_U_for_r2,
    # DML estimator
    dml_plr_estimator,
    # Monte Carlo
    run_single_replication,
    run_simulation_grid,
    compute_summary_statistics,
    format_summary_for_latex,
    # Configs
    DGP_CONFIGS,
    OVERLAP_CONFIG,
)

__version__ = "1.0.0"

__all__ = [
    # DGP
    "generate_plr_data",
    "get_beta_D",
    "get_gamma",
    "get_sigma_U_sq",
    "compute_theoretical_r2",
    "calibrate_sigma_U_for_r2",
    # Estimation
    "dml_plr_estimator",
    # Monte Carlo
    "run_single_replication",
    "run_simulation_grid",
    "compute_summary_statistics",
    "format_summary_for_latex",
    # Configuration
    "DGP_CONFIGS",
    "OVERLAP_CONFIG",
]
