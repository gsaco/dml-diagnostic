"""
DML Condition Number Analysis Package
=====================================

This package implements Monte Carlo simulations for studying the condition
number κ_DML in Double Machine Learning, aligned with the theoretical
framework on conditioning regimes.

Main Components
---------------
Data Generation:
    - generate_plr_data : Generate PLR data with R²(D|X)-calibrated overlap
    - g0_function : Nonlinear nuisance function
    - make_toeplitz_cov : Toeplitz covariance construction
    - calibrate_sigma_xi_sq : Overlap calibration via R²(D|X)

DML Estimation:
    - run_dml_plr : Cross-fitted DML estimator with κ_DML computation
    - get_nuisance_model : Factory for LIN/LAS/RF learners
    - DMLResult : Structured estimation results

Monte Carlo:
    - run_simulation : Full MC simulation across design grid
    - run_single_replication : Single MC replication
    - run_full_study : Complete study with tables and figures

Summary & Visualization:
    - compute_cell_summary : Cell-level aggregation
    - make_table1, make_table2 : Paper tables
    - plot_coverage_vs_kappa : Figure 1 (killer plot)
    - plot_ci_length_vs_kappa : Figure 2

Configuration:
    - R2_TARGETS : R²(D|X) levels for overlap calibration
    - THETA0 : True treatment effect
    - B_DEFAULT : Default MC replications

Theory
------
The paper's main linearization shows:
    θ̂ − θ₀ = κ_DML · (Sₙ + Bₙ) + Rₙ

where κ_DML = n / Σᵢ Ûᵢ² = 1/|Ĵ_θ| measures ill-conditioning.

Three regimes:
    (i)   Well-conditioned: κₙ = O_P(1)
    (ii)  Moderately ill-conditioned: κₙ = O_P(n^β), 0 < β < 1/2
    (iii) Severely ill-conditioned: κₙ ≍ c√n

References
----------
- Robinson, P.M. (1988). Econometrica, 56(4), 931–954.
- Chernozhukov et al. (2018). Econometrics Journal, 21(1), C1–C68.
- Chernozhukov, Newey & Singh (2023). Biometrika, 110(1), 257–264.
- Bach et al. (2022). JMLR, 23(53), 1–6.
"""

from .core import (
    # Constants
    THETA0,
    R2_TARGETS,
    DEFAULT_SEED,
    B_DEFAULT,
    # DGP functions
    make_toeplitz_cov,
    get_gamma_coeffs,
    compute_V_gamma,
    calibrate_sigma_xi_sq,
    g0_function,
    generate_plr_data,
    DGPInfo,
    # DML estimation
    get_nuisance_model,
    run_dml_plr,
    DMLResult,
    # Simulation
    run_single_replication,
    run_simulation,
    ReplicationResult,
    # Summary and tables
    compute_cell_summary,
    compute_regime_summary,
    assign_kappa_regime,
    make_table1,
    make_table2,
    table_to_latex,
    # Visualization
    plot_coverage_vs_kappa,
    plot_ci_length_vs_kappa,
    # Main entry point
    run_full_study,
)

__version__ = "2.0.0"

__all__ = [
    # Constants
    "THETA0",
    "R2_TARGETS",
    "DEFAULT_SEED",
    "B_DEFAULT",
    # DGP functions
    "make_toeplitz_cov",
    "get_gamma_coeffs",
    "compute_V_gamma",
    "calibrate_sigma_xi_sq",
    "g0_function",
    "generate_plr_data",
    "DGPInfo",
    # DML estimation
    "get_nuisance_model",
    "run_dml_plr",
    "DMLResult",
    # Simulation
    "run_single_replication",
    "run_simulation",
    "ReplicationResult",
    # Summary and tables
    "compute_cell_summary",
    "compute_regime_summary",
    "assign_kappa_regime",
    "make_table1",
    "make_table2",
    "table_to_latex",
    # Visualization
    "plot_coverage_vs_kappa",
    "plot_ci_length_vs_kappa",
    # Main entry point
    "run_full_study",
]
