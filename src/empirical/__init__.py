"""
Empirical application module for DML condition number diagnostics.

This module provides:
- data_lalonde: Functions to load and clean the LaLonde/NSW job training data
- dml_kappa: DML estimator with κ_DML condition number diagnostic
- utils_tables: Utility functions for tables and plots
"""

from .data_lalonde import (
    load_lalonde_data,
    get_experimental_sample,
    get_observational_sample,
    get_covariate_matrix,
    get_covariate_names,
    summary_statistics
)

from .dml_kappa import (
    PLRDoubleMLKappa,
    get_learner,
    estimate_propensity_score,
    compute_overlap_diagnostics,
    run_dml_analysis
)

from .utils_tables import (
    results_to_dataframe,
    combine_results_tables,
    format_results_table,
    results_to_latex,
    plot_propensity_histogram,
    plot_overlap_comparison,
    plot_kappa_by_design,
    plot_ci_length_vs_kappa,
    plot_theta_estimates
)

__all__ = [
    # Data loading
    'load_lalonde_data',
    'get_experimental_sample',
    'get_observational_sample',
    'get_covariate_matrix',
    'get_covariate_names',
    'summary_statistics',
    # DML estimation
    'PLRDoubleMLKappa',
    'get_learner',
    'estimate_propensity_score',
    'compute_overlap_diagnostics',
    'run_dml_analysis',
    # Utilities
    'results_to_dataframe',
    'combine_results_tables',
    'format_results_table',
    'results_to_latex',
    'plot_propensity_histogram',
    'plot_overlap_comparison',
    'plot_kappa_by_design',
    'plot_ci_length_vs_kappa',
    'plot_theta_estimates'
]
