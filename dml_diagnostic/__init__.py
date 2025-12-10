"""
dml_diagnostic: Condition Number Diagnostics for Double Machine Learning
=========================================================================

This package provides tools for diagnosing the reliability of Double Machine
Learning (DML) estimators through the condition number κ_DML.

Quick Start
-----------
>>> from dml_diagnostic import DMLDiagnostic, load_lalonde
>>> 
>>> # Load data
>>> Y, D, X = load_lalonde(sample='experimental')
>>> 
>>> # Fit DML with diagnostics
>>> dml = DMLDiagnostic(learner='lasso')
>>> results = dml.fit(Y, D, X)
>>> print(results)

Theory
------
The DML condition number κ_DML = n / Σᵢ Ûᵢ² measures the curvature of the
orthogonal score. Large κ_DML indicates:
- Weak overlap (treatment highly predictable from covariates)
- Inflated variance and potential bias amplification
- Fragile inference similar to weak-IV problems

IMPORTANT: κ_DML is a continuous diagnostic. We do not impose specific
numerical thresholds. The interpretation depends on context, sample size,
and how κ_DML varies across specifications.

Reference
---------
Saco, G. (2025). "Finite-Sample Failures and Condition-Number Diagnostics
in Double Machine Learning." The Econometrics Journal.

Author: Gabriel Saco
License: MIT
GitHub: https://github.com/gsaco/dml-diagnostic
"""

from dml_diagnostic.estimator import DMLDiagnostic, DMLResult
from dml_diagnostic.diagnostics import (
    compute_kappa,
    kappa_interpretation,
    classify_regime,
    overlap_check,
)
from dml_diagnostic.plotting import (
    plot_kappa_summary,
    plot_overlap,
    plot_kappa_ci_cone,
    plot_kappa_comparison,
    plot_comparison,
    set_publication_style,
    save_figure,
    COLORS,
    FIGURE_SIZES,
)
from dml_diagnostic.reporting import (
    summary_table,
    to_latex,
)
from dml_diagnostic.data.lalonde import load_lalonde

__version__ = "1.0.0"
__author__ = "Gabriel Saco"

__all__ = [
    # Main estimator
    "DMLDiagnostic",
    "DMLResult",
    # Diagnostics
    "compute_kappa",
    "kappa_interpretation",
    "classify_regime", 
    "overlap_check",
    # Plotting
    "plot_kappa_summary",
    "plot_overlap",
    "plot_kappa_ci_cone",
    "plot_kappa_comparison",
    "plot_comparison",
    "set_publication_style",
    "save_figure",
    "COLORS",
    "FIGURE_SIZES",
    # Reporting
    "summary_table",
    "to_latex",
    # Data
    "load_lalonde",
]
