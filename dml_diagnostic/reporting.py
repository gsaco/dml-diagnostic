"""
DML Diagnostic Reporting Functions
===================================

Functions for creating summary tables and LaTeX output.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np

# Try importing pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _check_pandas():
    """Check if pandas is available."""
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for reporting. "
            "Install it with: pip install pandas"
        )


def summary_table(
    results: List["DMLResult"],
    include_kappa: bool = True,
    include_r_squared: bool = True,
) -> "pd.DataFrame":
    """
    Create a summary table comparing multiple DML results.
    
    Parameters
    ----------
    results : list of DMLResult
        List of results from different learners/specifications.
    include_kappa : bool, default True
        Whether to include the condition number column.
    include_r_squared : bool, default True
        Whether to include R²(D|X) column.
        
    Returns
    -------
    pd.DataFrame
        Summary table with one row per result.
        
    Examples
    --------
    >>> from dml_diagnostic import DMLDiagnostic, load_lalonde, summary_table
    >>> Y, D, X = load_lalonde('experimental')
    >>> results = [DMLDiagnostic(learner=l).fit(Y, D, X) for l in ['lasso', 'rf']]
    >>> print(summary_table(results))
    """
    _check_pandas()
    
    rows = []
    for r in results:
        row = {
            'Learner': r.learner,
            'θ̂': r.theta,
            'SE': r.se,
            '95% CI': f'[{r.ci_lower:.3f}, {r.ci_upper:.3f}]',
            'CI Length': r.ci_length,
        }
        
        if include_kappa:
            row['κ_DML'] = r.kappa
        
        if include_r_squared:
            row['R²(D|X)'] = r.r_squared_d
        
        row['n'] = r.n
        rows.append(row)
    
    return pd.DataFrame(rows)


def results_to_dict(
    results: List["DMLResult"],
) -> List[Dict[str, Any]]:
    """
    Convert a list of DMLResult objects to a list of dictionaries.
    
    Parameters
    ----------
    results : list of DMLResult
        List of results.
        
    Returns
    -------
    list of dict
        Each result as a dictionary.
    """
    return [r.to_dict() for r in results]


def to_latex(
    df: "pd.DataFrame",
    caption: Optional[str] = None,
    label: Optional[str] = None,
    float_format: str = "%.3f",
) -> str:
    """
    Export a DataFrame to LaTeX table format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Table to export.
    caption : str, optional
        Table caption.
    label : str, optional
        LaTeX label for referencing.
    float_format : str, default '%.3f'
        Format string for floating point numbers.
        
    Returns
    -------
    str
        LaTeX table code.
    """
    _check_pandas()
    
    latex = df.to_latex(
        index=False,
        float_format=float_format,
        escape=False,
    )
    
    # Add caption and label if provided
    if caption or label:
        lines = latex.split('\n')
        
        # Find the end of tabular
        for i, line in enumerate(lines):
            if '\\end{tabular}' in line:
                insert_point = i + 1
                break
        
        # Insert caption and label
        additions = []
        if caption:
            additions.append(f'\\caption{{{caption}}}')
        if label:
            additions.append(f'\\label{{{label}}}')
        
        for j, add in enumerate(additions):
            lines.insert(insert_point + j, add)
        
        latex = '\n'.join(lines)
    
    return latex


def format_estimate(
    theta: float,
    se: float,
    kappa: float,
    digits: int = 3,
) -> str:
    """
    Format a DML estimate for text reporting.
    
    Parameters
    ----------
    theta : float
        Point estimate.
    se : float
        Standard error.
    kappa : float
        Condition number.
    digits : int, default 3
        Number of decimal places.
        
    Returns
    -------
    str
        Formatted string like "θ̂ = 1.234 (0.045) [κ = 2.34]"
    """
    return f"θ̂ = {theta:.{digits}f} ({se:.{digits}f}) [κ = {kappa:.2f}]"


def print_summary(results: List["DMLResult"]) -> None:
    """
    Print a formatted summary of multiple DML results to console.
    
    Parameters
    ----------
    results : list of DMLResult
        List of results to summarize.
    """
    print("\n" + "=" * 70)
    print("DML DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    for i, r in enumerate(results):
        if i > 0:
            print("-" * 70)
        
        print(f"\nLearner: {r.learner}")
        print(f"  θ̂ = {r.theta:.4f}  (SE = {r.se:.4f})")
        print(f"  95% CI: [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
        print(f"  κ_DML = {r.kappa:.2f}  |  R²(D|X) = {r.r_squared_d:.3f}  |  n = {r.n}")
    
    print("\n" + "=" * 70)
    
    # Summary statistics across results
    if len(results) > 1:
        kappas = [r.kappa for r in results]
        thetas = [r.theta for r in results]
        
        print(f"\nκ_DML range: [{min(kappas):.2f}, {max(kappas):.2f}]")
        print(f"θ̂ range: [{min(thetas):.4f}, {max(thetas):.4f}]")
        print(f"Estimate spread: {max(thetas) - min(thetas):.4f}")
        print("=" * 70 + "\n")
