"""
DML Diagnostic Plotting Functions
==================================

Publication-quality visualizations for DML condition number diagnostics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Matplotlib import with fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        )


def plot_kappa_summary(
    result: "DMLResult",
    ax: Optional[Any] = None,
    show_interpretation: bool = True,
    figsize: Tuple[float, float] = (8, 5),
) -> Any:
    """
    Create a summary visualization of DML results with κ_DML diagnostic.
    
    Displays the point estimate with confidence interval, the condition
    number κ_DML, and contextual information about the estimation.
    
    Parameters
    ----------
    result : DMLResult
        Results object from DMLDiagnostic.fit().
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    show_interpretation : bool, default True
        Whether to include interpretation text.
    figsize : tuple, default (8, 5)
        Figure size if creating new figure.
        
    Returns
    -------
    matplotlib Axes
        The axes object with the plot.
        
    Examples
    --------
    >>> from dml_diagnostic import DMLDiagnostic, load_lalonde, plot_kappa_summary
    >>> Y, D, X = load_lalonde('experimental')
    >>> result = DMLDiagnostic().fit(Y, D, X)
    >>> plot_kappa_summary(result)
    """
    _check_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Main estimate visualization
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Point estimate with CI
    ax.errorbar(
        x=0.5, y=result.theta,
        yerr=[[result.theta - result.ci_lower], [result.ci_upper - result.theta]],
        fmt='o', color='#2E86AB', markersize=12, capsize=8, capthick=2,
        elinewidth=2, label=f'θ̂ = {result.theta:.3f}'
    )
    
    # Annotations
    text_x = 0.75
    
    # Main result text
    ax.text(
        text_x, result.theta,
        f'θ̂ = {result.theta:.3f}\n'
        f'SE = {result.se:.3f}\n'
        f'95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]',
        fontsize=11, va='center', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # κ_DML display with context
    kappa_text = f'κ_DML = {result.kappa:.2f}'
    
    # Add effective sample size interpretation
    effective_n = result.n / result.kappa
    kappa_text += f'\n(effective n ≈ {effective_n:.0f})'
    
    ax.text(
        0.95, 0.95, kappa_text,
        transform=ax.transAxes,
        fontsize=12, va='top', ha='right',
        fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#F0F0F0', edgecolor='gray')
    )
    
    # Design info
    info_text = f'n = {result.n}  |  R²(D|X) = {result.r_squared_d:.3f}  |  learner: {result.learner}'
    ax.text(
        0.5, -0.12, info_text,
        transform=ax.transAxes,
        fontsize=10, va='top', ha='center',
        style='italic', color='gray'
    )
    
    # Interpretation
    if show_interpretation:
        from dml_diagnostic.diagnostics import kappa_interpretation
        interp = kappa_interpretation(result.kappa, result.n, result.r_squared_d)
        
        ax.text(
            0.5, -0.22, 
            f"Guidance: {interp['guidance'][:100]}...",
            transform=ax.transAxes,
            fontsize=9, va='top', ha='center',
            wrap=True, color='#555555'
        )
    
    # Styling
    ax.set_xlim(0, 1.5)
    ax.set_xticks([])
    ax.set_ylabel('Treatment Effect (θ)', fontsize=12)
    ax.set_title('DML Estimate with Condition Number Diagnostic', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    plt.tight_layout()
    
    return ax


def plot_overlap(
    D: NDArray,
    X: NDArray,
    method: str = 'logistic',
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Any:
    """
    Visualize propensity score overlap for treatment effect estimation.
    
    Creates a histogram of estimated propensity scores by treatment group,
    highlighting the overlap region.
    
    Parameters
    ----------
    D : array-like
        Treatment indicator (binary).
    X : array-like
        Covariates.
    method : str, default 'logistic'
        Method for propensity estimation: 'logistic' or 'rf'.
    ax : matplotlib Axes, optional
        Axes to plot on.
    figsize : tuple, default (8, 5)
        Figure size if creating new figure.
        
    Returns
    -------
    matplotlib Axes
        The axes object with the plot.
    """
    _check_matplotlib()
    
    from dml_diagnostic.diagnostics import overlap_check
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Compute propensity scores
    overlap_info = overlap_check(D, X, method=method)
    e_hat = overlap_info['e_hat']
    D = np.asarray(D).ravel()
    
    # Histograms by treatment status
    bins = np.linspace(0, 1, 31)
    
    ax.hist(
        e_hat[D == 0], bins=bins, alpha=0.6, 
        label=f'Control (n={np.sum(D==0)})', color='#E8505B',
        edgecolor='white', linewidth=0.5
    )
    ax.hist(
        e_hat[D == 1], bins=bins, alpha=0.6,
        label=f'Treated (n={np.sum(D==1)})', color='#2E86AB',
        edgecolor='white', linewidth=0.5
    )
    
    # Overlap region shading
    ax.axvspan(0.1, 0.9, alpha=0.1, color='green', label='Common support region')
    
    # Summary statistics
    summary_text = (
        f"Propensity Score Summary:\n"
        f"  Range: [{overlap_info['e_min']:.3f}, {overlap_info['e_max']:.3f}]\n"
        f"  Mean (treated): {overlap_info['e_mean_treated']:.3f}\n"
        f"  Mean (control): {overlap_info['e_mean_control']:.3f}\n"
        f"  R²(D|X): {overlap_info['r_squared_d']:.3f}"
    )
    
    ax.text(
        0.98, 0.95, summary_text,
        transform=ax.transAxes,
        fontsize=9, va='top', ha='right',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )
    
    # Styling
    ax.set_xlabel('Propensity Score e(X)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Propensity Score Overlap Diagnostic', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    return ax


def plot_kappa_ci_cone(
    results: List["DMLResult"],
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (8, 6),
    show_labels: bool = True,
) -> Any:
    """
    Create a 'cone of uncertainty' plot showing CI width vs κ_DML.
    
    This visualization demonstrates how confidence interval width
    scales with the condition number across different specifications.
    
    Parameters
    ----------
    results : list of DMLResult
        List of results from different learners/specifications.
    ax : matplotlib Axes, optional
        Axes to plot on.
    figsize : tuple, default (8, 6)
        Figure size if creating new figure.
    show_labels : bool, default True
        Whether to label each point with learner name.
        
    Returns
    -------
    matplotlib Axes
        The axes object with the plot.
    """
    _check_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    kappas = [r.kappa for r in results]
    ci_lengths = [r.ci_length for r in results]
    learners = [r.learner for r in results]
    
    # Colors for different learners
    unique_learners = list(set(learners))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_learners)))
    color_map = {l: c for l, c in zip(unique_learners, colors)}
    
    # Scatter plot
    for r in results:
        ax.scatter(
            r.kappa, r.ci_length,
            s=100, c=[color_map[r.learner]], 
            edgecolor='white', linewidth=1.5,
            zorder=3
        )
        
        if show_labels:
            ax.annotate(
                r.learner,
                (r.kappa, r.ci_length),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.8
            )
    
    # Theoretical relationship: CI ∝ κ/√n
    if len(kappas) > 1:
        kappa_range = np.linspace(min(kappas) * 0.9, max(kappas) * 1.1, 100)
        # Use median n and median score variance to estimate scaling
        median_n = np.median([r.n for r in results])
        theoretical_ci = kappa_range / np.sqrt(median_n) * np.median([r.se * np.sqrt(r.n) / r.kappa for r in results]) * 2 * 1.96
        ax.plot(
            kappa_range, theoretical_ci,
            '--', color='gray', alpha=0.5,
            label='Theoretical: CI ∝ κ/√n'
        )
    
    # Styling
    ax.set_xlabel('Condition Number κ_DML', fontsize=12)
    ax.set_ylabel('95% CI Length', fontsize=12)
    ax.set_title('Conditioning and Confidence Interval Width', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    return ax


def plot_comparison(
    results: List["DMLResult"],
    true_theta: Optional[float] = None,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> Any:
    """
    Compare DML estimates across multiple learners/specifications.
    
    Parameters
    ----------
    results : list of DMLResult
        List of results to compare.
    true_theta : float, optional
        True parameter value (if known) for reference line.
    ax : matplotlib Axes, optional
        Axes to plot on.
    figsize : tuple, default (10, 6)
        Figure size if creating new figure.
        
    Returns
    -------
    matplotlib Axes
        The axes object with the plot.
    """
    _check_matplotlib()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    n_results = len(results)
    y_positions = np.arange(n_results)
    
    # Plot each result
    for i, r in enumerate(results):
        # Point estimate with CI
        ax.errorbar(
            x=r.theta, y=i,
            xerr=[[r.theta - r.ci_lower], [r.ci_upper - r.theta]],
            fmt='o', markersize=8, capsize=5, capthick=1.5,
            elinewidth=1.5
        )
        
        # Label with κ_DML
        label = f'{r.learner}  (κ = {r.kappa:.2f})'
        ax.text(
            r.ci_upper + (r.ci_upper - r.ci_lower) * 0.1, i,
            label, va='center', fontsize=10
        )
    
    # True value reference
    if true_theta is not None:
        ax.axvline(
            x=true_theta, color='red', linestyle='--', 
            alpha=0.7, label=f'True θ = {true_theta}'
        )
        ax.legend(loc='best')
    
    # Styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(['' for _ in results])  # Labels are in the plot
    ax.set_xlabel('Treatment Effect (θ)', fontsize=12)
    ax.set_title('DML Estimates Comparison with κ_DML', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    
    plt.tight_layout()
    
    return ax
