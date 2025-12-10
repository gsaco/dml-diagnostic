"""
DML Diagnostic Plotting Functions
==================================

Publication-quality visualizations for DML condition number diagnostics.

This module provides figures suitable for academic publication, following
The Econometrics Journal style guidelines. All plots are designed to clearly
communicate the role of κ_DML in finite-sample DML inference.

Reference: Saco (2025), "Finite-Sample Failures and Condition-Number 
Diagnostics in Double Machine Learning"
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

# Matplotlib import with fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Publication-Quality Style Settings
# =============================================================================

# Colorblind-friendly palette for academic figures
COLORS = {
    'primary': '#2E86AB',       # Steel blue - main estimates
    'secondary': '#A23B72',     # Raspberry - alternative/comparison
    'accent': '#F18F01',        # Orange - highlights
    'success': '#2C7BB6',       # Blue - experimental/well-conditioned
    'warning': '#D95F02',       # Dark orange - moderate
    'danger': '#D7191C',        # Red - observational/ill-conditioned
    'neutral': '#666666',       # Gray - reference lines
    'light': '#E0E0E0',         # Light gray - backgrounds
    # Learner-specific colors
    'LIN': '#1B9E77',           # Teal
    'LASSO': '#D95F02',         # Orange  
    'RF': '#7570B3',            # Purple
}

# Standard figure sizes (width, height) in inches
# The Econometrics Journal: single column ~3.5in, full page ~7in
FIGURE_SIZES = {
    'single': (3.5, 2.8),       # Single column
    'wide': (5.5, 3.5),         # 1.5 column
    'full': (7.0, 4.5),         # Full page width
    'square': (4.5, 4.5),       # Square format
}


def set_publication_style():
    """
    Configure matplotlib for publication-quality figures.
    
    Sets font sizes, line widths, and other parameters suitable for
    The Econometrics Journal and similar academic publications.
    """
    if not HAS_MATPLOTLIB:
        return
        
    plt.rcParams.update({
        # Figure
        'figure.figsize': FIGURE_SIZES['wide'],
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        
        # Font
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'serif'],
        'font.size': 10,
        'mathtext.fontset': 'cm',
        
        # Axes
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Ticks
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        
        # Legend
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        
        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        
        # Grid
        'axes.grid': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })


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
    figsize: Tuple[float, float] = None,
    benchmark: Optional[float] = None,
    title: Optional[str] = None,
) -> Any:
    """
    Create a publication-quality summary visualization of DML results.
    
    Displays the point estimate with confidence interval and the condition
    number κ_DML, emphasizing the diagnostic information.
    
    This visualization is designed for inclusion in academic papers and
    presentations, following the style of Saco (2025).
    
    Parameters
    ----------
    result : DMLResult
        Results object from DMLDiagnostic.fit().
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    show_interpretation : bool, default True
        Whether to include interpretation text.
    figsize : tuple, optional
        Figure size. If None, uses publication default.
    benchmark : float, optional
        Reference value (e.g., experimental benchmark) to display.
    title : str, optional
        Custom title. If None, uses default.
        
    Returns
    -------
    matplotlib Axes
        The axes object with the plot.
        
    Examples
    --------
    >>> from dml_diagnostic import DMLDiagnostic, load_lalonde, plot_kappa_summary
    >>> Y, D, X = load_lalonde('experimental')
    >>> result = DMLDiagnostic().fit(Y, D, X)
    >>> plot_kappa_summary(result, benchmark=1794)
    """
    _check_matplotlib()
    set_publication_style()
    
    if figsize is None:
        figsize = FIGURE_SIZES['wide']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Reference line at zero
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='-', alpha=0.3, linewidth=0.8)
    
    # Benchmark reference if provided
    if benchmark is not None:
        ax.axhline(y=benchmark, color=COLORS['success'], linestyle='--', 
                   alpha=0.7, linewidth=1.2, label=f'Benchmark = {benchmark:,.0f}')
    
    # Point estimate with CI
    ax.errorbar(
        x=0.5, y=result.theta,
        yerr=[[result.theta - result.ci_lower], [result.ci_upper - result.theta]],
        fmt='o', color=COLORS['primary'], markersize=10, capsize=6, capthick=1.5,
        elinewidth=1.5, label=f'θ̂ = {result.theta:,.0f}'
    )
    
    # Main result annotation
    text_x = 0.72
    result_text = (
        f"$\\hat{{\\theta}}$ = {result.theta:,.0f}\n"
        f"SE = {result.se:,.0f}\n"
        f"95% CI: [{result.ci_lower:,.0f}, {result.ci_upper:,.0f}]"
    )
    ax.text(
        text_x, result.theta,
        result_text,
        fontsize=10, va='center', ha='left',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                  edgecolor=COLORS['light'], alpha=0.95)
    )
    
    # κ_DML diagnostic display (emphasized)
    effective_n = result.n / result.kappa
    kappa_text = (
        f"$\\kappa_{{\\mathrm{{DML}}}}$ = {result.kappa:.2f}\n"
        f"Effective $n$ ≈ {effective_n:.0f}"
    )
    
    ax.text(
        0.97, 0.97, kappa_text,
        transform=ax.transAxes,
        fontsize=11, va='top', ha='right',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9E6', 
                  edgecolor=COLORS['accent'], linewidth=1.5)
    )
    
    # Design information
    info_text = f"$n$ = {result.n}  |  $R^2(D|X)$ = {result.r_squared_d:.3f}  |  {result.learner}"
    ax.text(
        0.5, -0.10, info_text,
        transform=ax.transAxes,
        fontsize=9, va='top', ha='center',
        style='italic', color=COLORS['neutral']
    )
    
    # Interpretation guidance
    if show_interpretation:
        from dml_diagnostic.diagnostics import kappa_interpretation
        interp = kappa_interpretation(result.kappa, result.n, result.r_squared_d)
        
        guidance_short = interp['guidance'][:120] + "..." if len(interp['guidance']) > 120 else interp['guidance']
        ax.text(
            0.5, -0.18, 
            guidance_short,
            transform=ax.transAxes,
            fontsize=8, va='top', ha='center',
            wrap=True, color='#555555'
        )
    
    # Styling
    ax.set_xlim(0, 1.4)
    ax.set_xticks([])
    ax.set_ylabel('Treatment Effect ($\\theta$)', fontsize=11)
    
    if title is None:
        title = 'DML Estimate with Condition Number Diagnostic'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    ax.spines['bottom'].set_visible(False)
    
    if benchmark is not None:
        ax.legend(loc='lower left', framealpha=0.9)
    
    plt.tight_layout()
    
    return ax


def plot_overlap(
    D: NDArray,
    X: NDArray,
    method: str = 'logistic',
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = None,
    title: Optional[str] = None,
) -> Any:
    """
    Visualize propensity score overlap for treatment effect estimation.
    
    Creates a publication-quality histogram of estimated propensity scores 
    by treatment group, highlighting potential overlap violations that
    contribute to ill-conditioned DML estimation.
    
    As discussed in Saco (2025), limited overlap directly affects the
    condition number κ_DML through the relationship:
        κ_DML ≈ 1 / Var(D - m₀(X)) ≈ 1 / (Var(D)(1 - R²(D|X)))
    
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
    figsize : tuple, optional
        Figure size. If None, uses publication default.
    title : str, optional
        Custom title.
        
    Returns
    -------
    matplotlib Axes
        The axes object with the plot.
        
    Notes
    -----
    The overlap region (propensity scores between 0.1 and 0.9) is highlighted
    to indicate the common support region. Concentration of propensity scores
    near 0 or 1 indicates limited overlap and predicts a large κ_DML.
    """
    _check_matplotlib()
    set_publication_style()
    
    from dml_diagnostic.diagnostics import overlap_check
    
    if figsize is None:
        figsize = FIGURE_SIZES['wide']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Compute propensity scores
    overlap_info = overlap_check(D, X, method=method)
    e_hat = overlap_info['e_hat']
    D = np.asarray(D).ravel()
    
    # Histograms by treatment status with publication colors
    bins = np.linspace(0, 1, 31)
    
    ax.hist(
        e_hat[D == 0], bins=bins, alpha=0.65, 
        label=f'Control ($n={np.sum(D==0)}$)', color=COLORS['danger'],
        edgecolor='white', linewidth=0.5
    )
    ax.hist(
        e_hat[D == 1], bins=bins, alpha=0.65,
        label=f'Treated ($n={np.sum(D==1)}$)', color=COLORS['success'],
        edgecolor='white', linewidth=0.5
    )
    
    # Overlap region shading
    ax.axvspan(0.1, 0.9, alpha=0.08, color='green', label='Common support')
    
    # Summary statistics box
    r2_d = overlap_info['r_squared_d']
    summary_text = (
        f"$e(X)$ range: [{overlap_info['e_min']:.2f}, {overlap_info['e_max']:.2f}]\n"
        f"$\\bar{{e}}$ (treated): {overlap_info['e_mean_treated']:.2f}\n"
        f"$\\bar{{e}}$ (control): {overlap_info['e_mean_control']:.2f}\n"
        f"$R^2(D|X)$: {r2_d:.3f}"
    )
    
    ax.text(
        0.97, 0.95, summary_text,
        transform=ax.transAxes,
        fontsize=9, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                  edgecolor=COLORS['light'], alpha=0.95)
    )
    
    # Styling
    ax.set_xlabel('Propensity Score $\\hat{e}(X)$', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    
    if title is None:
        title = 'Propensity Score Overlap Diagnostic'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    return ax


def plot_kappa_ci_cone(
    results: List["DMLResult"],
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = None,
    show_labels: bool = True,
    show_theory: bool = True,
    title: Optional[str] = None,
    group_by: str = 'learner',
) -> Any:
    """
    Create a publication-quality 'cone of uncertainty' plot.
    
    This visualization demonstrates how confidence interval width scales
    with the condition number κ_DML across different specifications,
    implementing the key insight from Proposition 3.4 of Saco (2025):
    
        |CI| ∝ κ_DML / √n
    
    The plot shows how increasing κ_DML expands the "cone of uncertainty,"
    making inference less informative even if coverage is maintained.
    
    Parameters
    ----------
    results : list of DMLResult
        List of results from different learners/specifications.
    ax : matplotlib Axes, optional
        Axes to plot on.
    figsize : tuple, optional
        Figure size. If None, uses publication default.
    show_labels : bool, default True
        Whether to label each point.
    show_theory : bool, default True
        Whether to show theoretical CI ∝ κ/√n relationship.
    title : str, optional
        Custom title.
    group_by : str, default 'learner'
        How to color points: 'learner' or 'design'.
        
    Returns
    -------
    matplotlib Axes
        The axes object with the plot.
        
    Notes
    -----
    This figure corresponds to the theoretical prediction that CI widths
    scale with κ_DML: as overlap deteriorates, residual treatment variance
    decreases, κ_DML increases, and inference becomes less precise.
    """
    _check_matplotlib()
    set_publication_style()
    
    if figsize is None:
        figsize = FIGURE_SIZES['wide']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    kappas = np.array([r.kappa for r in results])
    ci_lengths = np.array([r.ci_length for r in results])
    learners = [r.learner for r in results]
    
    # Use learner-specific colors from COLORS dict
    learner_colors = {
        'LIN': COLORS.get('LIN', '#1B9E77'),
        'LASSO': COLORS.get('LASSO', '#D95F02'),
        'RF': COLORS.get('RF', '#7570B3'),
        'lin': COLORS.get('LIN', '#1B9E77'),
        'lasso': COLORS.get('LASSO', '#D95F02'),
        'rf': COLORS.get('RF', '#7570B3'),
    }
    
    # Plot points by learner
    plotted_learners = set()
    for r in results:
        learner_upper = r.learner.upper()
        color = learner_colors.get(r.learner, learner_colors.get(learner_upper, COLORS['primary']))
        
        # Only add label once per learner
        label = r.learner.upper() if r.learner.upper() not in plotted_learners else None
        plotted_learners.add(r.learner.upper())
        
        ax.scatter(
            r.kappa, r.ci_length,
            s=80, c=color, 
            edgecolor='white', linewidth=1.0,
            zorder=3, label=label, alpha=0.85
        )
        
        if show_labels:
            # Annotate with design info
            ax.annotate(
                '',  # No text, just for spacing
                (r.kappa, r.ci_length),
                fontsize=8, alpha=0.7
            )
    
    # Theoretical relationship: CI ∝ κ/√n
    if show_theory and len(kappas) > 1:
        kappa_range = np.linspace(max(0.1, kappas.min() * 0.8), kappas.max() * 1.15, 100)
        
        # Estimate the scaling constant from the data
        # CI_length = 2 * 1.96 * SE = 2 * 1.96 * κ * σ_ψ / √n
        # So CI_length / κ ≈ constant * 1/√n
        median_n = np.median([r.n for r in results])
        scale_const = np.median(ci_lengths / kappas) if np.all(kappas > 0) else 1.0
        
        theoretical_ci = scale_const * kappa_range
        ax.plot(
            kappa_range, theoretical_ci,
            '--', color=COLORS['neutral'], alpha=0.6, linewidth=1.5,
            label='$|\\mathrm{CI}| \\propto \\kappa_{\\mathrm{DML}}$'
        )
    
    # Styling
    ax.set_xlabel('Condition Number $\\kappa_{\\mathrm{DML}}$', fontsize=11)
    ax.set_ylabel('95\\% CI Length', fontsize=11)
    
    if title is None:
        title = 'Confidence Interval Width vs Conditioning'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    return ax


def plot_kappa_comparison(
    results_dict: Dict[str, List["DMLResult"]],
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = None,
    title: Optional[str] = None,
) -> Any:
    """
    Compare κ_DML across different designs (e.g., experimental vs observational).
    
    Creates a grouped bar chart showing κ_DML by learner for each design,
    highlighting the conditioning differences between designs.
    
    This visualization directly implements the comparison in Section 7 of
    Saco (2025), contrasting well-conditioned experimental designs with
    ill-conditioned observational re-analyses.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping design names to lists of DMLResult objects.
        Example: {'Experimental': [r1, r2, r3], 'Observational': [r4, r5, r6]}
    ax : matplotlib Axes, optional
        Axes to plot on.
    figsize : tuple, optional
        Figure size. If None, uses publication default.
    title : str, optional
        Custom title.
        
    Returns
    -------
    matplotlib Axes
        The axes object with the plot.
    """
    _check_matplotlib()
    set_publication_style()
    
    if figsize is None:
        figsize = FIGURE_SIZES['wide']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    design_names = list(results_dict.keys())
    n_designs = len(design_names)
    
    # Get learner names from first design
    first_results = list(results_dict.values())[0]
    learner_names = [r.learner.upper() for r in first_results]
    n_learners = len(learner_names)
    
    # Bar positioning
    x = np.arange(n_learners)
    width = 0.35 if n_designs == 2 else 0.8 / n_designs
    
    # Design colors
    design_colors = [COLORS['success'], COLORS['danger']] if n_designs == 2 else \
                    plt.cm.Set2(np.linspace(0, 1, n_designs))
    
    # Plot bars for each design
    for i, (design_name, results) in enumerate(results_dict.items()):
        kappas = [r.kappa for r in results]
        offset = (i - (n_designs - 1) / 2) * width
        
        bars = ax.bar(
            x + offset, kappas, width,
            label=design_name, color=design_colors[i],
            edgecolor='white', linewidth=0.8
        )
    
    # Styling
    ax.set_ylabel('$\\kappa_{\\mathrm{DML}}$', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(learner_names)
    ax.set_xlabel('Nuisance Learner', fontsize=11)
    
    if title is None:
        title = 'Condition Number by Design and Learner'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    return ax


def plot_comparison(
    results: List["DMLResult"],
    true_theta: Optional[float] = None,
    benchmark: Optional[float] = None,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = None,
    title: Optional[str] = None,
    show_kappa: bool = True,
) -> Any:
    """
    Publication-quality forest plot comparing DML estimates.
    
    Creates a horizontal forest plot showing point estimates and confidence
    intervals across multiple learners or specifications, with κ_DML
    annotations to highlight conditioning differences.
    
    This visualization is designed for academic papers, enabling direct
    comparison of estimates while emphasizing the diagnostic role of κ_DML.
    
    Parameters
    ----------
    results : list of DMLResult
        List of results to compare.
    true_theta : float, optional
        True parameter value (if known, e.g., from simulation) for reference.
    benchmark : float, optional
        Benchmark value (e.g., experimental estimate) for reference.
    ax : matplotlib Axes, optional
        Axes to plot on.
    figsize : tuple, optional
        Figure size. If None, uses publication default.
    title : str, optional
        Custom title.
    show_kappa : bool, default True
        Whether to display κ_DML values.
        
    Returns
    -------
    matplotlib Axes
        The axes object with the plot.
    """
    _check_matplotlib()
    set_publication_style()
    
    if figsize is None:
        figsize = FIGURE_SIZES['wide']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    n_results = len(results)
    y_positions = np.arange(n_results)
    
    # Learner-specific colors
    learner_colors = {
        'LIN': COLORS.get('LIN', '#1B9E77'),
        'LASSO': COLORS.get('LASSO', '#D95F02'),
        'RF': COLORS.get('RF', '#7570B3'),
        'lin': COLORS.get('LIN', '#1B9E77'),
        'lasso': COLORS.get('LASSO', '#D95F02'),
        'rf': COLORS.get('RF', '#7570B3'),
    }
    
    # Plot each result
    for i, r in enumerate(results):
        learner_upper = r.learner.upper()
        color = learner_colors.get(r.learner, learner_colors.get(learner_upper, COLORS['primary']))
        
        # Point estimate with CI
        ax.errorbar(
            x=r.theta, y=i,
            xerr=[[r.theta - r.ci_lower], [r.ci_upper - r.theta]],
            fmt='o', markersize=7, capsize=4, capthick=1.2,
            elinewidth=1.2, color=color
        )
        
        # Label with κ_DML
        if show_kappa:
            label = f'{r.learner.upper()}  ($\\kappa$ = {r.kappa:.1f})'
        else:
            label = f'{r.learner.upper()}'
        
        ci_width = r.ci_upper - r.ci_lower
        ax.text(
            r.ci_upper + ci_width * 0.05, i,
            label, va='center', fontsize=9, color=color
        )
    
    # Reference lines
    if true_theta is not None:
        ax.axvline(
            x=true_theta, color=COLORS['success'], linestyle='-', 
            alpha=0.8, linewidth=1.5, label=f'True $\\theta_0$ = {true_theta:,.0f}'
        )
    
    if benchmark is not None:
        ax.axvline(
            x=benchmark, color=COLORS['accent'], linestyle='--', 
            alpha=0.8, linewidth=1.5, label=f'Benchmark = {benchmark:,.0f}'
        )
    
    # Zero reference
    ax.axvline(x=0, color=COLORS['neutral'], linestyle='-', alpha=0.3, linewidth=0.8)
    
    # Styling
    ax.set_yticks(y_positions)
    ax.set_yticklabels(['' for _ in results])
    ax.set_xlabel('Treatment Effect ($\\theta$)', fontsize=11)
    
    if title is None:
        title = 'DML Estimates Comparison'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    if true_theta is not None or benchmark is not None:
        ax.legend(loc='best', framealpha=0.9)
    
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    
    plt.tight_layout()
    
    return ax


def save_figure(
    fig_or_ax,
    filename: str,
    formats: List[str] = None,
    dpi: int = 300,
) -> List[str]:
    """
    Save a figure in publication-quality formats.
    
    Convenience function for saving figures in multiple formats
    (PDF for LaTeX, PNG for slides/web).
    
    Parameters
    ----------
    fig_or_ax : matplotlib Figure or Axes
        The figure or axes to save.
    filename : str
        Base filename (without extension).
    formats : list of str, optional
        File formats to save. Default is ['pdf', 'png'].
    dpi : int, default 300
        Resolution for raster formats.
        
    Returns
    -------
    list of str
        Paths to saved files.
    """
    _check_matplotlib()
    
    if formats is None:
        formats = ['pdf', 'png']
    
    # Get figure from axes if needed
    if hasattr(fig_or_ax, 'figure'):
        fig = fig_or_ax.figure
    else:
        fig = fig_or_ax
    
    saved_paths = []
    for fmt in formats:
        path = f"{filename}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        saved_paths.append(path)
    
    return saved_paths
