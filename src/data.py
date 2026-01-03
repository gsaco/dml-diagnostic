"""
LaLonde Data Loader for Empirical Application.

This module provides functions to load the LaLonde (1986) / Dehejia-Wahba (1999)
dataset for the empirical illustration in Section 6 of "Ill-Conditioned
Orthogonal Scores in Double Machine Learning."

The LaLonde dataset is a canonical benchmark in causal inference where the
randomized experimental sample (NSW) provides ground truth against which
observational methods can be validated.

Experimental vs Observational Samples
-------------------------------------
- Experimental (NSW): Randomized → strong overlap → low κ ≈ 1-2
- Observational (NSW-PSID): Selection bias → weak overlap → high κ > 2

Per Theorem 3.11, higher κ in the observational sample should produce
greater sensitivity to learner choice and wider dispersion of estimates,
which is confirmed in Figure 3 (forest plot).

Data Sources
------------
- LaLonde, R. (1986). Evaluating the Econometric Evaluations of Training
  Programs with Experimental Data. American Economic Review 76(4): 604-620.
- Dehejia, R. & Wahba, S. (1999). Causal Effects in Nonexperimental Studies:
  Reevaluating the Evaluation of Training Programs. JASA 94(448): 1053-1062.
"""

from __future__ import annotations

from typing import Literal, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONSTANTS
# =============================================================================

# URLs for Dehejia-Wahba data (NBER hosting)
NSW_TREATED_URL = "https://users.nber.org/~rdehejia/data/nswre74_treated.txt"
NSW_CONTROL_URL = "https://users.nber.org/~rdehejia/data/nswre74_control.txt"
PSID_CONTROL_URL = "https://users.nber.org/~rdehejia/data/psid_controls.txt"

# Column names for the Dehejia-Wahba format
COLUMN_NAMES = [
    'treat', 'age', 'education', 'black', 'hispanic', 
    'married', 'nodegree', 're74', 're75', 're78'
]

# Covariates used in the analysis
COVARIATE_COLS = ['age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74', 're75']
CONTINUOUS_COLS = ['age', 'education', 're74', 're75']
BINARY_COLS = ['black', 'hispanic', 'married', 'nodegree']

# Experimental benchmark ATE (Section 6)
# This is the "ground truth" from the randomized experiment
EXPERIMENTAL_BENCHMARK = 1794


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def _load_dehejia_wahba_txt(url: str) -> pd.DataFrame:
    """
    Load a Dehejia-Wahba formatted text file from URL.
    
    Parameters
    ----------
    url : str
        URL to the whitespace-separated data file.
    
    Returns
    -------
    df : pd.DataFrame
        Loaded dataframe with standard column names.
    
    Raises
    ------
    RuntimeError
        If data cannot be loaded (e.g., network issues).
    """
    try:
        df = pd.read_csv(
            url, 
            sep=r'\s+',  # Whitespace separated
            header=None, 
            names=COLUMN_NAMES
        )
        return df
    except Exception as e:
        raise RuntimeError(
            f"Failed to load data from {url}. "
            f"Check internet connection. Error: {e}"
        )


def load_lalonde(
    mode: Literal['experimental', 'observational'] = 'observational',
    standardize: bool = True,
    return_df: bool = False,
) -> Tuple[NDArray, NDArray, NDArray] | Tuple[NDArray, NDArray, NDArray, pd.DataFrame]:
    """
    Load the LaLonde / Dehejia-Wahba dataset.
    
    This function loads the canonical LaLonde dataset in two modes:
    - 'experimental': NSW treated vs. NSW control (randomized experiment)
    - 'observational': NSW treated vs. PSID control (observational comparison)
    
    The experimental sample provides a benchmark treatment effect (~$1,794),
    while the observational sample exhibits selection bias and weak overlap.
    Per Theorem 3.11, the higher κ in the observational sample should
    produce greater learner sensitivity.
    
    Parameters
    ----------
    mode : {'experimental', 'observational'}, default 'observational'
        Which sample to load:
        - 'experimental': NSW treated + NSW control (n ≈ 445)
        - 'observational': NSW treated + PSID control (n ≈ 2,675)
    standardize : bool, default True
        Whether to standardize continuous covariates (age, education, re74, re75).
        Recommended for neural network convergence.
    return_df : bool, default False
        If True, also return the full DataFrame.
    
    Returns
    -------
    y : ndarray of shape (n,)
        Outcome variable (re78 - earnings in 1978, dollars).
    d : ndarray of shape (n,)
        Treatment indicator (1 = NSW job training program, 0 = control).
    X : ndarray of shape (n, 8)
        Covariate matrix [age, education, black, hispanic, married, 
        nodegree, re74, re75].
    df : pd.DataFrame (optional)
        Full dataframe if return_df=True.
    
    Notes
    -----
    The experimental sample has low κ ≈ 1-2 because treatment was randomized,
    ensuring strong overlap. The observational sample has higher κ because
    PSID controls differ systematically from NSW participants (different
    labor market attachment, demographics). This creates the "ill-conditioned"
    setting where DML estimates are sensitive to learner choice.
    
    Examples
    --------
    >>> y, d, X = load_lalonde(mode='experimental')
    >>> print(f"N = {len(y)}, Treated = {int(d.sum())}")
    N = 445, Treated = 185
    """
    mode = mode.lower()
    if mode not in ['experimental', 'observational']:
        raise ValueError(f"mode must be 'experimental' or 'observational', got '{mode}'")
    
    # Load NSW treated (always needed)
    print(f"Loading LaLonde data (mode='{mode}')...")
    df_treated = _load_dehejia_wahba_txt(NSW_TREATED_URL)
    
    # Load appropriate control group
    if mode == 'experimental':
        df_control = _load_dehejia_wahba_txt(NSW_CONTROL_URL)
        sample_name = "Experimental (NSW)"
    else:
        df_control = _load_dehejia_wahba_txt(PSID_CONTROL_URL)
        sample_name = "Observational (NSW-PSID)"
    
    # Combine treated and control
    df = pd.concat([df_treated, df_control], ignore_index=True)
    
    # Extract variables
    y = df['re78'].values.astype(np.float64)
    d = df['treat'].values.astype(np.float64)
    X_raw = df[COVARIATE_COLS].values.astype(np.float64)
    
    # Standardize continuous covariates if requested
    if standardize:
        X = X_raw.copy()
        cont_indices = [COVARIATE_COLS.index(col) for col in CONTINUOUS_COLS]
        scaler = StandardScaler()
        X[:, cont_indices] = scaler.fit_transform(X[:, cont_indices])
    else:
        X = X_raw
    
    # Print summary
    n_treated = int(d.sum())
    n_control = len(d) - n_treated
    print(f"  Sample: {sample_name}")
    print(f"  N = {len(y):,} (Treated: {n_treated}, Control: {n_control})")
    print(f"  Covariates: {COVARIATE_COLS}")
    print(f"  Standardized: {standardize}")
    
    if return_df:
        return y, d, X, df
    return y, d, X


# =============================================================================
# SAMPLE DIAGNOSTICS
# =============================================================================

def get_sample_summary(y: NDArray, d: NDArray, X: NDArray) -> dict:
    """
    Compute summary statistics for a sample.
    
    Parameters
    ----------
    y : ndarray of shape (n,)
        Outcome variable.
    d : ndarray of shape (n,)
        Treatment indicator.
    X : ndarray of shape (n, p)
        Covariate matrix.
    
    Returns
    -------
    summary : dict
        Dictionary containing:
        - n, n_treated, n_control: Sample sizes
        - prop_treated: Treatment proportion
        - mean_outcome, mean_outcome_treated, mean_outcome_control: Outcomes
        - naive_ate: Simple difference in means (biased in observational data)
        - n_covariates: Number of covariates
    """
    n = len(y)
    n_treated = int(d.sum())
    n_control = n - n_treated
    
    # Boolean masks (handle float treatment indicator)
    treated_mask = d > 0.5
    control_mask = d < 0.5
    
    return {
        'n': n,
        'n_treated': n_treated,
        'n_control': n_control,
        'prop_treated': n_treated / n,
        'mean_outcome': float(y.mean()),
        'mean_outcome_treated': float(y[treated_mask].mean()),
        'mean_outcome_control': float(y[control_mask].mean()),
        'naive_ate': float(y[treated_mask].mean() - y[control_mask].mean()),
        'n_covariates': X.shape[1],
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'load_lalonde',
    'get_sample_summary',
    'EXPERIMENTAL_BENCHMARK',
    'COVARIATE_COLS',
]
