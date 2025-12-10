"""
LaLonde/NSW Data Loader
=======================

Functions for loading the LaLonde (1986) / NSW job training dataset.

This canonical dataset is ideal for demonstrating DML condition number
diagnostics because:
- Experimental sample: Good overlap, moderate κ_DML
- Observational sample: Poor overlap, large κ_DML

References:
    LaLonde, R.J. (1986). "Evaluating the Econometric Evaluations of Training
    Programs with Experimental Data." American Economic Review, 76(4), 604–620.
    
    Dehejia, R.H. and Wahba, S. (1999). "Causal Effects in Nonexperimental 
    Studies: Reevaluating the Evaluation of Training Programs." JASA.
"""

from typing import Optional, Tuple, Literal

import numpy as np
import pandas as pd


# Data URLs (Dehejia-Wahba versions)
NSW_URL = "https://users.nber.org/~rdehejia/data/nsw_dw.dta"
PSID_URL = "https://users.nber.org/~rdehejia/data/psid_controls.dta"
CPS_URL = "https://users.nber.org/~rdehejia/data/cps_controls.dta"


def _download_stata(url: str) -> pd.DataFrame:
    """Download a Stata file from URL."""
    try:
        df = pd.read_stata(url)
        df.columns = df.columns.str.lower().str.strip()
        return df
    except Exception as e:
        raise RuntimeError(
            f"Failed to download data from {url}. "
            f"Check your internet connection. Error: {e}"
        )


def load_lalonde(
    sample: Literal["experimental", "observational", "both"] = "experimental",
    return_dataframe: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load LaLonde/NSW data for DML analysis.
    
    Provides two contrasting datasets:
    - Experimental: NSW randomised experiment (good overlap)
    - Observational: NSW treated + PSID controls (poor overlap)
    
    Parameters
    ----------
    sample : str, default 'experimental'
        Which sample to load:
        - 'experimental': NSW experiment only (n ≈ 445)
        - 'observational': NSW treated + PSID controls (n ≈ 2675)
        - 'both': Return both samples as a dictionary
    return_dataframe : bool, default False
        If True, return a DataFrame instead of arrays.
    verbose : bool, default False
        Print summary information.
        
    Returns
    -------
    Y : np.ndarray
        Outcome: post-treatment earnings (re78).
    D : np.ndarray
        Treatment indicator (1 = NSW participation).
    X : np.ndarray
        Covariates: age, education, black, hispanic, married, nodegree, re74, re75.
        
    If return_dataframe=True, returns pd.DataFrame instead.
    If sample='both', returns dict with keys 'experimental' and 'observational'.
        
    Examples
    --------
    >>> from dml_diagnostic import load_lalonde
    >>> Y, D, X = load_lalonde(sample='experimental')
    >>> print(f"n = {len(Y)}, treated = {D.sum()}")
    
    >>> # Compare experimental vs observational
    >>> from dml_diagnostic import DMLDiagnostic, load_lalonde
    >>> Y_exp, D_exp, X_exp = load_lalonde('experimental')
    >>> Y_obs, D_obs, X_obs = load_lalonde('observational')
    >>> dml = DMLDiagnostic(learner='lasso')
    >>> print("Experimental:", dml.fit(Y_exp, D_exp, X_exp).kappa)
    >>> print("Observational:", dml.fit(Y_obs, D_obs, X_obs).kappa)
    """
    if sample == "both":
        return {
            "experimental": load_lalonde("experimental", return_dataframe, verbose),
            "observational": load_lalonde("observational", return_dataframe, verbose),
        }
    
    if verbose:
        print(f"Loading LaLonde data (sample='{sample}')...")
    
    # Load NSW experimental data
    nsw = _download_stata(NSW_URL)
    nsw["data_source"] = "NSW"
    
    if sample == "experimental":
        df = nsw
    elif sample == "observational":
        # Load PSID controls and combine with NSW treated
        psid = _download_stata(PSID_URL)
        psid["data_source"] = "PSID"
        if "treat" not in psid.columns:
            psid["treat"] = 0
        
        # Keep NSW treated + PSID controls
        nsw_treated = nsw[nsw["treat"] == 1]
        df = pd.concat([nsw_treated, psid], ignore_index=True)
    else:
        raise ValueError(
            f"Unknown sample: '{sample}'. "
            f"Choose from: 'experimental', 'observational', 'both'"
        )
    
    # Standardise variables
    df["Y"] = df["re78"]
    df["D"] = df["treat"].astype(int)
    
    # Covariate columns
    covariate_cols = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
    
    # Ensure numeric types
    for col in covariate_cols + ["Y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop missing values
    df = df.dropna(subset=["Y", "D"] + covariate_cols)
    
    if verbose:
        n_treated = df["D"].sum()
        n_control = len(df) - n_treated
        print(f"  n = {len(df)} (treated: {n_treated}, control: {n_control})")
    
    if return_dataframe:
        return df
    
    Y = df["Y"].values
    D = df["D"].values
    X = df[covariate_cols].values.astype(np.float64)
    
    return Y, D, X


def get_covariate_names() -> list:
    """
    Get the names of covariates in the LaLonde data.
    
    Returns
    -------
    list
        List of covariate names in the same order as X columns.
    """
    return ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
