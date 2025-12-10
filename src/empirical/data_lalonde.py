"""
Data loading and cleaning functions for the LaLonde/NSW job training dataset.

The LaLonde (1986) dataset is a canonical benchmark for causal inference methods.
It consists of:
- NSW (National Supported Work) experimental sample: randomised treated and controls
- Non-experimental controls from PSID (Panel Study of Income Dynamics) or CPS 
  (Current Population Survey)

We use the Dehejia-Wahba (1999) version of the data, which is publicly available
and widely used in the causal inference literature.

This dataset is ideal for demonstrating the DML condition number diagnostic because:
- Experimental sample: Good overlap, low κ_DML (≈ 4)
- Observational sample: Poor overlap, high κ_DML (> 15)

References:
    LaLonde, R.J. (1986). "Evaluating the Econometric Evaluations of Training
    Programs with Experimental Data." American Economic Review, 76(4), 604–620.
    
    Dehejia, R.H. and Wahba, S. (1999). "Causal Effects in Nonexperimental 
    Studies: Reevaluating the Evaluation of Training Programs." Journal of
    the American Statistical Association, 94(448), 1053–1062.
    
    Busso, M., DiNardo, J., and McCrary, J. (2014). "New Evidence on the Finite
    Sample Properties of Propensity Score Reweighting and Matching Estimators."
    Review of Economics and Statistics, 96(5), 885–897.

Data source: Dehejia-Wahba website (https://users.nber.org/~rdehejia/data/)

Author: Gabriel Saco
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
import os
from io import StringIO


# =============================================================================
# Data URLs (Dehejia-Wahba versions)
# =============================================================================

# NSW experimental treated
NSW_TREATED_URL = "https://users.nber.org/~rdehejia/data/nsw_dw.dta"

# NSW experimental controls (randomised controls)
NSW_CONTROL_URL = "https://users.nber.org/~rdehejia/data/nsw_dw.dta"

# PSID comparison group (non-experimental)
PSID_CONTROL_URL = "https://users.nber.org/~rdehejia/data/psid_controls.dta"

# CPS comparison group (non-experimental)  
CPS_CONTROL_URL = "https://users.nber.org/~rdehejia/data/cps_controls.dta"


def _download_stata_file(url: str, description: str = "data") -> pd.DataFrame:
    """
    Download a Stata .dta file from a URL and return as DataFrame.
    
    Parameters
    ----------
    url : str
        URL to the .dta file.
    description : str
        Description for error messages.
        
    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    try:
        df = pd.read_stata(url)
        return df
    except Exception as e:
        raise RuntimeError(
            f"Failed to download {description} from {url}. "
            f"Error: {e}. "
            f"Please check your internet connection or download manually."
        )


def load_nsw_experimental() -> pd.DataFrame:
    """
    Load the NSW experimental sample (treated + randomised controls).
    
    This is the "good overlap" experimental benchmark where treatment was
    randomly assigned, so propensity scores are approximately 0.5.
    
    Returns
    -------
    pd.DataFrame
        NSW experimental data with columns:
        - treat: treatment indicator (1 = NSW participant)
        - age, education, black, hispanic, married, nodegree: covariates
        - re74, re75: pre-treatment earnings (1974, 1975)
        - re78: post-treatment earnings (1978, outcome)
    """
    df = _download_stata_file(NSW_TREATED_URL, "NSW experimental data")
    
    # Standardise column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Add source indicator
    df['data_source'] = 'NSW_experimental'
    
    return df


def load_psid_controls() -> pd.DataFrame:
    """
    Load the PSID non-experimental comparison group.
    
    This creates a "poor overlap" observational design when combined with
    NSW treated units. The PSID controls are demographically very different
    from the NSW treated population.
    
    Returns
    -------
    pd.DataFrame
        PSID control data with same structure as NSW data.
    """
    df = _download_stata_file(PSID_CONTROL_URL, "PSID controls")
    
    # Standardise column names
    df.columns = df.columns.str.lower().str.strip()
    
    # PSID controls have treat = 0 by construction
    if 'treat' not in df.columns:
        df['treat'] = 0
    
    # Add source indicator
    df['data_source'] = 'PSID_control'
    
    return df


def load_cps_controls() -> pd.DataFrame:
    """
    Load the CPS non-experimental comparison group.
    
    Alternative non-experimental controls from Current Population Survey.
    Also creates poor overlap when combined with NSW treated.
    
    Returns
    -------
    pd.DataFrame
        CPS control data with same structure as NSW data.
    """
    df = _download_stata_file(CPS_CONTROL_URL, "CPS controls")
    
    # Standardise column names
    df.columns = df.columns.str.lower().str.strip()
    
    # CPS controls have treat = 0 by construction
    if 'treat' not in df.columns:
        df['treat'] = 0
    
    # Add source indicator
    df['data_source'] = 'CPS_control'
    
    return df


def load_lalonde_data(
    include_psid: bool = True,
    include_cps: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load and combine LaLonde/NSW datasets for empirical analysis.
    
    Creates a combined dataset with:
    - NSW experimental sample (treated + randomised controls)
    - Optionally: PSID and/or CPS non-experimental controls
    
    The resulting dataset allows construction of:
    1. Experimental design (good overlap): NSW treated vs NSW controls
    2. Observational design (poor overlap): NSW treated vs PSID/CPS controls
    
    Parameters
    ----------
    include_psid : bool, default True
        Include PSID non-experimental controls.
    include_cps : bool, default False
        Include CPS non-experimental controls.
    verbose : bool, default True
        Print summary information.
        
    Returns
    -------
    pd.DataFrame
        Combined dataset with columns:
        - Y: outcome (re78, post-treatment earnings)
        - D: treatment indicator (1 = NSW participation)
        - age, education, black, hispanic, married, nodegree: covariates
        - re74, re75: pre-treatment earnings
        - data_source: indicator for data source
        - is_experimental: whether observation is from experimental sample
    """
    if verbose:
        print("Loading LaLonde/NSW data...")
    
    # Load NSW experimental sample
    nsw = load_nsw_experimental()
    if verbose:
        print(f"  NSW experimental: {len(nsw)} observations "
              f"({nsw['treat'].sum()} treated, {(1-nsw['treat']).sum():.0f} control)")
    
    datasets = [nsw]
    
    # Load PSID controls if requested
    if include_psid:
        psid = load_psid_controls()
        datasets.append(psid)
        if verbose:
            print(f"  PSID controls: {len(psid)} observations")
    
    # Load CPS controls if requested
    if include_cps:
        cps = load_cps_controls()
        datasets.append(cps)
        if verbose:
            print(f"  CPS controls: {len(cps)} observations")
    
    # Combine all datasets
    df = pd.concat(datasets, ignore_index=True)
    
    # Create standardised variable names for the analysis
    # Y: outcome (post-treatment earnings 1978)
    df['Y'] = df['re78']
    
    # D: treatment indicator
    df['D'] = df['treat'].astype(int)
    
    # Indicator for experimental vs non-experimental
    df['is_experimental'] = df['data_source'] == 'NSW_experimental'
    
    # Ensure numeric types for covariates
    covariate_cols = ['age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74', 're75']
    for col in covariate_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values in key variables
    key_vars = ['Y', 'D', 'age', 'education', 're74', 're75']
    n_before = len(df)
    df = df.dropna(subset=[v for v in key_vars if v in df.columns])
    if verbose and len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} rows with missing values")
    
    if verbose:
        print(f"  Total: {len(df)} observations")
        print(f"    Treated (NSW): {df['D'].sum()}")
        print(f"    Control (total): {(1 - df['D']).sum():.0f}")
        print(f"    Experimental sample: {df['is_experimental'].sum()}")
    
    return df


def get_experimental_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the experimental subsample (NSW treated + NSW controls).
    
    This is the "good overlap" design where treatment was randomised.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full LaLonde dataset from load_lalonde_data().
        
    Returns
    -------
    pd.DataFrame
        Experimental subsample only.
    """
    return df[df['is_experimental']].copy()


def get_observational_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the observational subsample (NSW treated + non-experimental controls).
    
    This is the "poor overlap" design that mimics typical observational studies.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full LaLonde dataset from load_lalonde_data().
        
    Returns
    -------
    pd.DataFrame
        Observational subsample (NSW treated + PSID/CPS controls).
    """
    # NSW treated + non-experimental controls
    is_nsw_treated = (df['D'] == 1)
    is_nonexp_control = (~df['is_experimental']) & (df['D'] == 0)
    
    return df[is_nsw_treated | is_nonexp_control].copy()


def get_covariate_matrix(
    df: pd.DataFrame,
    include_earnings: bool = True,
    include_squared: bool = False
) -> np.ndarray:
    """
    Extract the covariate matrix X from the LaLonde data.
    
    Parameters
    ----------
    df : pd.DataFrame
        LaLonde dataset.
    include_earnings : bool, default True
        Include pre-treatment earnings (re74, re75).
    include_squared : bool, default False
        Include squared terms for continuous variables.
        
    Returns
    -------
    np.ndarray
        Covariate matrix X of shape (n, p).
    """
    # Base covariates
    base_cols = ['age', 'education', 'black', 'hispanic', 'married', 'nodegree']
    
    if include_earnings:
        base_cols += ['re74', 're75']
    
    X = df[base_cols].values.astype(np.float64)
    
    if include_squared:
        # Add squared terms for continuous variables
        continuous_idx = [0, 1]  # age, education
        if include_earnings:
            continuous_idx += [6, 7]  # re74, re75
        
        X_sq = X[:, continuous_idx] ** 2
        X = np.hstack([X, X_sq])
    
    return X


def get_covariate_names(
    include_earnings: bool = True,
    include_squared: bool = False
) -> list:
    """
    Get the names of covariates in the X matrix.
    
    Parameters
    ----------
    include_earnings : bool, default True
        Include pre-treatment earnings.
    include_squared : bool, default False
        Include squared terms.
        
    Returns
    -------
    list
        List of covariate names.
    """
    names = ['age', 'education', 'black', 'hispanic', 'married', 'nodegree']
    
    if include_earnings:
        names += ['re74', 're75']
    
    if include_squared:
        sq_names = ['age_sq', 'education_sq']
        if include_earnings:
            sq_names += ['re74_sq', 're75_sq']
        names += sq_names
    
    return names


def summary_statistics(
    df: pd.DataFrame,
    by_treatment: bool = True
) -> pd.DataFrame:
    """
    Compute summary statistics for the LaLonde data.
    
    Parameters
    ----------
    df : pd.DataFrame
        LaLonde dataset.
    by_treatment : bool, default True
        Compute statistics separately by treatment status.
        
    Returns
    -------
    pd.DataFrame
        Summary statistics table.
    """
    vars_to_summarise = ['age', 'education', 'black', 'hispanic', 'married', 
                         'nodegree', 're74', 're75', 're78']
    vars_to_summarise = [v for v in vars_to_summarise if v in df.columns]
    
    if by_treatment:
        # Compute means by treatment group
        means = df.groupby('D')[vars_to_summarise].mean()
        means.index = ['Control', 'Treated']
        means = means.T
        
        # Add difference column
        means['Difference'] = means['Treated'] - means['Control']
        
        # Add sample sizes
        n_treat = df['D'].sum()
        n_control = len(df) - n_treat
        means.loc['N', :] = [n_control, n_treat, np.nan]
        
        return means.round(2)
    else:
        return df[vars_to_summarise].describe().T.round(2)


if __name__ == "__main__":
    # Test the data loading
    print("Testing LaLonde data loading...\n")
    
    # Load full data
    df = load_lalonde_data(include_psid=True, include_cps=False)
    
    print("\n--- Full Dataset ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\n--- Experimental Sample ---")
    exp = get_experimental_sample(df)
    print(f"N = {len(exp)}, Treated = {exp['D'].sum()}")
    print(summary_statistics(exp))
    
    print("\n--- Observational Sample ---")
    obs = get_observational_sample(df)
    print(f"N = {len(obs)}, Treated = {obs['D'].sum()}")
    print(summary_statistics(obs))
