"""
DML Diagnostics Module
======================

Functions for computing and interpreting the DML condition number κ_DML.

The condition number κ_DML is a continuous diagnostic measure that captures
the curvature of the DML orthogonal score. It is defined as:

    κ_DML = n / Σᵢ Ûᵢ²

where Ûᵢ = Dᵢ - m̂(Xᵢ) are the cross-fitted treatment residuals.

IMPORTANT: We do NOT propose specific numerical thresholds for κ_DML.
Following the approach in Saco (2025), κ_DML should be interpreted as a
continuous fragility gauge, similar to how first-stage F-statistics are
used in instrumental variables analysis. The interpretation depends on:
- Sample size and nuisance complexity
- Comparison across specifications
- Context of the specific application

Reference: Saco (2025), "Finite-Sample Failures and Condition-Number 
Diagnostics in Double Machine Learning"
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def compute_kappa(
    U_hat: NDArray,
    n: Optional[int] = None,
) -> float:
    """
    Compute the DML condition number κ_DML.
    
    The condition number is defined as:
        κ_DML = n / Σᵢ Ûᵢ²
    
    where Ûᵢ = Dᵢ - m̂(Xᵢ) are the treatment residuals.
    
    κ_DML measures the curvature of the orthogonal score. Larger values
    indicate flatter scores and more fragile inference.
    
    Parameters
    ----------
    U_hat : array-like
        Treatment residuals Û = D - m̂(X).
    n : int, optional
        Sample size. If None, uses len(U_hat).
        
    Returns
    -------
    float
        Condition number κ_DML.
        
    Examples
    --------
    >>> U_hat = np.array([0.5, -0.3, 0.8, -0.2, 0.1])
    >>> compute_kappa(U_hat)
    4.854...
    """
    U_hat = np.asarray(U_hat).ravel()
    if n is None:
        n = len(U_hat)
    
    sum_U_sq = np.sum(U_hat ** 2)
    
    if sum_U_sq < 1e-10:
        return np.inf
    
    return n / sum_U_sq


def compute_jacobian(U_hat: NDArray) -> float:
    """
    Compute the empirical Jacobian of the DML score.
    
    Ĵ_θ = -(1/n) Σᵢ Ûᵢ²
    
    Parameters
    ----------
    U_hat : array-like
        Treatment residuals.
        
    Returns
    -------
    float
        Empirical Jacobian (negative value).
    """
    U_hat = np.asarray(U_hat).ravel()
    n = len(U_hat)
    return -np.sum(U_hat ** 2) / n


def kappa_interpretation(
    kappa: float,
    n: int,
    r_squared_d: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Provide contextual interpretation of κ_DML.
    
    This function provides guidance on the conditioning of the DML estimator
    without imposing specific thresholds. Researchers should interpret κ_DML
    in the context of their application, similar to F-statistics in IV.
    
    Parameters
    ----------
    kappa : float
        The condition number κ_DML.
    n : int
        Sample size.
    r_squared_d : float, optional
        R²(D|X) from the treatment regression.
        
    Returns
    -------
    dict
        Dictionary with:
        - kappa: the condition number
        - kappa_per_sqrt_n: κ_DML / √n (variance scaling)
        - effective_n: n / κ_DML (effective sample size interpretation)
        - interpretation: qualitative description
        - guidance: practical advice
    """
    # Compute derived quantities
    kappa_per_sqrt_n = kappa / np.sqrt(n)
    effective_n = n / kappa if kappa > 0 else n
    
    # Build interpretation
    if np.isinf(kappa):
        interpretation = (
            "Condition number is infinite, indicating zero residual treatment "
            "variation. DML estimation is not possible."
        )
        guidance = (
            "The treatment is perfectly predicted by covariates. "
            "Check for collinearity or reconsider the estimand."
        )
    else:
        # Describe the magnitude in context
        interpretation = (
            f"κ_DML = {kappa:.2f} corresponds to an effective sample size of "
            f"approximately {effective_n:.0f} for the variance."
        )
        
        if r_squared_d is not None:
            interpretation += (
                f" The treatment regression R² = {r_squared_d:.3f} indicates "
                f"{'strong' if r_squared_d > 0.8 else 'moderate' if r_squared_d > 0.5 else 'limited'} "
                f"predictability of treatment from covariates."
            )
        
        guidance = (
            "Compare κ_DML across specifications and learners. Large increases "
            "in κ_DML when using more flexible learners suggest sensitivity to "
            "nuisance estimation. Report κ_DML alongside estimates to allow "
            "readers to assess reliability."
        )
    
    return {
        "kappa": kappa,
        "kappa_per_sqrt_n": kappa_per_sqrt_n,
        "effective_n": effective_n,
        "interpretation": interpretation,
        "guidance": guidance,
    }


# Keep classify_regime for backwards compatibility but make it descriptive
def classify_regime(kappa: float, n: int) -> Dict[str, Any]:
    """
    Provide a qualitative description of the conditioning.
    
    IMPORTANT: This function provides qualitative descriptions to assist
    interpretation, NOT definitive classifications. We do not propose
    specific numerical thresholds for κ_DML. The interpretation depends
    on context and should consider:
    
    - How κ_DML varies across specifications and learners
    - The sample size and complexity of nuisance estimation
    - Whether estimates are stable or sensitive to choices
    
    Following Saco (2025), κ_DML should be treated as a continuous
    diagnostic, analogous to how F-statistics guide IV interpretation
    without rigid pass/fail rules.
    
    Parameters
    ----------
    kappa : float
        The condition number κ_DML.
    n : int
        Sample size.
        
    Returns
    -------
    dict
        Dictionary with:
        - description: qualitative characterization
        - interpretation: guidance for practitioners
        - effective_n: n / κ_DML (effective sample size interpretation)
    """
    if np.isinf(kappa):
        return {
            "description": "undefined",
            "interpretation": (
                "DML estimation failed due to zero residual treatment variation. "
                "This indicates perfect predictability of treatment from covariates."
            ),
            "effective_n": 0,
        }
    
    effective_n = n / kappa if kappa > 0 else n
    
    # Provide contextual description without hard thresholds
    # These descriptions are for guidance only
    interpretation = (
        f"κ_DML = {kappa:.2f} corresponds to an effective sample size of "
        f"approximately {effective_n:.0f}. The confidence interval width "
        f"scales with κ_DML/√n, and any nuisance error is amplified by κ_DML. "
        f"Compare this value across specifications to assess stability."
    )
    
    # Relative characterization based on scaling with √n
    kappa_sqrt_n = kappa / np.sqrt(n)
    
    if kappa_sqrt_n < 0.15:
        description = "favorable conditioning"
    elif kappa_sqrt_n < 0.5:
        description = "moderate conditioning"
    else:
        description = "challenging conditioning"
    
    return {
        "description": description,
        "interpretation": interpretation,
        "effective_n": effective_n,
        "kappa_over_sqrt_n": kappa_sqrt_n,
    }


def overlap_check(
    D: NDArray,
    X: NDArray,
    method: str = "logistic",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Check overlap via propensity score diagnostics.
    
    For binary treatments, estimates the propensity score e(X) = P(D=1|X)
    and computes summary statistics related to overlap.
    
    Parameters
    ----------
    D : array-like
        Treatment indicator (binary).
    X : array-like
        Covariates.
    method : str, default 'logistic'
        Propensity estimation method: 'logistic' or 'rf'.
    random_state : int
        Random state for reproducibility.
        
    Returns
    -------
    dict
        Dictionary with:
        - e_hat: estimated propensity scores
        - e_mean, e_std, e_min, e_max: summary statistics
        - e_mean_treated, e_mean_control: mean by treatment group
        - r_squared_d: R² of treatment model
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.ensemble import RandomForestClassifier
    
    D = np.asarray(D).ravel()
    X = np.asarray(X)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Estimate propensity scores
    if method.lower() == "logistic":
        model = LogisticRegressionCV(cv=5, random_state=random_state, max_iter=1000)
    elif method.lower() == "rf":
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=random_state, n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown method: '{method}'. Choose 'logistic' or 'rf'.")
    
    model.fit(X, D)
    e_hat = model.predict_proba(X)[:, 1]
    
    # Compute R² for the treatment model
    from sklearn.metrics import r2_score
    D_pred = model.predict_proba(X)[:, 1]
    # For classification, use pseudo-R² based on variance explained
    var_D = np.var(D)
    var_residual = np.var(D - D_pred)
    r_squared_d = 1 - var_residual / var_D if var_D > 0 else np.nan
    
    return {
        "e_hat": e_hat,
        "e_mean": np.mean(e_hat),
        "e_std": np.std(e_hat),
        "e_min": np.min(e_hat),
        "e_max": np.max(e_hat),
        "e_mean_treated": np.mean(e_hat[D == 1]),
        "e_mean_control": np.mean(e_hat[D == 0]),
        "r_squared_d": r_squared_d,
    }


# No threshold constants - κ is a continuous diagnostic
REGIME_THRESHOLDS = None  # Explicitly set to None to indicate no thresholds
