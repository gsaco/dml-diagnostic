"""
Core simulation functions for DML finite-sample conditioning study.

This module implements the Partially Linear Regression (PLR) model and 
Double Machine Learning estimator for studying the condition number κ_DML.

The theoretical expansion from the paper is:
    θ̂ − θ₀ = κ_DML · (Sₙ + Bₙ) + Rₙ
    
where:
    - κ_DML = n / Σᵢ Ûᵢ² = 1/|Ĵ_θ| is the condition number
    - Sₙ = (1/n) Σᵢ ψ(Wᵢ; θ₀, η₀) is the leading score term
    - Bₙ is the nuisance estimation bias
    - Rₙ is a remainder term

References:
    - Robinson (1988), Econometrica
    - Chernozhukov et al. (2018), Econometrics Journal
    - Chernozhukov, Newey & Singh (2023), Biometrika
    - Bach et al. (2022), Journal of Statistical Software
"""

from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold


# =============================================================================
# DGP COEFFICIENTS
# =============================================================================
# Following PLR simulation practice in Chernozhukov et al. (2018) and 
# Bach et al. (2022, DoubleML), we use decaying coefficient patterns.

def get_beta_D(p: int = 10) -> NDArray:
    """
    Coefficient vector for treatment equation D = X'β_D + U.
    
    Uses a decaying pattern on the first 5 covariates to avoid dominance
    by a single covariate, following simulation practice in PLR/DML 
    literature (Chernozhukov et al. 2018; Bach et al. 2022).
    """
    beta_D = np.zeros(p)
    beta_D[:5] = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    return beta_D


def get_gamma(p: int = 10) -> NDArray:
    """
    Coefficient vector for nuisance function g₀(X) = γ' sin(X).
    
    Uses geometric decay (factor 0.5) to create a smooth nonlinear 
    nuisance function that requires ML to estimate but remains tractable.
    Similar to semiparametric designs in Robinson (1988) and 
    Chernozhukov et al. (2018).
    """
    gamma = np.zeros(p)
    gamma[:5] = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    return gamma


# =============================================================================
# OVERLAP CALIBRATION
# =============================================================================

def compute_theoretical_r2(
    rho: float = 0.5,
    p: int = 10,
    sigma_U_sq: float = 0.25,
) -> float:
    """
    Compute theoretical R²(D|X) for the PLR design.
    
    For D = X'β_D + U with X ~ N(0, Σ(ρ)) and U ~ N(0, σ_U²):
        Var(D) = β_D' Σ β_D + σ_U²
        R²(D|X) = β_D' Σ β_D / Var(D)
    
    Parameters
    ----------
    rho : float
        AR(1) correlation parameter for Σ.
    p : int
        Dimension of covariates.
    sigma_U_sq : float
        Variance of residual U.
    
    Returns
    -------
    r2 : float
        Theoretical R²(D|X).
    """
    # AR(1) covariance matrix
    idx = np.arange(p)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    
    # Treatment coefficient
    beta_D = get_beta_D(p)
    
    # Var(X'β_D) = β_D' Σ β_D
    var_Xbeta = beta_D @ Sigma @ beta_D
    
    # R²(D|X) = Var(X'β_D) / [Var(X'β_D) + σ_U²]
    r2 = var_Xbeta / (var_Xbeta + sigma_U_sq)
    
    return r2


def calibrate_sigma_U_for_r2(
    target_r2: float,
    rho: float = 0.5,
    p: int = 10,
) -> float:
    """
    Calibrate σ_U² to achieve a target R²(D|X).
    
    Given target R² and design parameters, solve for σ_U²:
        R² = Var(X'β_D) / [Var(X'β_D) + σ_U²]
        σ_U² = Var(X'β_D) * (1 - R²) / R²
    
    Parameters
    ----------
    target_r2 : float
        Target R²(D|X), must be in (0, 1).
    rho : float
        AR(1) correlation parameter.
    p : int
        Dimension of covariates.
    
    Returns
    -------
    sigma_U_sq : float
        Variance of U to achieve target R².
    """
    if not 0 < target_r2 < 1:
        raise ValueError("target_r2 must be in (0, 1)")
    
    # Compute Var(X'β_D)
    idx = np.arange(p)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    beta_D = get_beta_D(p)
    var_Xbeta = beta_D @ Sigma @ beta_D
    
    # Solve for σ_U²
    sigma_U_sq = var_Xbeta * (1 - target_r2) / target_r2
    
    return sigma_U_sq


# Overlap levels calibrated to target R²(D|X) values
# Following Zimmert (2018) and Naghi (2021) who calibrate designs using R²
OVERLAP_CONFIG: Dict[str, Dict] = {
    "high": {
        "target_r2": 0.75,      # Good overlap: 75% of D explained by X
        "description": "High overlap (R² ≈ 0.75)",
    },
    "moderate": {
        "target_r2": 0.90,      # Moderate overlap: 90% explained
        "description": "Moderate overlap (R² ≈ 0.90)",
    },
    "low": {
        "target_r2": 0.97,      # Poor overlap: 97% explained, little residual variation
        "description": "Low overlap (R² ≈ 0.97)",
    },
}


def get_sigma_U_sq(
    overlap: Literal["high", "moderate", "low"],
    rho: float = 0.5,
    p: int = 10,
) -> float:
    """
    Get calibrated σ_U² for a given overlap level and design.
    
    The σ_U² is calibrated to achieve the target R²(D|X) for each
    overlap level, accounting for the correlation structure ρ.
    """
    target_r2 = OVERLAP_CONFIG[overlap]["target_r2"]
    return calibrate_sigma_U_for_r2(target_r2, rho, p)


# =============================================================================
# DGP CONFIGURATIONS
# =============================================================================

# Pre-defined DGP configurations for the simulation study
# Designed to span well-conditioned to severely ill-conditioned regimes
DGP_CONFIGS: List[Dict] = [
    # Group A: Well-conditioned designs (high overlap → small κ_DML)
    {"dgp_id": "A1", "n": 500,  "rho": 0.0, "overlap": "high",
     "description": "Baseline: independent X, high overlap"},
    {"dgp_id": "A2", "n": 500,  "rho": 0.5, "overlap": "high",
     "description": "Moderate correlation, high overlap"},
    {"dgp_id": "A3", "n": 2000, "rho": 0.5, "overlap": "high",
     "description": "Large n, moderate correlation, high overlap"},
    
    # Group B: Moderately ill-conditioned designs
    {"dgp_id": "B1", "n": 500,  "rho": 0.0, "overlap": "moderate",
     "description": "Independent X, moderate overlap"},
    {"dgp_id": "B2", "n": 500,  "rho": 0.5, "overlap": "moderate",
     "description": "Moderate correlation and overlap"},
    {"dgp_id": "B3", "n": 2000, "rho": 0.5, "overlap": "moderate",
     "description": "Large n, moderate correlation and overlap"},
    
    # Group C: Severely ill-conditioned designs (low overlap → large κ_DML)
    {"dgp_id": "C1", "n": 500,  "rho": 0.5, "overlap": "low",
     "description": "Moderate correlation, low overlap"},
    {"dgp_id": "C2", "n": 2000, "rho": 0.5, "overlap": "low",
     "description": "Large n, moderate correlation, low overlap"},
    {"dgp_id": "C3", "n": 2000, "rho": 0.9, "overlap": "low",
     "description": "Large n, high correlation, low overlap (worst case)"},
]


# =============================================================================
# DATA GENERATING PROCESS
# =============================================================================

def generate_plr_data(
    n: int,
    p: int = 10,
    rho: float = 0.5,
    overlap: Literal["high", "moderate", "low"] = "moderate",
    theta0: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray, Dict]:
    """
    Generate data from the Partially Linear Regression (PLR) model.
    
    Model (cf. Robinson 1988; Chernozhukov et al. 2018):
        Y = D · θ₀ + g₀(X) + ε,    ε ~ N(0, 1)
        D = X'β_D + U,              U ~ N(0, σ_U²)
        X ~ N(0, Σ(ρ)),            Σ_jk = ρ^|j-k| (AR(1)/Toeplitz)
    
    where:
        - g₀(X) = γ' sin(X) is a nonlinear nuisance function
        - β_D has decaying coefficients on first 5 covariates
        - σ_U² is calibrated to achieve target R²(D|X) for each overlap level
    
    Parameters
    ----------
    n : int
        Sample size.
    p : int, default=10
        Dimension of covariates X.
    rho : float, default=0.5
        Correlation parameter for AR(1) covariate structure.
    overlap : {"high", "moderate", "low"}, default="moderate"
        Overlap level, calibrated via R²(D|X):
        - "high": R²(D|X) ≈ 0.75 (well-conditioned)
        - "moderate": R²(D|X) ≈ 0.90 (moderately ill-conditioned)
        - "low": R²(D|X) ≈ 0.97 (severely ill-conditioned)
    theta0 : float, default=1.0
        True treatment effect parameter.
    seed : int or None, default=None
        Random seed for reproducibility.
    
    Returns
    -------
    Y : ndarray of shape (n,)
        Outcome variable.
    D : ndarray of shape (n,)
        Treatment variable.
    X : ndarray of shape (n, p)
        Covariate matrix.
    info : dict
        Design information including sigma_U_sq, theoretical R², sample R².
    """
    rng = np.random.default_rng(seed)
    
    # Get calibrated σ_U² for this design
    sigma_U_sq = get_sigma_U_sq(overlap, rho, p)
    sigma_U = np.sqrt(sigma_U_sq)
    
    # Theoretical R²(D|X) for this design
    theoretical_r2 = compute_theoretical_r2(rho, p, sigma_U_sq)
    
    # AR(1) covariance matrix: Σ_jk = ρ^|j-k|
    idx = np.arange(p)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    
    # Generate X ~ N(0, Σ)
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    
    # Coefficients
    beta_D = get_beta_D(p)
    gamma = get_gamma(p)
    
    # Generate treatment: D = X'β_D + U
    U = rng.normal(0, sigma_U, size=n)
    m0_X = X @ beta_D  # True propensity E[D|X]
    D = m0_X + U
    
    # Generate nuisance function: g₀(X) = γ' sin(X)
    g0_X = np.sin(X) @ gamma
    
    # Generate outcome: Y = D·θ₀ + g₀(X) + ε
    eps = rng.normal(0, 1.0, size=n)
    Y = D * theta0 + g0_X + eps
    
    # Compute sample R²(D|X) for diagnostics
    sample_r2 = 1 - np.var(U) / np.var(D) if np.var(D) > 0 else 0.0
    
    info = {
        "sigma_U_sq": sigma_U_sq,
        "theoretical_r2": theoretical_r2,
        "sample_r2": np.var(m0_X) / np.var(D) if np.var(D) > 0 else 0.0,
        "var_D": np.var(D),
        "var_U": np.var(U),
    }
    
    return Y, D, X, info


# =============================================================================
# DML ESTIMATOR
# =============================================================================

def dml_plr_estimator(
    Y: NDArray,
    D: NDArray,
    X: NDArray,
    K: int = 5,
    learner: Literal["rf", "lasso"] = "rf",
    random_state: int = 42,
) -> Dict:
    """
    Double Machine Learning estimator for the Partially Linear Regression model.
    
    Implements the partialling-out/orthogonal score approach with K-fold 
    cross-fitting (Chernozhukov et al. 2018).
    
    The orthogonal score is:
        ψ(W; θ, η) = (D − m(X)) · (Y − g(X) − θ·(D − m(X)))
    
    where η = (m, g) are nuisance functions:
        m₀(X) = E[D|X],  g₀(X) = E[Y|X] - θ₀·E[D|X]
    
    Algorithm:
        1. Split data into K folds
        2. For each fold k, fit on complement I^c_k:
           - m̂(X) ≈ E[D|X]
           - ℓ̂(X) ≈ E[Y|X]
        3. Compute residuals (out-of-fold):
           - Û_i = D_i − m̂(X_i)  (residualized treatment)
           - V̂_i = Y_i − ℓ̂(X_i)  (residualized outcome)
        4. Compute DML estimate:
           θ̂ = (Σᵢ Ûᵢ V̂ᵢ) / (Σᵢ Ûᵢ²)
        5. Compute condition number:
           κ_DML = n / Σᵢ Ûᵢ² = 1/|Ĵ_θ|
    
    Parameters
    ----------
    Y : ndarray of shape (n,)
        Outcome variable.
    D : ndarray of shape (n,)
        Treatment variable.
    X : ndarray of shape (n, p)
        Covariate matrix.
    K : int, default=5
        Number of folds for cross-fitting.
    learner : {"rf", "lasso"}, default="rf"
        ML method for nuisance estimation.
    random_state : int, default=42
        Random state for reproducibility.
    
    Returns
    -------
    results : dict
        Contains theta_hat, se, ci_lower, ci_upper, ci_length, kappa_dml, J_hat.
    
    Notes
    -----
    The condition number κ_DML = n / Σᵢ Ûᵢ² measures ill-conditioning.
    From the paper's linearization:
        θ̂ − θ₀ = κ_DML · (Sₙ + Bₙ) + Rₙ
    Large κ_DML amplifies both the score variance (Sₙ) and nuisance bias (Bₙ),
    leading to poor coverage and slow CI shrinkage.
    """
    n = len(Y)
    
    # Initialize arrays for out-of-fold predictions
    m_hat = np.zeros(n)    # Predictions for E[D|X]
    ell_hat = np.zeros(n)  # Predictions for E[Y|X]
    
    # K-fold cross-fitting
    kf = KFold(n_splits=K, shuffle=True, random_state=random_state)
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        D_train = D[train_idx]
        Y_train = Y[train_idx]
        
        # Fit ML models for nuisance functions
        if learner == "rf":
            # Random Forest with conservative hyperparameters
            # (Following Bach et al. 2022, DoubleML defaults)
            model_m = RandomForestRegressor(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=-1
            )
            model_ell = RandomForestRegressor(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=5,
                random_state=random_state,
                n_jobs=-1
            )
        elif learner == "lasso":
            model_m = LassoCV(cv=5, random_state=random_state, max_iter=5000)
            model_ell = LassoCV(cv=5, random_state=random_state, max_iter=5000)
        else:
            raise ValueError(f"Unknown learner: {learner}")
        
        # Fit m̂(X) = E[D|X]
        model_m.fit(X_train, D_train)
        m_hat[test_idx] = model_m.predict(X_test)
        
        # Fit ℓ̂(X) = E[Y|X]
        model_ell.fit(X_train, Y_train)
        ell_hat[test_idx] = model_ell.predict(X_test)
    
    # Compute residuals
    # Û = D − m̂(X): residualized treatment
    U_hat = D - m_hat
    # V̂ = Y − ℓ̂(X): residualized outcome  
    V_hat = Y - ell_hat
    
    # DML estimate: θ̂ = Σ(Û·V̂) / Σ(Û²)
    sum_U_sq = np.sum(U_hat ** 2)
    sum_UV = np.sum(U_hat * V_hat)
    
    # Handle near-zero denominator
    if sum_U_sq < 1e-10:
        return {
            "theta_hat": np.nan, "se": np.nan,
            "ci_lower": np.nan, "ci_upper": np.nan, "ci_length": np.nan,
            "kappa_dml": np.inf, "J_hat": 0.0,
        }
    
    theta_hat = sum_UV / sum_U_sq
    
    # Empirical Jacobian: Ĵ_θ = −(1/n) Σ Û²
    # This is the derivative of the score with respect to θ
    J_hat = -sum_U_sq / n
    
    # Condition number: κ_DML = 1/|Ĵ_θ| = n / Σ Û²
    kappa_dml = n / sum_U_sq
    
    # Score values: ψᵢ = Ûᵢ · (V̂ᵢ − θ̂ · Ûᵢ)
    psi = U_hat * (V_hat - theta_hat * U_hat)
    
    # Variance estimation (sandwich formula)
    sigma_sq = np.mean(psi ** 2)
    var_theta = sigma_sq / (n * J_hat ** 2)
    se = np.sqrt(var_theta)
    
    # 95% confidence interval
    z_alpha = 1.96
    ci_lower = theta_hat - z_alpha * se
    ci_upper = theta_hat + z_alpha * se
    ci_length = ci_upper - ci_lower
    
    return {
        "theta_hat": theta_hat,
        "se": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_length": ci_length,
        "kappa_dml": kappa_dml,
        "J_hat": J_hat,
    }


# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

def run_single_replication(
    dgp_id: str,
    n: int,
    p: int = 10,
    rho: float = 0.5,
    overlap: Literal["high", "moderate", "low"] = "moderate",
    theta0: float = 1.0,
    K: int = 5,
    learner: Literal["rf", "lasso"] = "rf",
    seed: int = 0,
) -> Dict:
    """
    Run a single Monte Carlo replication.
    
    Returns a dict with DGP identifiers, estimation results, and diagnostics.
    """
    # Generate data
    Y, D, X, dgp_info = generate_plr_data(
        n=n, p=p, rho=rho, overlap=overlap, theta0=theta0, seed=seed
    )
    
    # Fit DML estimator
    dml_result = dml_plr_estimator(
        Y=Y, D=D, X=X, K=K, learner=learner, random_state=seed
    )
    
    # Compute coverage and error metrics
    theta_hat = dml_result["theta_hat"]
    ci_lower = dml_result["ci_lower"]
    ci_upper = dml_result["ci_upper"]
    
    if np.isnan(theta_hat):
        coverage = np.nan
        bias = np.nan
        squared_error = np.nan
    else:
        coverage = 1 if (ci_lower <= theta0 <= ci_upper) else 0
        bias = theta_hat - theta0
        squared_error = bias ** 2
    
    return {
        # DGP identifiers
        "dgp_id": dgp_id,
        "n": n,
        "rho": rho,
        "overlap": overlap,
        "sigma_U_sq": dgp_info["sigma_U_sq"],
        "sample_r2": dgp_info["sample_r2"],
        "replication_seed": seed,
        # Estimation results
        "theta_hat": theta_hat,
        "se": dml_result["se"],
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_length": dml_result["ci_length"],
        # Condition number (key diagnostic)
        "kappa_dml": dml_result["kappa_dml"],
        # Performance metrics
        "coverage": coverage,
        "bias": bias,
        "squared_error": squared_error,
    }


def run_simulation_grid(
    dgp_configs: List[Dict],
    n_reps: int = 500,
    p: int = 10,
    theta0: float = 1.0,
    K: int = 5,
    learner: Literal["rf", "lasso"] = "rf",
    base_seed: int = 2024,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulations across a grid of DGP configurations.
    
    Parameters
    ----------
    dgp_configs : list of dict
        List of DGP configurations with keys: dgp_id, n, rho, overlap.
    n_reps : int, default=500
        Number of Monte Carlo replications per DGP.
    p : int, default=10
        Dimension of covariates.
    theta0 : float, default=1.0
        True treatment effect.
    K : int, default=5
        Number of cross-fitting folds.
    learner : {"rf", "lasso"}, default="rf"
        ML method for nuisance estimation.
    base_seed : int, default=2024
        Base random seed for reproducibility.
    verbose : bool, default=True
        Print progress.
    
    Returns
    -------
    results_df : pd.DataFrame
        One row per replication with all diagnostics.
    """
    results = []
    n_dgps = len(dgp_configs)
    
    for dgp_idx, config in enumerate(dgp_configs):
        dgp_id = config["dgp_id"]
        n = config["n"]
        rho = config["rho"]
        overlap = config["overlap"]
        
        if verbose:
            desc = config.get("description", "")
            print(f"[{dgp_idx + 1}/{n_dgps}] DGP {dgp_id}: n={n}, ρ={rho}, {overlap} overlap")
            if desc:
                print(f"    {desc}")
        
        for rep in range(n_reps):
            seed = base_seed + dgp_idx * 10000 + rep
            
            rep_result = run_single_replication(
                dgp_id=dgp_id, n=n, p=p, rho=rho, overlap=overlap,
                theta0=theta0, K=K, learner=learner, seed=seed,
            )
            rep_result["replication"] = rep
            results.append(rep_result)
            
            if verbose and (rep + 1) % 100 == 0:
                print(f"    Completed {rep + 1}/{n_reps} replications")
    
    return pd.DataFrame(results)


def compute_summary_statistics(
    results_df: pd.DataFrame,
    theta0: float = 1.0,
) -> pd.DataFrame:
    """
    Compute summary statistics by DGP from Monte Carlo results.
    
    Returns DataFrame with coverage, RMSE, mean κ_DML, etc. for each DGP.
    """
    grouped = results_df.groupby(["dgp_id", "n", "rho", "overlap"])
    
    summary = grouped.agg(
        n_reps=("replication", "count"),
        sigma_U_sq=("sigma_U_sq", "first"),
        mean_sample_r2=("sample_r2", "mean"),
        mean_theta=("theta_hat", "mean"),
        sd_theta=("theta_hat", "std"),
        mean_kappa=("kappa_dml", "mean"),
        sd_kappa=("kappa_dml", "std"),
        coverage=("coverage", "mean"),
        mean_ci_length=("ci_length", "mean"),
        mean_se=("se", "mean"),
    ).reset_index()
    
    # Compute bias and RMSE
    summary["mean_bias"] = summary["mean_theta"] - theta0
    rmse = grouped["squared_error"].mean().apply(np.sqrt).reset_index(name="rmse")
    summary = summary.merge(rmse, on=["dgp_id", "n", "rho", "overlap"])
    
    # Coverage as percentage
    summary["coverage_pct"] = summary["coverage"] * 100
    
    # Reorder columns
    cols = ["dgp_id", "n", "rho", "overlap", "sigma_U_sq", "mean_sample_r2",
            "mean_kappa", "sd_kappa", "coverage_pct", "mean_ci_length", 
            "mean_bias", "rmse", "n_reps"]
    summary = summary[[c for c in cols if c in summary.columns]]
    
    return summary


def format_summary_for_latex(
    summary_df: pd.DataFrame,
    float_format: str = "%.3f",
) -> str:
    """
    Format summary table as LaTeX.
    """
    # Select and rename columns for paper
    cols = ["dgp_id", "n", "rho", "overlap", "mean_sample_r2", 
            "mean_kappa", "coverage_pct", "mean_ci_length", "rmse"]
    
    df = summary_df[[c for c in cols if c in summary_df.columns]].copy()
    
    # Round
    for col in ["mean_sample_r2", "mean_kappa", "mean_ci_length", "rmse"]:
        if col in df.columns:
            df[col] = df[col].round(3 if col != "coverage_pct" else 1)
    if "coverage_pct" in df.columns:
        df["coverage_pct"] = df["coverage_pct"].round(1)
    
    rename = {
        "dgp_id": "DGP",
        "n": "$n$",
        "rho": r"$\rho$",
        "overlap": "Overlap",
        "mean_sample_r2": r"$R^2(D|X)$",
        "mean_kappa": r"$\bar{\kappa}_{\mathrm{DML}}$",
        "coverage_pct": r"Coverage (\%)",
        "mean_ci_length": "CI Length",
        "rmse": "RMSE",
    }
    df = df.rename(columns=rename)
    
    return df.to_latex(index=False, escape=False, float_format=float_format)
