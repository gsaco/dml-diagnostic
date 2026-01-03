"""
Double Machine Learning Estimator Implementation.

This module implements the DML estimator with cross-fitting for the Partially
Linear Regression model, as described in "Ill-Conditioned Orthogonal Scores
in Double Machine Learning."

Theoretical Foundation
----------------------
The DML estimator uses the Neyman-orthogonal score for PLR (Equation 2):
    ψ(W; θ, η) = (Y - ℓ(X) - θ(D - m(X))) · (D - m(X))

Setting E[ψ] = 0 yields the partialling-out estimator:
    θ̂ = Σ V̂ᵢ(Yᵢ - ℓ̂(Xᵢ)) / Σ V̂ᵢ²

where V̂ᵢ = Dᵢ - m̂(Xᵢ) is the residualized treatment.

Condition Number κ (Definition 3.5)
-----------------------------------
The standardized condition number measures identification strength:
    κ = σ²_D / σ²_V = 1 / (1 - R²(D|X))

Per Theorem 3.8, κ enters the exact decomposition:
    θ̂ - θ₀ = κ̂(S'_n + B'_n)

where S'_n is the scaled sampling term and B'_n contains nuisance bias.
Per Theorem 3.11, this implies the stochastic-order bound:
    θ̂ - θ₀ = O_P(√(κ/n) + κ·Rem_n)

Implementation Notes
--------------------
- Uses K-fold cross-fitting to avoid overfitting bias (default K=5)
- Supports repeated cross-fitting with median aggregation for stability
- Computes structural κ from true nuisance values when available
- Returns comprehensive diagnostics for mechanism validation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold

if TYPE_CHECKING:
    from src.dgp import DGP
    from src.learners import OracleLearner


# =============================================================================
# CONSTANTS
# =============================================================================

THETA0: float = 1.0          # True treatment effect θ₀
K_FOLDS: int = 5             # Number of cross-fitting folds
N_REPEATS: int = 5           # Number of repeated cross-fitting iterations
Z_ALPHA: float = 1.96        # Critical value for 95% CI (α = 0.05)


# =============================================================================
# DML RESULT CONTAINER
# =============================================================================

@dataclass
class DMLResult:
    """
    Container for DML estimation results with diagnostic information.
    
    Attributes
    ----------
    theta_hat : float
        Estimated treatment effect θ̂.
    se : float
        Heteroskedasticity-robust standard error.
    ci_lower, ci_upper : float
        95% confidence interval bounds.
    kappa : float
        Estimated condition number κ̂ = 1/(1 - R̂²(D|X)).
        Computed from learner-estimated residuals V̂ = D - m̂(X).
    structural_kappa : float
        Structural κ computed from true residuals V = D - m₀(X).
        Essential for Corrupted Oracle analysis where learner residuals
        are contaminated by injected bias. Equals kappa when true values
        are not provided.
    bias : float
        Estimated bias θ̂ - θ₀.
    nuisance_mse_m, nuisance_mse_l : float
        Mean squared error of m̂(X) and ℓ̂(X) relative to truth.
    sample_r2 : float
        Sample R²(D|X) = 1 - 1/κ.
    n : int
        Sample size.
    
    Notes
    -----
    The distinction between kappa and structural_kappa is critical for
    the Corrupted Oracle experiments (Section 5). When learners are
    biased, their residuals are contaminated, inflating κ̂. The structural
    κ computed from true m₀(X) remains stable across bias levels and
    correctly captures the DGP's identification strength.
    """
    theta_hat: float
    se: float
    ci_lower: float
    ci_upper: float
    kappa: float
    structural_kappa: float
    bias: float
    nuisance_mse_m: float
    nuisance_mse_l: float
    sample_r2: float
    n: int
    
    @property
    def rmse_m(self) -> float:
        """Root mean squared error of m̂(X)."""
        import numpy as np
        return np.sqrt(self.nuisance_mse_m) if not np.isnan(self.nuisance_mse_m) else np.nan
    
    @property
    def rmse_l(self) -> float:
        """Root mean squared error of ℓ̂(X)."""
        import numpy as np
        return np.sqrt(self.nuisance_mse_l) if not np.isnan(self.nuisance_mse_l) else np.nan
    
    @property
    def ci_length(self) -> float:
        """Length of 95% confidence interval."""
        return self.ci_upper - self.ci_lower
    
    def covers(self, theta0: float = THETA0) -> bool:
        """Check if confidence interval covers true parameter value."""
        return self.ci_lower <= theta0 <= self.ci_upper
    
    @property
    def conditioning_regime(self) -> str:
        """
        Conditioning regime classification per Definition 4.1.
        
        Returns
        -------
        regime : str
            - "well-conditioned": κ < 5
            - "moderately ill-conditioned": 5 ≤ κ ≤ 20
            - "severely ill-conditioned": κ > 20
        """
        if self.kappa < 5:
            return "well-conditioned"
        elif self.kappa <= 20:
            return "moderately ill-conditioned"
        else:
            return "severely ill-conditioned"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'theta_hat': self.theta_hat,
            'se': self.se,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'ci_length': self.ci_length,
            'kappa': self.kappa,
            'structural_kappa': self.structural_kappa,
            'bias': self.bias,
            'nuisance_mse_m': self.nuisance_mse_m,
            'nuisance_mse_l': self.nuisance_mse_l,
            'rmse_m': self.rmse_m,
            'rmse_l': self.rmse_l,
            'sample_r2': self.sample_r2,
            'n': self.n,
            'conditioning_regime': self.conditioning_regime,
        }


# =============================================================================
# DML ESTIMATOR CLASS
# =============================================================================

class DMLEstimator:
    """
    Double Machine Learning Estimator with K-fold cross-fitting.
    
    Implements the partialling-out estimator for the PLR model with
    cross-fitting to eliminate overfitting bias in nuisance estimation.
    
    Algorithm
    ---------
    1. Split sample into K folds
    2. For each fold k:
       a. Fit m̂(X) = E[D|X] on remaining folds, predict on fold k
       b. Fit ℓ̂(X) = E[Y|X] on remaining folds, predict on fold k
    3. Compute residuals: V̂ = D - m̂(X), Û = Y - ℓ̂(X)
    4. Estimate: θ̂ = Σ V̂·Û / Σ V̂²
    5. Variance: σ̂² = n⁻¹ Σ V̂²·ε̂² / (n⁻¹ Σ V̂²)² where ε̂ = Û - θ̂V̂
    
    Parameters
    ----------
    learner_m : BaseEstimator
        Scikit-learn compatible learner for m(X) = E[D|X].
    learner_l : BaseEstimator
        Scikit-learn compatible learner for ℓ(X) = E[Y|X].
    K : int, default 5
        Number of cross-fitting folds.
    n_repeats : int, default 1
        Number of repeated cross-fitting iterations. If > 1, uses
        median aggregation for robustness to sample splitting.
    theta0 : float, default 1.0
        True treatment effect (for bias calculation in simulations).
    random_state : int, default 42
        Random state for reproducibility.
    
    Attributes
    ----------
    result_ : DMLResult or None
        Estimation results after calling fit().
    """
    
    def __init__(
        self,
        learner_m: BaseEstimator,
        learner_l: BaseEstimator,
        K: int = K_FOLDS,
        n_repeats: int = 1,
        theta0: float = THETA0,
        random_state: int = 42,
    ) -> None:
        self.learner_m = learner_m
        self.learner_l = learner_l
        self.K = K
        self.n_repeats = n_repeats
        self.theta0 = theta0
        self.random_state = random_state
        self.result_: Optional[DMLResult] = None
    
    def _single_crossfit(
        self,
        Y: NDArray,
        D: NDArray,
        X: NDArray,
        m0_X: Optional[NDArray],
        ell0_X: Optional[NDArray],
        split_seed: int,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Perform a single cross-fitting iteration.
        
        Returns
        -------
        theta_hat : float
            Point estimate.
        var_hat : float
            Variance estimate.
        kappa : float
            Condition number from estimated residuals.
        structural_kappa : float
            Condition number from true residuals (if available).
        mse_m, mse_l : float
            Nuisance MSEs (if true values available).
        """
        n = len(Y)
        m_hat = np.zeros(n)
        l_hat = np.zeros(n)
        
        kf = KFold(n_splits=self.K, shuffle=True, random_state=split_seed)
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            D_train, Y_train = D[train_idx], Y[train_idx]
            
            # Clone learners for each fold
            model_m = clone(self.learner_m)
            model_l = clone(self.learner_l)
            
            # Fit m̂(X) = E[D|X]
            model_m.fit(X_train, D_train)
            m_hat[test_idx] = model_m.predict(X_test)
            
            # Fit ℓ̂(X) = E[Y|X]
            model_l.fit(X_train, Y_train)
            l_hat[test_idx] = model_l.predict(X_test)
        
        # Compute residuals
        V_hat = D - m_hat  # Residualized treatment
        U_hat = Y - l_hat  # Residualized outcome
        
        # DML estimate: θ̂ = Σ(V̂·Û) / Σ V̂²
        sum_V_sq = np.sum(V_hat ** 2)
        
        if sum_V_sq < 1e-12:
            return np.nan, np.nan, np.inf, np.inf, np.nan, np.nan
        
        theta_hat = np.sum(V_hat * U_hat) / sum_V_sq
        
        # Final residuals for variance estimation
        eps_hat = U_hat - theta_hat * V_hat
        
        # Heteroskedasticity-robust variance (sandwich form)
        var_component = np.mean(V_hat ** 2 * eps_hat ** 2)
        var_hat = (n / sum_V_sq) ** 2 * var_component / n
        
        # Condition number κ = n·Var(D) / Σ V̂²
        # This equals 1/(1 - R²) where R² = Var(m̂)/Var(D)
        var_D = np.var(D)
        kappa_raw = (n * var_D) / sum_V_sq
        # Clamp to ensure κ ≥ 1 (corresponds to R²₊ = max{0, R²})
        kappa = np.maximum(1.0, kappa_raw)
        
        # Nuisance MSE (if true values provided)
        if m0_X is not None:
            mse_m = np.mean((m_hat - m0_X) ** 2)
        else:
            mse_m = np.nan
        
        if ell0_X is not None:
            mse_l = np.mean((l_hat - ell0_X) ** 2)
        else:
            mse_l = np.nan
        
        # Structural κ: from TRUE residuals (critical for Corrupted Oracle)
        # When learner residuals are contaminated by injected bias,
        # the estimated κ is inflated. Structural κ remains stable.
        if m0_X is not None:
            V_true = D - m0_X
            sum_V_true_sq = np.sum(V_true ** 2)
            if sum_V_true_sq > 1e-12:
                structural_kappa_raw = (n * var_D) / sum_V_true_sq
                structural_kappa = np.maximum(1.0, structural_kappa_raw)
            else:
                structural_kappa = np.inf
        else:
            # When true values unavailable, use estimated κ
            structural_kappa = kappa
        
        return theta_hat, var_hat, kappa, structural_kappa, mse_m, mse_l
    
    def fit(
        self,
        Y: NDArray,
        D: NDArray,
        X: NDArray,
        m0_X: Optional[NDArray] = None,
        ell0_X: Optional[NDArray] = None,
    ) -> DMLResult:
        """
        Fit DML estimator with cross-fitting.
        
        Parameters
        ----------
        Y : ndarray of shape (n,)
            Outcome variable.
        D : ndarray of shape (n,)
            Treatment variable.
        X : ndarray of shape (n, p)
            Covariate matrix.
        m0_X : ndarray or None, default None
            True propensity scores m₀(X) for MSE calculation and
            structural κ computation.
        ell0_X : ndarray or None, default None
            True reduced-form outcomes ℓ₀(X) for MSE calculation.
        
        Returns
        -------
        result : DMLResult
            Estimation results with diagnostics.
        """
        n = len(Y)
        
        if self.n_repeats == 1:
            # Single cross-fitting
            theta_hat, var_hat, kappa, structural_kappa, mse_m, mse_l = self._single_crossfit(
                Y, D, X, m0_X, ell0_X, self.random_state
            )
            se = np.sqrt(var_hat)
        else:
            # Repeated cross-fitting with median aggregation
            theta_hats = []
            var_hats = []
            kappas = []
            structural_kappas = []
            mse_ms = []
            mse_ls = []
            
            for rep in range(self.n_repeats):
                split_seed = self.random_state + rep * 1000
                th, vh, ks, sk, mm, ml = self._single_crossfit(
                    Y, D, X, m0_X, ell0_X, split_seed
                )
                theta_hats.append(th)
                var_hats.append(vh)
                kappas.append(ks)
                structural_kappas.append(sk)
                mse_ms.append(mm)
                mse_ls.append(ml)
            
            # Median aggregation (Chernozhukov et al. 2018)
            theta_hat = np.nanmedian(theta_hats)
            
            # Variance: median + adjustment for splitting variability
            median_var = np.nanmedian(var_hats)
            split_var = np.nanmedian((np.array(theta_hats) - theta_hat) ** 2) / self.n_repeats
            var_hat = median_var + split_var
            se = np.sqrt(var_hat)
            
            kappa = np.nanmedian(kappas)
            structural_kappa = np.nanmedian(structural_kappas)
            mse_m = np.nanmedian(mse_ms)
            mse_l = np.nanmedian(mse_ls)
        
        # Confidence interval
        ci_lower = theta_hat - Z_ALPHA * se
        ci_upper = theta_hat + Z_ALPHA * se
        
        # Bias (for simulation evaluation)
        bias = theta_hat - self.theta0
        
        # Sample R²(D|X) = 1 - 1/κ
        sample_r2 = 1.0 - 1.0 / kappa if kappa > 0 and kappa < np.inf else 0.0
        
        self.result_ = DMLResult(
            theta_hat=theta_hat,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            kappa=kappa,
            structural_kappa=structural_kappa,
            bias=bias,
            nuisance_mse_m=mse_m,
            nuisance_mse_l=mse_l,
            sample_r2=sample_r2,
            n=n,
        )
        
        return self.result_


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_dml(
    Y: NDArray,
    D: NDArray,
    X: NDArray,
    learner_m: BaseEstimator,
    learner_l: BaseEstimator,
    m0_X: Optional[NDArray] = None,
    ell0_X: Optional[NDArray] = None,
    K: int = K_FOLDS,
    n_repeats: int = 1,
    theta0: float = THETA0,
    random_state: int = 42,
) -> DMLResult:
    """
    Run DML estimation (convenience function).
    
    Parameters
    ----------
    Y : ndarray of shape (n,)
        Outcome.
    D : ndarray of shape (n,)
        Treatment.
    X : ndarray of shape (n, p)
        Covariates.
    learner_m, learner_l : BaseEstimator
        Learners for m(X) and ℓ(X).
    m0_X, ell0_X : ndarray or None
        True nuisance values for diagnostics.
    K : int, default 5
        Number of cross-fitting folds.
    n_repeats : int, default 1
        Number of repeated cross-fittings.
    theta0 : float, default 1.0
        True treatment effect.
    random_state : int, default 42
        Random seed.
    
    Returns
    -------
    result : DMLResult
        Estimation results.
    """
    estimator = DMLEstimator(
        learner_m=learner_m,
        learner_l=learner_l,
        K=K,
        n_repeats=n_repeats,
        theta0=theta0,
        random_state=random_state,
    )
    return estimator.fit(Y, D, X, m0_X, ell0_X)
