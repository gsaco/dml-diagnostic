"""
DML estimator with condition number diagnostic (κ_DML) for the PLR model.

This module implements the Double Machine Learning (DML) estimator for the
Partially Linear Regression (PLR) model, with explicit computation of the
condition number diagnostic κ_DML.

The PLR model is:
    Y = θ₀ D + g₀(X) + ε,    E[ε | D, X] = 0

where:
    - Y is the outcome
    - D is the treatment
    - X are covariates
    - θ₀ is the parameter of interest
    - g₀(X) is an unknown nuisance function

The DML estimator uses the Neyman-orthogonal score:
    ψ(W; θ, η) = (D - m(X)) (Y - g(X) - θ(D - m(X)))

where:
    - m(X) = E[D | X] (treatment regression)
    - g(X) = E[Y | X] (outcome regression, or ℓ₀(X) in notation)
    - η = (g, m) are nuisance functions

The DML estimator is:
    θ̂ = Σᵢ Ûᵢ V̂ᵢ / Σᵢ Ûᵢ²

where Ûᵢ = Dᵢ - m̂(Xᵢ) and V̂ᵢ = Yᵢ - ĝ(Xᵢ) are cross-fitted residuals.

The condition number is:
    κ_DML = n / Σᵢ Ûᵢ²

which measures the flatness of the orthogonal score. Large κ_DML indicates
poor conditioning and potential fragility of DML inference.

Three conditioning regimes (from the paper):
    (i)   Well-conditioned:     κ_DML ≈ O(1)      → nominal coverage, CI ∝ 1/√n
    (ii)  Moderately ill-cond:  κ_DML = O(n^β)    → inflated CI, reduced coverage
    (iii) Severely ill-cond:    κ_DML ≍ c√n       → uninformative CIs, severe bias

Practical threshold: κ_DML > 10 suggests caution in interpreting DML inference.

References:
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
    Newey, W., and Robins, J. (2018). "Double/Debiased Machine Learning for
    Treatment and Structural Parameters." The Econometrics Journal, 21(1), C1–C68.
    
    Saco, G. (2025). "Finite-Sample Failures and Condition-Number Diagnostics
    in Double Machine Learning." The Econometrics Journal.

Author: Gabriel Saco
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import clone, BaseEstimator, RegressorMixin
from typing import Dict, Tuple, Optional, Union, List, Any
import warnings


# =============================================================================
# Nuisance Learner Configurations
# =============================================================================

def get_learner(name: str, random_state: int = 42) -> BaseEstimator:
    """
    Get a nuisance learner by name.
    
    Parameters
    ----------
    name : str
        Learner name: 'LIN', 'LASSO', 'RIDGE', 'RF', 'GBM'.
    random_state : int, default 42
        Random state for reproducibility.
        
    Returns
    -------
    BaseEstimator
        Scikit-learn compatible estimator.
    """
    name = name.upper()
    
    if name == 'LIN':
        return LinearRegression()
    
    elif name == 'LASSO':
        return LassoCV(cv=5, random_state=random_state, max_iter=10000)
    
    elif name == 'RIDGE':
        return RidgeCV(cv=5)
    
    elif name == 'RF':
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1
        )
    
    elif name == 'GBM':
        return GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state
        )
    
    else:
        raise ValueError(f"Unknown learner: {name}. "
                        f"Choose from: LIN, LASSO, RIDGE, RF, GBM")


# =============================================================================
# PLR DML Estimator with κ_DML Diagnostic
# =============================================================================

class PLRDoubleMLKappa:
    """
    Double Machine Learning estimator for the PLR model with κ_DML diagnostic.
    
    Implements the cross-fitted DML estimator for the Partially Linear
    Regression model and computes the condition number diagnostic κ_DML.
    
    Parameters
    ----------
    learner_m : str or BaseEstimator, default 'LASSO'
        Learner for the treatment regression m(X) = E[D|X].
        If str, uses get_learner() to instantiate.
    learner_g : str or BaseEstimator, default 'LASSO'
        Learner for the outcome regression g(X) = E[Y|X].
        If str, uses get_learner() to instantiate.
    n_folds : int, default 5
        Number of folds for cross-fitting.
    random_state : int, default 42
        Random state for reproducibility.
        
    Attributes
    ----------
    theta_hat : float
        Estimated treatment effect.
    se_dml : float
        Standard error of the DML estimator.
    kappa_dml : float
        Condition number κ_DML = n / Σᵢ Ûᵢ².
    jacobian : float
        Empirical Jacobian Ĵ_θ = -(1/n) Σᵢ Ûᵢ².
    ci_95 : tuple
        95% confidence interval (lower, upper).
    n_obs : int
        Number of observations.
    U_hat : np.ndarray
        Cross-fitted treatment residuals Ûᵢ = Dᵢ - m̂(Xᵢ).
    V_hat : np.ndarray
        Cross-fitted outcome residuals V̂ᵢ = Yᵢ - ĝ(Xᵢ).
    eps_hat : np.ndarray
        Final residuals ε̂ᵢ = Yᵢ - ĝ(Xᵢ) - θ̂(Dᵢ - m̂(Xᵢ)).
    r_squared_d : float
        R² of the treatment regression (in-sample, for diagnostics).
    """
    
    def __init__(
        self,
        learner_m: Union[str, BaseEstimator] = 'LASSO',
        learner_g: Union[str, BaseEstimator] = 'LASSO',
        n_folds: int = 5,
        random_state: int = 42
    ):
        self.learner_m = learner_m
        self.learner_g = learner_g
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Will be set after fitting
        self.theta_hat: Optional[float] = None
        self.se_dml: Optional[float] = None
        self.kappa_dml: Optional[float] = None
        self.jacobian: Optional[float] = None
        self.ci_95: Optional[Tuple[float, float]] = None
        self.n_obs: Optional[int] = None
        self.U_hat: Optional[np.ndarray] = None
        self.V_hat: Optional[np.ndarray] = None
        self.eps_hat: Optional[np.ndarray] = None
        self.r_squared_d: Optional[float] = None
        
        # Store learner names for reporting
        self._learner_m_name = learner_m if isinstance(learner_m, str) else type(learner_m).__name__
        self._learner_g_name = learner_g if isinstance(learner_g, str) else type(learner_g).__name__
    
    def _get_learner_instance(self, learner: Union[str, BaseEstimator]) -> BaseEstimator:
        """Get a fresh learner instance."""
        if isinstance(learner, str):
            return get_learner(learner, self.random_state)
        else:
            return clone(learner)
    
    def fit(
        self,
        Y: np.ndarray,
        D: np.ndarray,
        X: np.ndarray
    ) -> 'PLRDoubleMLKappa':
        """
        Fit the DML estimator and compute κ_DML.
        
        Parameters
        ----------
        Y : np.ndarray of shape (n,)
            Outcome variable.
        D : np.ndarray of shape (n,)
            Treatment variable.
        X : np.ndarray of shape (n, p)
            Covariate matrix.
            
        Returns
        -------
        self
            Fitted estimator.
        """
        Y = np.asarray(Y).ravel()
        D = np.asarray(D).ravel()
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n = len(Y)
        self.n_obs = n
        
        # Initialise arrays for cross-fitted predictions
        m_hat = np.zeros(n)  # E[D|X] predictions
        g_hat = np.zeros(n)  # E[Y|X] predictions
        
        # K-fold cross-fitting
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for train_idx, test_idx in kf.split(X):
            # Training data (out-of-fold)
            X_train, Y_train, D_train = X[train_idx], Y[train_idx], D[train_idx]
            
            # Test data (in-fold for predictions)
            X_test = X[test_idx]
            
            # Fit treatment regression: m(X) = E[D|X]
            learner_m = self._get_learner_instance(self.learner_m)
            learner_m.fit(X_train, D_train)
            m_hat[test_idx] = learner_m.predict(X_test)
            
            # Fit outcome regression: g(X) = E[Y|X]
            learner_g = self._get_learner_instance(self.learner_g)
            learner_g.fit(X_train, Y_train)
            g_hat[test_idx] = learner_g.predict(X_test)
        
        # =====================================================================
        # Compute DML estimator
        # =====================================================================
        
        # Cross-fitted residuals
        # Û_i = D_i - m̂(X_i): treatment residuals
        self.U_hat = D - m_hat
        
        # V̂_i = Y_i - ĝ(X_i): outcome residuals  
        self.V_hat = Y - g_hat
        
        # PLR estimator: θ̂ = Σ_i Û_i V̂_i / Σ_i Û_i²
        sum_U_sq = np.sum(self.U_hat ** 2)
        sum_UV = np.sum(self.U_hat * self.V_hat)
        
        if sum_U_sq < 1e-10:
            warnings.warn("Near-zero residual treatment variance. "
                         "Estimation is severely ill-conditioned.")
            self.theta_hat = np.nan
            self.kappa_dml = np.inf
        else:
            self.theta_hat = sum_UV / sum_U_sq
        
        # =====================================================================
        # Condition number: κ_DML = n / Σ_i Û_i²
        # =====================================================================
        # This is the key diagnostic from the paper.
        # Large κ_DML indicates poor conditioning (flat score).
        
        self.kappa_dml = n / sum_U_sq
        
        # Empirical Jacobian: Ĵ_θ = -(1/n) Σ_i Û_i²
        self.jacobian = -sum_U_sq / n
        
        # =====================================================================
        # Standard error estimation
        # =====================================================================
        
        # Final residuals: ε̂_i = Y_i - ĝ(X_i) - θ̂(D_i - m̂(X_i))
        #                      = V̂_i - θ̂ Û_i
        self.eps_hat = self.V_hat - self.theta_hat * self.U_hat
        
        # Variance of the score: Var(ψ) ≈ (1/n) Σ_i Û_i² ε̂_i²
        score_variance = np.mean((self.U_hat ** 2) * (self.eps_hat ** 2))
        
        # Standard error of θ̂:
        # SE_DML = (κ_DML / √n) × √((1/n) Σ_i Û_i² ε̂_i²)
        #        = √(n × Σ_i Û_i² ε̂_i² / (Σ_i Û_i²)²)
        #        = √(Σ_i Û_i² ε̂_i² / (Σ_i Û_i²)² × n)
        
        self.se_dml = (self.kappa_dml / np.sqrt(n)) * np.sqrt(score_variance)
        
        # 95% confidence interval
        z_95 = 1.96
        self.ci_95 = (
            self.theta_hat - z_95 * self.se_dml,
            self.theta_hat + z_95 * self.se_dml
        )
        
        # Diagnostic: R² of treatment regression
        var_D = np.var(D)
        var_U = np.var(self.U_hat)
        if var_D > 0:
            self.r_squared_d = 1 - var_U / var_D
        else:
            self.r_squared_d = np.nan
        
        return self
    
    def summary_dict(self) -> Dict[str, Any]:
        """
        Return a summary dictionary of estimation results.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - theta_hat: point estimate
            - se_dml: standard error
            - ci_95_lower, ci_95_upper: 95% CI bounds
            - ci_length: CI width
            - kappa_dml: condition number
            - jacobian: empirical Jacobian
            - n: sample size
            - learner_m: treatment learner name
            - learner_g: outcome learner name
            - r_squared_d: R² of treatment regression
        """
        if self.theta_hat is None:
            raise ValueError("Model not fitted. Call .fit() first.")
        
        return {
            'theta_hat': self.theta_hat,
            'se_dml': self.se_dml,
            'ci_95_lower': self.ci_95[0],
            'ci_95_upper': self.ci_95[1],
            'ci_length': self.ci_95[1] - self.ci_95[0],
            'kappa_dml': self.kappa_dml,
            'jacobian': self.jacobian,
            'n': self.n_obs,
            'learner_m': self._learner_m_name,
            'learner_g': self._learner_g_name,
            'r_squared_d': self.r_squared_d
        }
    
    def __repr__(self) -> str:
        if self.theta_hat is None:
            return f"PLRDoubleMLKappa(learner_m={self._learner_m_name}, " \
                   f"learner_g={self._learner_g_name}, n_folds={self.n_folds}) [not fitted]"
        else:
            return (
                f"PLRDoubleMLKappa Results:\n"
                f"  θ̂ = {self.theta_hat:.4f} (SE = {self.se_dml:.4f})\n"
                f"  95% CI: [{self.ci_95[0]:.4f}, {self.ci_95[1]:.4f}]\n"
                f"  κ_DML = {self.kappa_dml:.4f}\n"
                f"  n = {self.n_obs}, learners = ({self._learner_m_name}, {self._learner_g_name})"
            )


# =============================================================================
# Utility Functions
# =============================================================================

def estimate_propensity_score(
    D: np.ndarray,
    X: np.ndarray,
    method: str = 'logistic',
    random_state: int = 42
) -> np.ndarray:
    """
    Estimate propensity scores e(X) = P(D=1|X) for overlap diagnostics.
    
    Parameters
    ----------
    D : np.ndarray
        Binary treatment indicator.
    X : np.ndarray
        Covariates.
    method : str, default 'logistic'
        Method for propensity estimation: 'logistic', 'rf'.
    random_state : int
        Random state for reproducibility.
        
    Returns
    -------
    np.ndarray
        Estimated propensity scores.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.ensemble import RandomForestClassifier
    
    X = np.asarray(X)
    D = np.asarray(D).ravel()
    
    if method == 'logistic':
        model = LogisticRegressionCV(cv=5, random_state=random_state, max_iter=1000)
    elif method == 'rf':
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=random_state, n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    model.fit(X, D)
    e_hat = model.predict_proba(X)[:, 1]
    
    return e_hat


def compute_overlap_diagnostics(
    D: np.ndarray,
    X: np.ndarray,
    method: str = 'logistic',
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute overlap diagnostics based on propensity scores.
    
    Parameters
    ----------
    D : np.ndarray
        Binary treatment indicator.
    X : np.ndarray
        Covariates.
    method : str
        Propensity score estimation method.
    random_state : int
        Random state.
        
    Returns
    -------
    dict
        Dictionary with:
        - e_mean_treated: mean propensity among treated
        - e_mean_control: mean propensity among controls
        - e_min: minimum propensity
        - e_max: maximum propensity
        - pct_in_01_09: percentage with e(X) ∈ [0.1, 0.9]
    """
    e_hat = estimate_propensity_score(D, X, method=method, random_state=random_state)
    D = np.asarray(D).ravel()
    
    return {
        'e_mean_treated': np.mean(e_hat[D == 1]),
        'e_mean_control': np.mean(e_hat[D == 0]),
        'e_min': np.min(e_hat),
        'e_max': np.max(e_hat),
        'e_std': np.std(e_hat),
        'pct_in_01_09': np.mean((e_hat >= 0.1) & (e_hat <= 0.9)) * 100,
        'e_hat': e_hat  # Include the full vector for plotting
    }


def run_dml_analysis(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    learners: List[str] = ['LIN', 'LASSO', 'RF'],
    n_folds: int = 5,
    random_state: int = 42
) -> List[Dict[str, Any]]:
    """
    Run DML analysis with multiple learners.
    
    Parameters
    ----------
    Y, D, X : np.ndarray
        Outcome, treatment, and covariates.
    learners : list of str
        List of learner names.
    n_folds : int
        Number of cross-fitting folds.
    random_state : int
        Random state.
        
    Returns
    -------
    list of dict
        List of summary dictionaries, one per learner.
    """
    results = []
    
    for learner in learners:
        model = PLRDoubleMLKappa(
            learner_m=learner,
            learner_g=learner,
            n_folds=n_folds,
            random_state=random_state
        )
        model.fit(Y, D, X)
        results.append(model.summary_dict())
    
    return results


if __name__ == "__main__":
    # Test the DML estimator with synthetic data
    print("Testing PLRDoubleMLKappa with synthetic data...\n")
    
    np.random.seed(42)
    n = 500
    p = 5
    
    # Generate data: PLR model
    X = np.random.randn(n, p)
    D = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n)
    D = (D > np.median(D)).astype(float)  # Binary treatment
    theta_true = 1.0
    Y = theta_true * D + 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.5 * np.random.randn(n)
    
    # Fit with different learners
    for learner in ['LIN', 'LASSO', 'RF']:
        print(f"\n--- Learner: {learner} ---")
        model = PLRDoubleMLKappa(learner_m=learner, learner_g=learner)
        model.fit(Y, D, X)
        print(model)
        print(f"True θ = {theta_true}")
