"""
DML Diagnostic Estimator
========================

Core estimator class for Double Machine Learning with condition number diagnostics.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold


# =============================================================================
# Learner Factory
# =============================================================================

LearnerName = Literal["lin", "lasso", "ridge", "rf", "gbm"]


def get_learner(name: str, random_state: int = 42) -> BaseEstimator:
    """
    Get a nuisance learner by name.
    
    Parameters
    ----------
    name : str
        Learner name: 'lin', 'lasso', 'ridge', 'rf', 'gbm'.
    random_state : int, default 42
        Random state for reproducibility.
        
    Returns
    -------
    BaseEstimator
        Scikit-learn compatible estimator.
    """
    name = name.lower()
    
    if name == "lin":
        return LinearRegression()
    elif name == "lasso":
        return LassoCV(cv=5, random_state=random_state, max_iter=10000)
    elif name == "ridge":
        return RidgeCV(cv=5)
    elif name == "rf":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )
    elif name == "gbm":
        return GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Unknown learner: '{name}'. "
            f"Choose from: lin, lasso, ridge, rf, gbm"
        )


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class DMLResult:
    """
    Container for DML estimation results with diagnostics.
    
    Attributes
    ----------
    theta : float
        Estimated treatment effect.
    se : float
        Standard error.
    ci_lower : float
        Lower bound of 95% confidence interval.
    ci_upper : float
        Upper bound of 95% confidence interval.
    kappa : float
        Condition number κ_DML.
    regime : str
        Conditioning regime: 'well-conditioned', 'moderate', or 'severe'.
    jacobian : float
        Empirical Jacobian Ĵ_θ.
    n : int
        Sample size.
    r_squared_d : float
        R² of treatment regression.
    learner : str
        Name of learner used.
    U_hat : NDArray
        Treatment residuals.
    V_hat : NDArray
        Outcome residuals.
    """
    theta: float
    se: float
    ci_lower: float
    ci_upper: float
    kappa: float
    regime: str
    jacobian: float
    n: int
    r_squared_d: float
    learner: str
    U_hat: NDArray
    V_hat: NDArray
    
    @property
    def ci(self) -> Tuple[float, float]:
        """95% confidence interval as tuple."""
        return (self.ci_lower, self.ci_upper)
    
    @property
    def ci_length(self) -> float:
        """Length of confidence interval."""
        return self.ci_upper - self.ci_lower
    
    @property
    def t_stat(self) -> float:
        """t-statistic."""
        if self.se > 0:
            return self.theta / self.se
        return np.nan
    
    @property
    def pvalue(self) -> float:
        """Two-sided p-value."""
        from scipy import stats
        if np.isnan(self.t_stat):
            return np.nan
        return 2 * (1 - stats.norm.cdf(np.abs(self.t_stat)))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding large arrays)."""
        return {
            "theta": self.theta,
            "se": self.se,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_length": self.ci_length,
            "kappa": self.kappa,
            "regime": self.regime,
            "jacobian": self.jacobian,
            "n": self.n,
            "r_squared_d": self.r_squared_d,
            "learner": self.learner,
        }
    
    def __repr__(self) -> str:
        regime_symbol = {"well-conditioned": "✓", "moderate": "⚠", "severe": "✗"}
        symbol = regime_symbol.get(self.regime, "")
        
        return (
            f"\n"
            f"DML Diagnostic Results\n"
            f"──────────────────────\n"
            f"  θ̂ = {self.theta:.4f} (SE = {self.se:.4f})\n"
            f"  95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"\n"
            f"  Condition Number: κ_DML = {self.kappa:.2f}\n"
            f"  Regime: {self.regime.upper()} {symbol}\n"
            f"\n"
            f"  n = {self.n}, R²(D|X) = {self.r_squared_d:.3f}, learner = {self.learner}\n"
        )


# =============================================================================
# Main Estimator Class
# =============================================================================

class DMLDiagnostic:
    """
    Double Machine Learning with condition number diagnostics.
    
    This class implements the cross-fitted DML estimator for the Partially
    Linear Regression (PLR) model and computes the condition number κ_DML
    to diagnose inference reliability.
    
    Parameters
    ----------
    learner : str or BaseEstimator, default 'lasso'
        Nuisance learner for both treatment and outcome regressions.
        If str, one of: 'lin', 'lasso', 'ridge', 'rf', 'gbm'.
        If BaseEstimator, a scikit-learn compatible regressor.
    learner_m : str or BaseEstimator, optional
        Separate learner for treatment regression m(X) = E[D|X].
        If None, uses `learner`.
    learner_g : str or BaseEstimator, optional
        Separate learner for outcome regression g(X) = E[Y|X].
        If None, uses `learner`.
    n_folds : int, default 5
        Number of folds for cross-fitting.
    random_state : int, default 42
        Random state for reproducibility.
        
    Examples
    --------
    >>> from dml_diagnostic import DMLDiagnostic, load_lalonde
    >>> Y, D, X = load_lalonde(sample='experimental')
    >>> dml = DMLDiagnostic(learner='lasso')
    >>> results = dml.fit(Y, D, X)
    >>> print(results)
    
    DML Diagnostic Results
    ──────────────────────
      θ̂ = 1794.34 (SE = 632.15)
      95% CI: [555.32, 3033.36]
    
      Condition Number: κ_DML = 2.15
      Regime: WELL-CONDITIONED ✓
    """
    
    def __init__(
        self,
        learner: Union[str, BaseEstimator] = "lasso",
        learner_m: Optional[Union[str, BaseEstimator]] = None,
        learner_g: Optional[Union[str, BaseEstimator]] = None,
        n_folds: int = 5,
        random_state: int = 42,
    ):
        self.learner = learner
        self.learner_m = learner_m if learner_m is not None else learner
        self.learner_g = learner_g if learner_g is not None else learner
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Will be populated after fit
        self.result_: Optional[DMLResult] = None
        self._learner_name = (
            learner if isinstance(learner, str)
            else type(learner).__name__
        )
    
    def _get_learner_instance(
        self, learner: Union[str, BaseEstimator]
    ) -> BaseEstimator:
        """Get a fresh learner instance."""
        if isinstance(learner, str):
            return get_learner(learner, self.random_state)
        else:
            return clone(learner)
    
    def fit(
        self,
        Y: NDArray,
        D: NDArray,
        X: NDArray,
    ) -> DMLResult:
        """
        Fit DML estimator and compute condition number diagnostic.
        
        Parameters
        ----------
        Y : array-like of shape (n,)
            Outcome variable.
        D : array-like of shape (n,)
            Treatment variable.
        X : array-like of shape (n, p)
            Covariate matrix.
            
        Returns
        -------
        DMLResult
            Results object with estimate, SE, CI, κ_DML, and diagnostics.
        """
        # Input validation
        Y = np.asarray(Y).ravel()
        D = np.asarray(D).ravel()
        X = np.asarray(X)
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n = len(Y)
        
        if len(D) != n or X.shape[0] != n:
            raise ValueError("Y, D, and X must have the same number of observations")
        
        # Initialize arrays for cross-fitted predictions
        m_hat = np.zeros(n)  # E[D|X] predictions
        g_hat = np.zeros(n)  # E[Y|X] predictions
        
        # K-fold cross-fitting
        kf = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        
        for train_idx, test_idx in kf.split(X):
            X_train, Y_train, D_train = X[train_idx], Y[train_idx], D[train_idx]
            X_test = X[test_idx]
            
            # Fit treatment regression: m(X) = E[D|X]
            model_m = self._get_learner_instance(self.learner_m)
            model_m.fit(X_train, D_train)
            m_hat[test_idx] = model_m.predict(X_test)
            
            # Fit outcome regression: g(X) = E[Y|X]
            model_g = self._get_learner_instance(self.learner_g)
            model_g.fit(X_train, Y_train)
            g_hat[test_idx] = model_g.predict(X_test)
        
        # =====================================================================
        # Compute DML estimator
        # =====================================================================
        
        # Cross-fitted residuals
        U_hat = D - m_hat  # Treatment residuals
        V_hat = Y - g_hat  # Outcome residuals
        
        # PLR estimator: θ̂ = Σ Û V̂ / Σ Û²
        sum_U_sq = np.sum(U_hat ** 2)
        sum_UV = np.sum(U_hat * V_hat)
        
        if sum_U_sq < 1e-10:
            warnings.warn(
                "Near-zero residual treatment variance. "
                "Estimation is severely ill-conditioned."
            )
            theta_hat = np.nan
        else:
            theta_hat = sum_UV / sum_U_sq
        
        # =====================================================================
        # Condition number: κ_DML = n / Σ Û²
        # =====================================================================
        kappa = n / sum_U_sq
        jacobian = -sum_U_sq / n
        
        # =====================================================================
        # Standard error estimation
        # =====================================================================
        eps_hat = V_hat - theta_hat * U_hat  # Final residuals
        score_variance = np.mean((U_hat ** 2) * (eps_hat ** 2))
        se = (kappa / np.sqrt(n)) * np.sqrt(score_variance)
        
        # 95% confidence interval
        z_95 = 1.96
        ci_lower = theta_hat - z_95 * se
        ci_upper = theta_hat + z_95 * se
        
        # R² of treatment regression
        var_D = np.var(D)
        var_U = np.var(U_hat)
        r_squared_d = 1 - var_U / var_D if var_D > 0 else np.nan
        
        # Classify regime
        from dml_diagnostic.diagnostics import classify_regime
        regime_info = classify_regime(kappa, n)
        
        # Create result object
        self.result_ = DMLResult(
            theta=theta_hat,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            kappa=kappa,
            regime=regime_info["regime"],
            jacobian=jacobian,
            n=n,
            r_squared_d=r_squared_d,
            learner=self._learner_name,
            U_hat=U_hat,
            V_hat=V_hat,
        )
        
        return self.result_
    
    def summary(self) -> str:
        """
        Generate a detailed summary with interpretation.
        
        Returns
        -------
        str
            Human-readable summary including guidance on interpretation.
        """
        if self.result_ is None:
            return "Model not fitted. Call .fit() first."
        
        from dml_diagnostic.diagnostics import classify_regime
        regime_info = classify_regime(self.result_.kappa, self.result_.n)
        
        return str(self.result_) + f"\n  Interpretation: {regime_info['interpretation']}\n"
    
    def __repr__(self) -> str:
        if self.result_ is None:
            return (
                f"DMLDiagnostic(learner='{self._learner_name}', "
                f"n_folds={self.n_folds}) [not fitted]"
            )
        return self.summary()
