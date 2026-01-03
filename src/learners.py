"""
Machine Learning Learners for Double Machine Learning.

This module provides a factory function for creating nuisance learners used
in DML estimation, including traditional methods (OLS, Lasso), tree-based
methods (Random Forest, Gradient Boosting), and neural networks (MLP).

Special Learners for Mechanism Validation
-----------------------------------------
- OracleLearner: Returns true nuisance values m₀(X) or ℓ₀(X). Establishes
  the theoretical lower bound with zero approximation error.

- CorruptedOracle: Returns true values with multiplicative bias (1 + δ).
  Used in Section 5 to isolate how κ amplifies nuisance bias per Theorem 3.8.
  The multiplicative form ensures bias survives DML's centering.

Learner Selection Rationale
---------------------------
- OLS: Correctly specified for linear m₀(X), misspecified for polynomial g₀(X)
- Lasso: Regularized linear model, introduces shrinkage bias
- RF: Can approximate nonlinear functions, introduces regularization bias
- MLP: Deep learner, requires standardization for convergence

References
----------
- Theorem 3.8: Corrupted Oracle validates the exact error decomposition
- Section 5.1: Learner comparison in Monte Carlo study
- Section 6: LaLonde application with multiple learners (Table 8)
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


if TYPE_CHECKING:
    from src.dgp import DGP


# =============================================================================
# CONSTANTS
# =============================================================================

LEARNER_NAMES = Literal[
    "OLS", "Lasso", "Ridge", "RF", "RF_Tuned", "GBM", "MLP", "Oracle", "CorruptedOracle"
]

RF_PARAM_GRID: Dict[str, Any] = {
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'max_features': ['sqrt', None],
}


# =============================================================================
# ORACLE LEARNER
# =============================================================================

class OracleLearner(BaseEstimator, RegressorMixin):
    """
    Oracle learner that returns true nuisance function values.
    
    The Oracle learner provides the theoretical lower bound for
    nuisance estimation error. With zero approximation error,
    any remaining estimation error is attributable to sampling
    variance only, isolating the variance inflation mechanism.
    
    Parameters
    ----------
    dgp : DGP
        The data generating process object containing m₀(X) and ℓ₀(X).
    target : {'m', 'l'}, default 'm'
        Which nuisance function to return:
        - 'm': propensity score m₀(X) = E[D|X]
        - 'l': reduced-form outcome ℓ₀(X) = E[Y|X]
    
    Notes
    -----
    Used in the baseline simulations to establish that the DML estimator
    achieves nominal coverage when nuisance functions are known.
    """
    
    def __init__(
        self,
        dgp: Optional["DGP"] = None,
        target: Literal['m', 'l'] = 'm'
    ) -> None:
        self.dgp = dgp
        self.target = target
        self.true_values_: Optional[NDArray] = None
        self.X_train_: Optional[NDArray] = None
    
    def fit(
        self,
        X: NDArray,
        y: NDArray,
        **kwargs
    ) -> "OracleLearner":
        """
        Store reference to training data (no actual fitting needed).
        
        The Oracle computes true values directly from the DGP,
        so fit() simply records the training covariates.
        """
        if self.dgp is None:
            raise ValueError("DGP must be provided for Oracle learner")
        
        self.X_train_ = X.copy()
        
        if self.target == 'm':
            self.true_values_ = self.dgp.m0(X)
        else:
            self.true_values_ = self.dgp.ell0(X)
        
        return self
    
    def predict(self, X: NDArray) -> NDArray:
        """
        Return true nuisance function values.
        
        Computes m₀(X) or ℓ₀(X) directly from the DGP for any input X.
        """
        if self.dgp is None:
            raise ValueError("DGP must be provided for Oracle learner")
        
        if self.target == 'm':
            return self.dgp.m0(X)
        else:
            return self.dgp.ell0(X)


# =============================================================================
# CORRUPTED ORACLE (Section 5: Bias Amplification Mechanism)
# =============================================================================

class CorruptedOracle(BaseEstimator, RegressorMixin):
    """
    Corrupted Oracle for validating the bias amplification mechanism.
    
    This learner returns true nuisance values with controlled multiplicative
    bias, used to validate Theorem 3.8's prediction that κ acts as an
    error multiplier:
        θ̂ - θ₀ ≈ κ × Rem_n
    
    By setting predictions = truth × (1 + δ), we inject known bias δ and
    verify that |θ̂ - θ₀| ∝ κ × δ (Figure 1b in the paper).
    
    Multiplicative vs Additive Bias
    -------------------------------
    Multiplicative bias (1 + δ) is used instead of additive bias because:
    - DML centers residuals, absorbing constant shifts into the intercept
    - Multiplicative bias scales the entire function, surviving centering
    - This isolates the pure κ amplification effect
    
    Parameters
    ----------
    true_function_callback : callable
        Function that takes X and returns true nuisance values.
        Typically dgp.m0 or dgp.ell0.
    bias : float, default 0.01
        Multiplicative bias factor. Predictions = truth × (1 + bias).
    
    See Also
    --------
    get_corrupted_oracle_pair : Creates matched pair for m and ℓ
    
    References
    ----------
    - Section 5: Corrupted Oracle Monte Carlo design
    - Theorem 3.8: Exact decomposition showing κ as multiplier
    - Figure 1: Bias amplification validation
    """
    
    def __init__(
        self,
        true_function_callback: Optional[callable] = None,
        bias: float = 0.01,
    ) -> None:
        self.true_function_callback = true_function_callback
        self.bias = bias
    
    def fit(self, X: NDArray, y: NDArray, **kwargs) -> "CorruptedOracle":
        """No fitting needed; returns biased true values directly."""
        return self
    
    def predict(self, X: NDArray) -> NDArray:
        """
        Return true nuisance values with multiplicative bias.
        
        Computes: truth(X) × (1 + δ)
        
        Parameters
        ----------
        X : ndarray of shape (n, p)
            Covariate matrix.
        
        Returns
        -------
        predictions : ndarray of shape (n,)
            Biased nuisance predictions.
        """
        if self.true_function_callback is None:
            raise ValueError("true_function_callback must be provided")
        
        truth = self.true_function_callback(X)
        return truth * (1.0 + self.bias)


def get_corrupted_oracle_pair(
    dgp: "DGP",
    bias_m: float = 0.01,
    bias_l: float = 0.01,
) -> tuple[CorruptedOracle, CorruptedOracle]:
    """
    Create paired Corrupted Oracle learners for both nuisance functions.
    
    Parameters
    ----------
    dgp : DGP
        The data generating process.
    bias_m : float, default 0.01
        Multiplicative bias for m₀(X): m̂(X) = m₀(X) × (1 + bias_m)
    bias_l : float, default 0.01
        Multiplicative bias for ℓ₀(X): ℓ̂(X) = ℓ₀(X) × (1 + bias_l)
    
    Returns
    -------
    corrupted_m, corrupted_l : CorruptedOracle
        Corrupted oracle learners for treatment and outcome models.
    
    Notes
    -----
    Setting bias_m = bias_l ("same-sign" bias) creates partial cancellation
    in the DML residuals. Setting bias_m = -bias_l ("opposite-sign" bias)
    maximizes amplification, as validated in Table 3.
    """
    return (
        CorruptedOracle(true_function_callback=dgp.m0, bias=bias_m),
        CorruptedOracle(true_function_callback=dgp.ell0, bias=bias_l),
    )


# =============================================================================
# LEARNER FACTORY
# =============================================================================

def get_learner(
    name: LEARNER_NAMES,
    tuned: bool = False,
    dgp: Optional["DGP"] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    params: Optional[Dict[str, Any]] = None,
) -> BaseEstimator:
    """
    Factory function to create nuisance regression models.
    
    Parameters
    ----------
    name : str
        Learner name. Supported values:
        - 'OLS': Linear regression (correctly specified for m₀, misspecified for g₀)
        - 'Lasso': L1-regularized linear regression with CV
        - 'Ridge': L2-regularized linear regression with CV
        - 'RF': Random Forest with fixed hyperparameters
        - 'RF_Tuned': Random Forest with tuned hyperparameters
        - 'GBM': Histogram-based Gradient Boosting
        - 'MLP': Multi-layer perceptron with standardization
        - 'Oracle': Returns true nuisance values (requires dgp)
        - 'CorruptedOracle': Biased true values (requires dgp)
    tuned : bool, default False
        Backward compatibility flag. If True and name='RF', returns RF_Tuned.
    dgp : DGP or None, default None
        Required for Oracle and CorruptedOracle learners.
    random_state : int, default 42
        Random state for reproducibility.
    n_jobs : int, default -1
        Number of parallel jobs for parallelizable learners.
    params : dict or None, default None
        Pre-tuned hyperparameters for RF_Tuned.
    
    Returns
    -------
    model : BaseEstimator
        Configured scikit-learn compatible regression model.
    
    Raises
    ------
    ValueError
        If unknown learner name or missing dgp for Oracle learners.
    """
    name_upper = name.upper()
    
    # Backward compatibility
    if tuned:
        if 'RF' in name_upper and 'TUNED' not in name_upper:
            name_upper = 'RF_TUNED'
        elif 'XGB' in name_upper and 'TUNED' not in name_upper:
            name_upper = 'XGB_TUNED'
    
    if name_upper == 'OLS':
        return LinearRegression()
    
    elif name_upper == 'LASSO':
        return LassoCV(
            cv=5,
            fit_intercept=True,
            max_iter=10000,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    
    elif name_upper == 'RIDGE':
        return RidgeCV(
            cv=5,
            fit_intercept=True,
        )
    
    elif name_upper == 'RF':
        # Fixed hyperparameters for consistent simulation behavior
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=n_jobs,
        )
    
    elif name_upper == 'RF_TUNED':
        if params is not None:
            # Use pre-tuned hyperparameters
            return RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
                n_jobs=n_jobs,
                **params,
            )
        # Return search object for tuning phase
        base_rf = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
        )
        return RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=RF_PARAM_GRID,
            n_iter=10,
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=random_state,
            n_jobs=n_jobs,
        )

    elif name_upper == 'GBM':
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state,
        )
    
    elif name_upper == 'MLP':
        # StandardScaler required for neural network convergence
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                max_iter=500,
                random_state=random_state,
            )
        )
    
    elif name_upper == 'ORACLE':
        if dgp is None:
            raise ValueError("DGP must be provided for Oracle learner")
        return OracleLearner(dgp=dgp, target='m')
    
    elif name_upper == 'CORRUPTEDORACLE' or name_upper == 'CORRUPTED_ORACLE':
        if dgp is None:
            raise ValueError("DGP must be provided for CorruptedOracle learner")
        return CorruptedOracle(true_function_callback=dgp.m0, bias=0.01)
    
    else:
        raise ValueError(
            f"Unknown learner: '{name}'. "
            f"Choose from: OLS, Lasso, Ridge, RF, RF_Tuned, GBM, MLP, Oracle, CorruptedOracle"
        )


# =============================================================================
# STRUCTURAL κ COMPUTATION
# =============================================================================

def compute_structural_kappa(
    D: NDArray,
    m0_X: NDArray,
) -> float:
    """
    Compute structural condition number κ from true nuisance values.
    
    The structural κ is computed from true propensity residuals V = D - m₀(X),
    making it invariant to learner bias. This is essential for Corrupted Oracle
    analysis where estimated κ̂ from biased learners would be contaminated.
    
    Definition (Definition 3.5 in paper)
    ------------------------------------
    κ = σ²_D / σ²_V = 1 / (1 - R²(D|X))
    
    where σ²_V = Var(D - m₀(X)) is the variance of true treatment residuals.
    
    Parameters
    ----------
    D : ndarray of shape (n,)
        Treatment variable.
    m0_X : ndarray of shape (n,)
        True propensity scores m₀(X) = E[D|X].
    
    Returns
    -------
    structural_kappa : float
        The structural condition number.
    
    Notes
    -----
    Table 5 in the paper verifies that structural κ remains constant
    across bias levels δ for a given R² regime, confirming this metric
    correctly captures the DGP's identification strength.
    
    Examples
    --------
    >>> from src.dgp import generate_data
    >>> Y, D, X, info, dgp = generate_data(n=1000, target_r2=0.90)
    >>> kappa = compute_structural_kappa(D, info['m0_X'])
    >>> print(f"κ ≈ {kappa:.1f}")  # Expected: ~10 for R²=0.90
    κ ≈ 10.0
    """
    n = len(D)
    V_true = D - m0_X  # True treatment residual
    var_D = np.var(D)
    sum_V_sq = np.sum(V_true ** 2)
    
    if sum_V_sq < 1e-12:
        return np.inf
    
    return (n * var_D) / sum_V_sq


# =============================================================================
# LEARNER SETS
# =============================================================================

# All available learners
AVAILABLE_LEARNERS = ['OLS', 'Lasso', 'Ridge', 'RF', 'RF_Tuned', 'GBM', 'MLP']

# LaLonde application (Section 6, Table 8): comprehensive comparison
LALONDE_LEARNERS = ['OLS', 'Lasso', 'Ridge', 'RF_Tuned', 'GBM', 'MLP']

# Simulation study: RF (fixed params) for computational efficiency
SIMULATION_LEARNERS = ['OLS', 'Lasso', 'RF', 'MLP']

__all__ = [
    'OracleLearner',
    'CorruptedOracle',
    'get_learner',
    'get_corrupted_oracle_pair',
    'compute_structural_kappa',
    'AVAILABLE_LEARNERS',
    'LALONDE_LEARNERS',
    'SIMULATION_LEARNERS',
    'LEARNER_NAMES',
    'RF_PARAM_GRID',
]
