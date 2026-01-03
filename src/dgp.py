"""
Data Generating Process for Double Machine Learning Simulations.

This module implements the Partially Linear Regression (PLR) data generating
process used in Section 5 of "Ill-Conditioned Orthogonal Scores in Double
Machine Learning" for Monte Carlo validation of the bias amplification mechanism.

Theoretical Foundation
----------------------
The PLR model (Equation 1) specifies:
    Y = θ₀D + g₀(X) + ε,    E[ε|D,X] = 0
    D = m₀(X) + V,          E[V|X] = 0

The condition number κ (Definition 3.5) governs identification strength:
    κ = σ²_D / σ²_V = 1 / (1 - R²(D|X))

Theorem 3.8 shows that κ amplifies both variance (via 1/√n scaling) and 
nuisance bias (via the remainder term Rem_n).

DGP Specification (Section 5.1)
-------------------------------
Covariates:   X ~ N(0, Σ) with AR(1) correlation Σ_{jk} = ρ^{|j-k|}, ρ = 0.5
Treatment:    D = β'X + σ_U·U,  β = (0.7, 0.7², ..., 0.7^p)  [LINEAR]
Outcome:      Y = θ₀D + g₀(X) + ε,  g₀(X) = X₁ + X₂² + X₃X₄  [POLYNOMIAL]

Design Rationale:
- Linear m₀(X) = β'X: OLS achieves zero bias, isolating variance inflation
- Polynomial g₀(X): RF introduces regularization bias that κ amplifies

References
----------
- Theorem 3.8: Exact decomposition of estimator error
- Theorem 3.11: Stochastic-order bound on θ̂ - θ₀
- Section 5.1: Monte Carlo design specification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# CONSTANTS (Section 5.1)
# =============================================================================

THETA0: float = 1.0          # True treatment effect θ₀
P_DEFAULT: int = 10          # Covariate dimension p
RHO_DEFAULT: float = 0.5     # AR(1) correlation parameter ρ
SIGMA_EPS: float = 1.0       # Outcome noise standard deviation σ_ε


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def make_toeplitz_cov(p: int, rho: float) -> NDArray:
    """
    Construct Toeplitz covariance matrix Σ with entries Σ_{jk} = ρ^{|j-k|}.
    
    This AR(1) correlation structure is standard in econometric simulations
    and creates dependence between adjacent covariates.
    
    Parameters
    ----------
    p : int
        Dimension of the covariance matrix.
    rho : float
        Correlation decay parameter in [0, 1).
    
    Returns
    -------
    Sigma : ndarray of shape (p, p)
        Symmetric positive-definite Toeplitz covariance matrix.
    """
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


# =============================================================================
# PLR DATA GENERATING PROCESS (Section 5.1)
# =============================================================================

@dataclass
class DGP:
    """
    Partially Linear Regression DGP for Monte Carlo simulations.
    
    This class generates data from the PLR model specified in Section 5.1,
    with calibration to achieve target overlap conditions (R²(D|X)).
    
    Covariate Generation
    --------------------
    X ~ N(0, Σ) where Σ_{jk} = ρ^{|j-k|} (AR(1) correlation)
    
    Treatment Equation (Linear)
    ---------------------------
    D = β'X + σ_U·U,  U ~ N(0,1)
    β = (0.7, 0.7², ..., 0.7^p)  [Geometrically decaying coefficients]
    σ_U calibrated to achieve target R²(D|X)
    
    Outcome Equation
    ----------------
    Y = θ₀D + g₀(X) + ε,  ε ~ N(0, σ²_ε)
    g₀(X) = X₁ + X₂² + X₃X₄  [Polynomial with interaction]
    
    Parameters
    ----------
    p : int, default 10
        Covariate dimension.
    rho : float, default 0.5
        AR(1) correlation parameter.
    theta0 : float, default 1.0
        True treatment effect.
    sigma_eps : float, default 1.0
        Outcome noise standard deviation.
    target_r2 : float, default 0.90
        Target R²(D|X) determining overlap strength:
        - R² = 0.50 → κ ≈ 2 (strong overlap)
        - R² = 0.90 → κ ≈ 10 (moderate overlap)
        - R² = 0.97 → κ ≈ 33 (weak overlap)
    random_state : int or None, default None
        Random seed for reproducibility.
    
    Attributes
    ----------
    beta : ndarray of shape (p,)
        Treatment coefficients β = (0.7¹, 0.7², ..., 0.7^p).
    sigma_U : float
        Treatment noise std (calibrated to achieve target_r2).
    
    Notes
    -----
    The linear treatment equation ensures OLS achieves zero bias for m₀(X),
    isolating the variance inflation mechanism. The polynomial outcome
    equation introduces regularization bias in machine learning estimators
    that gets amplified by κ per Theorem 3.8.
    """
    p: int = P_DEFAULT
    rho: float = RHO_DEFAULT
    theta0: float = THETA0
    sigma_eps: float = SIGMA_EPS
    target_r2: float = 0.90
    random_state: Optional[int] = None
    
    # Computed after initialization
    beta: NDArray = field(default=None, init=False, repr=False)
    sigma_U: float = field(default=0.5, init=False)
    is_calibrated: bool = field(default=False, init=False)
    _Sigma: NDArray = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize coefficients, covariance matrix, and calibrate noise."""
        # Geometrically decaying treatment coefficients β_j = 0.7^j
        self.beta = np.array([0.7 ** j for j in range(1, self.p + 1)])
        
        # AR(1) Toeplitz covariance matrix
        self._Sigma = make_toeplitz_cov(self.p, self.rho)
        
        # Calibrate σ_U to achieve target R²(D|X)
        self.calibrate_noise(self.target_r2)
    
    def m0(self, X: NDArray) -> NDArray:
        """
        Propensity score: m₀(X) = E[D|X] = β'X.
        
        The linear specification ensures OLS is correctly specified,
        achieving zero approximation bias for the treatment model.
        """
        return X @ self.beta
    
    def g0(self, X: NDArray) -> NDArray:
        """
        Outcome confounding function: g₀(X) = X₁ + X₂² + X₃X₄.
        
        The polynomial specification with quadratic and interaction terms
        ensures that linear methods are misspecified, while machine learning
        methods incur regularization bias that κ amplifies.
        """
        return X[:, 0] + X[:, 1]**2 + X[:, 2] * X[:, 3]
    
    def ell0(self, X: NDArray) -> NDArray:
        """
        Reduced-form outcome: ℓ₀(X) = E[Y|X] = θ₀·m₀(X) + g₀(X).
        
        This follows from the PLR structure by iterated expectations.
        """
        return self.theta0 * self.m0(X) + self.g0(X)
    
    def calibrate_noise(
        self,
        target_r2: float,
        n_samples: int = 100000,
        seed: int = 42
    ) -> float:
        """
        Calibrate treatment noise σ_U to achieve target R²(D|X).
        
        For the linear treatment equation D = β'X + σ_U·U:
            Var(D) = Var(β'X) + σ²_U
            R²(D|X) = Var(β'X) / Var(D)
        
        Solving for σ_U:
            σ²_U = Var(m₀) × (1 - R²) / R²
        
        This calibration determines κ = 1/(1 - R²) per Definition 3.5.
        
        Parameters
        ----------
        target_r2 : float
            Target R²(D|X) in (0, 1). Higher values → higher κ → weaker overlap.
        n_samples : int, default 100000
            Monte Carlo sample size for variance estimation.
        seed : int, default 42
            Random seed for calibration.
        
        Returns
        -------
        sigma_U : float
            Calibrated treatment noise standard deviation.
        """
        if not 0 < target_r2 < 1:
            raise ValueError(f"target_r2 must be in (0, 1), got {target_r2}")
        
        rng = np.random.default_rng(seed)
        X_calib = rng.multivariate_normal(np.zeros(self.p), self._Sigma, size=n_samples)
        
        # Compute Var(β'X) = Var(m₀(X))
        m0_X = self.m0(X_calib)
        var_m0 = np.var(m0_X)
        
        # Solve: R² = Var(m₀) / (Var(m₀) + σ²_U)  →  σ²_U = Var(m₀) × (1-R²)/R²
        sigma_U_sq = var_m0 * (1 - target_r2) / target_r2
        self.sigma_U = np.sqrt(sigma_U_sq)
        
        self.target_r2 = target_r2
        self.is_calibrated = True
        
        return self.sigma_U
    
    def generate(
        self,
        n: int,
        random_state: Optional[int] = None
    ) -> Tuple[NDArray, NDArray, NDArray, Dict]:
        """
        Generate data from the PLR DGP.
        
        Parameters
        ----------
        n : int
            Sample size.
        random_state : int or None, default None
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
            Diagnostic information including:
            - target_r2, sample_r2: Target and realized R²(D|X)
            - sigma_U: Treatment noise std
            - m0_X, g0_X, ell0_X: True nuisance function values
            - beta: Treatment coefficient vector
        """
        if not self.is_calibrated:
            raise RuntimeError("DGP not calibrated. Call calibrate_noise() first.")
        
        seed = random_state if random_state is not None else self.random_state
        rng = np.random.default_rng(seed)
        
        # Covariates: X ~ N(0, Σ)
        X = rng.multivariate_normal(np.zeros(self.p), self._Sigma, size=n)
        
        # Treatment: D = β'X + σ_U·U (linear)
        m0_X = self.m0(X)
        U = rng.normal(0, 1, size=n)
        D = m0_X + self.sigma_U * U
        
        # Outcome: Y = θ₀D + g₀(X) + ε
        g0_X = self.g0(X)
        eps = rng.normal(0, self.sigma_eps, size=n)
        Y = self.theta0 * D + g0_X + eps
        
        # Sample R²(D|X) = Var(m₀) / Var(D)
        var_D = np.var(D)
        var_m0 = np.var(m0_X)
        sample_r2 = var_m0 / var_D if var_D > 0 else 0.0
        
        # Reduced-form: ℓ₀(X) = E[Y|X]
        ell0_X = self.ell0(X)
        
        info = {
            'target_r2': self.target_r2,
            'sample_r2': sample_r2,
            'sigma_U': self.sigma_U,
            'm0_X': m0_X,
            'g0_X': g0_X,
            'ell0_X': ell0_X,
            'beta': self.beta.copy(),
        }
        
        return Y, D, X, info


def generate_data(
    n: int,
    target_r2: float,
    p: int = P_DEFAULT,
    rho: float = RHO_DEFAULT,
    theta0: float = THETA0,
    random_state: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray, Dict, DGP]:
    """
    Generate data from the PLR DGP (convenience function).
    
    This is the primary data generation function for Monte Carlo simulations.
    It creates a DGP instance, calibrates the overlap condition, and generates
    a sample.
    
    Parameters
    ----------
    n : int
        Sample size.
    target_r2 : float
        Target R²(D|X) controlling overlap strength:
        - 0.50 → κ ≈ 2 (strong overlap, well-conditioned)
        - 0.90 → κ ≈ 10 (moderate overlap)
        - 0.97 → κ ≈ 33 (weak overlap, ill-conditioned)
    p : int, default 10
        Covariate dimension.
    rho : float, default 0.5
        AR(1) correlation parameter.
    theta0 : float, default 1.0
        True treatment effect.
    random_state : int or None
        Random seed.
    
    Returns
    -------
    Y : ndarray of shape (n,)
        Outcome.
    D : ndarray of shape (n,)
        Treatment.
    X : ndarray of shape (n, p)
        Covariates.
    info : dict
        Calibration and diagnostic information.
    dgp : DGP
        The DGP instance (useful for Oracle learners).
    
    Examples
    --------
    >>> Y, D, X, info, dgp = generate_data(n=1000, target_r2=0.90)
    >>> print(f"κ ≈ {1/(1 - info['sample_r2']):.1f}")
    κ ≈ 10.0
    """
    dgp = DGP(
        p=p,
        rho=rho,
        theta0=theta0,
        target_r2=target_r2,
        random_state=random_state,
    )
    
    Y, D, X, info = dgp.generate(n, random_state=random_state)
    
    return Y, D, X, info, dgp



# =============================================================================
# NON-LINEAR DGP (Robustness Testing, Section 5.2)
# =============================================================================

@dataclass
class NonLinearDGP(DGP):
    """
    Non-Linear DGP variant for robustness testing.
    
    This DGP replaces the linear treatment equation with a non-linear
    specification to verify that the κ mechanism is not an artifact
    of linear propensity scores (Table 6 in the paper).
    
    Treatment Equation (Non-Linear)
    -------------------------------
    D = f(β'X) × scale + σ_U·U
    
    where f ∈ {tanh, sin, polynomial with interactions} introduces
    non-linearities that test whether:
    1. RF can learn m₀(X) better than in the linear case
    2. κ still predicts bias amplification per Theorem 3.8
    
    Parameters
    ----------
    nonlinearity : str, default 'tanh'
        Type of nonlinearity: 'tanh', 'sin', or 'interaction'
    
    Notes
    -----
    Results from Table 6 confirm the bias amplification mechanism
    persists regardless of propensity score functional form.
    """
    nonlinearity: str = 'tanh'
    _scale: float = field(default=1.0, init=False)
    
    def __post_init__(self) -> None:
        """Initialize with non-linear calibration."""
        # Geometrically decaying treatment coefficients
        self.beta = np.array([0.7 ** j for j in range(1, self.p + 1)])
        
        # AR(1) Toeplitz covariance matrix
        self._Sigma = make_toeplitz_cov(self.p, self.rho)
        
        # Estimate scale to match variance of linear specification
        rng = np.random.default_rng(42)
        X_cal = rng.multivariate_normal(np.zeros(self.p), self._Sigma, size=10000)
        linear_part = X_cal @ self.beta
        
        if self.nonlinearity == 'tanh':
            self._scale = np.std(linear_part) / np.std(np.tanh(linear_part))
        elif self.nonlinearity == 'sin':
            self._scale = np.std(linear_part) / np.std(np.sin(linear_part))
        else:
            self._scale = 1.0
        
        # Calibrate σ_U to achieve target R²(D|X)
        self.calibrate_noise(self.target_r2)
    
    def m0(self, X: NDArray) -> NDArray:
        """
        Non-linear propensity score: m₀(X) = f(β'X).
        
        The non-linear transformation tests robustness of the κ mechanism
        beyond the linear specification.
        """
        linear_part = X @ self.beta
        
        if self.nonlinearity == 'tanh':
            return np.tanh(linear_part) * self._scale
        elif self.nonlinearity == 'sin':
            return np.sin(linear_part) * self._scale
        elif self.nonlinearity == 'interaction':
            # Add interaction terms: X₁X₂ + X₃X₄
            interactions = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]
            return linear_part + 0.5 * interactions
        else:
            return linear_part


def generate_nonlinear_data(
    n: int,
    target_r2: float,
    nonlinearity: str = 'tanh',
    p: int = P_DEFAULT,
    rho: float = RHO_DEFAULT,
    theta0: float = THETA0,
    random_state: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray, Dict, NonLinearDGP]:
    """
    Generate data from the Non-Linear DGP (robustness testing).
    
    Used for Table 6 (Non-Linear DGP Robustness) to verify that κ
    amplification is not specific to linear propensity scores.
    
    Parameters
    ----------
    n : int
        Sample size.
    target_r2 : float
        Target R²(D|X).
    nonlinearity : str, default 'tanh'
        Non-linearity type: 'tanh', 'sin', or 'interaction'
    p, rho, theta0, random_state : 
        Same as generate_data.
    
    Returns
    -------
    Y, D, X, info, dgp : same as generate_data
        info additionally contains 'nonlinearity' key
    """
    dgp = NonLinearDGP(
        p=p,
        rho=rho,
        theta0=theta0,
        target_r2=target_r2,
        random_state=random_state,
        nonlinearity=nonlinearity,
    )
    
    Y, D, X, info = dgp.generate(n, random_state=random_state)
    info['nonlinearity'] = nonlinearity
    
    return Y, D, X, info, dgp


__all__ = [
    'DGP',
    'NonLinearDGP',
    'generate_data',
    'generate_nonlinear_data',
    'make_toeplitz_cov',
    'THETA0',
    'P_DEFAULT',
    'RHO_DEFAULT',
    'SIGMA_EPS',
]
