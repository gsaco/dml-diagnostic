"""
Hyperparameter Tuning for Random Forest Learners.

This module provides pre-tuning functionality for Random Forest learners
used in the LaLonde empirical application (Section 6).

For the Monte Carlo simulations (Section 5), fixed hyperparameters are used
to ensure reproducibility and reduce computational overhead. For the empirical
application, data-adaptive tuning is appropriate since the optimal parameters
may differ between experimental and observational samples.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from src.learners import RF_PARAM_GRID


def tune_rf_for_data(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    n_iter: int = 10,
    cv: int = 3,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Tune Random Forest hyperparameters for a specific dataset.
    
    This function is used for the LaLonde empirical application (Section 6)
    where hyperparameters are tuned separately for experimental and
    observational samples to achieve best performance on each.
    
    Parameters
    ----------
    X : ndarray of shape (n, p)
        Covariate matrix.
    y : ndarray of shape (n,)
        Target variable (treatment D or outcome Y).
    random_state : int, default 42
        Random seed for reproducibility.
    n_iter : int, default 10
        Number of parameter settings sampled in randomized search.
    cv : int, default 3
        Number of cross-validation folds.
    n_jobs : int, default -1
        Number of parallel jobs.
    
    Returns
    -------
    best_params : dict
        Dictionary containing the best hyperparameters:
        - max_depth: Maximum tree depth (None, 10, or 20)
        - min_samples_leaf: Minimum samples per leaf (1, 5, or 10)
        - max_features: Features per split ('sqrt' or None)
    
    Notes
    -----
    The tuned parameters are used with get_learner('RF_Tuned', params=best_params)
    to create a configured RandomForestRegressor.
    """
    base_rf = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=1,
    )
    
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=RF_PARAM_GRID,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=random_state,
        n_jobs=n_jobs,
    )
    
    search.fit(X, y)
    
    return search.best_params_


__all__ = [
    'tune_rf_for_data',
]
