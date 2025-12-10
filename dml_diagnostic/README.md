# dml-diagnostic

**Condition Number Diagnostics for Double Machine Learning**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

`dml-diagnostic` provides the **DML condition number** κ_DML, a diagnostic measure for assessing the reliability of Double Machine Learning (DML) estimators. This diagnostic is analogous to the first-stage F-statistic in instrumental variables analysis: it quantifies how much residual treatment variation is available for identification.

**Key insight**: When treatment is highly predictable from covariates (poor overlap), the DML score becomes "flat" and estimation is fragile. The condition number κ_DML captures this directly.

### Theory

The DML condition number is defined as:

$$\kappa_{\text{DML}} = \frac{n}{\sum_{i=1}^n \hat{U}_i^2}$$

where $\hat{U}_i = D_i - \hat{m}(X_i)$ are the cross-fitted treatment residuals. Large κ_DML indicates:
- Weak overlap (treatment highly predictable from covariates)
- Inflated variance and potential bias amplification
- Fragile inference similar to weak-IV problems

**Reference**: Saco, G. (2025). "Finite-Sample Failures and Condition-Number Diagnostics in Double Machine Learning." *The Econometrics Journal*.

## Installation

```bash
pip install dml-diagnostic
```

For the latest development version:

```bash
pip install git+https://github.com/gsaco/dml-diagnostic.git
```

## Quick Start

```python
from dml_diagnostic import DMLDiagnostic, load_lalonde

# Load LaLonde (1986) experimental data
Y, D, X = load_lalonde(sample='experimental')

# Fit DML with condition number diagnostics
dml = DMLDiagnostic(learner='lasso')
result = dml.fit(Y, D, X)

print(result)
```

Output:
```
DML Diagnostic Results
──────────────────────
  θ̂ = 1793.42 (SE = 672.45)
  95% CI: [475.41, 3111.43]

  Condition Number: κ_DML = 4.10
  
  n = 445, R²(D|X) = -0.003, learner = lasso
```

## Comparing Experimental vs Observational Samples

The power of κ_DML is demonstrated by contrasting samples with different overlap:

```python
from dml_diagnostic import DMLDiagnostic, load_lalonde

# Experimental sample (randomised, good overlap)
Y_exp, D_exp, X_exp = load_lalonde('experimental')
result_exp = DMLDiagnostic(learner='lasso').fit(Y_exp, D_exp, X_exp)

# Observational sample (PSID controls, poor overlap)
Y_obs, D_obs, X_obs = load_lalonde('observational')
result_obs = DMLDiagnostic(learner='lasso').fit(Y_obs, D_obs, X_obs)

print(f"Experimental: θ̂ = {result_exp.theta:.0f}, κ = {result_exp.kappa:.2f}")
print(f"Observational: θ̂ = {result_obs.theta:.0f}, κ = {result_obs.kappa:.2f}")
```

```
Experimental: θ̂ = 1793, κ = 4.10
Observational: θ̂ = 56, κ = 15.71
```

The experimental benchmark of ~$1,800 is recovered when conditioning is good. The observational estimate is unreliable due to poor overlap (large κ).

## API Reference

### `DMLDiagnostic`

Main estimator class.

```python
DMLDiagnostic(
    learner='lasso',    # 'lin', 'lasso', 'ridge', 'rf', 'gbm'
    learner_m=None,     # Separate learner for E[D|X]
    learner_g=None,     # Separate learner for E[Y|X]
    n_folds=5,          # Cross-fitting folds
    random_state=42     # For reproducibility
)
```

**Methods:**
- `fit(Y, D, X)` → `DMLResult`: Fit DML and compute κ_DML
- `summary()` → `str`: Detailed results with interpretation

### `DMLResult`

Result container with attributes:
- `theta`: Point estimate θ̂
- `se`: Standard error
- `ci_lower`, `ci_upper`: 95% CI bounds
- `kappa`: Condition number κ_DML
- `jacobian`: Empirical Jacobian Ĵ_θ
- `r_squared_d`: R²(D|X)
- `U_hat`, `V_hat`: Cross-fitted residuals

### `load_lalonde`

Load LaLonde (1986) data.

```python
load_lalonde(
    sample='experimental',  # or 'observational'
    return_dataframe=False,
    verbose=False
)
```

### Diagnostic Functions

```python
# Compute κ from treatment residuals
compute_kappa(U_hat, n=None)

# Contextual interpretation
kappa_interpretation(kappa, n, r_squared_d=None)

# Overlap diagnostics via propensity scores
overlap_check(D, X, method='logistic')
```

## Plotting

```python
from dml_diagnostic import plot_kappa_summary, plot_overlap

# Summary plot with estimate and κ
plot_kappa_summary(result)

# Propensity score overlap diagnostic
plot_overlap(D, X)
```

## Dependencies

- numpy ≥ 1.20
- pandas ≥ 1.3
- scikit-learn ≥ 1.0
- matplotlib ≥ 3.4 (optional, for plotting)

## Citation

If you use this package in your research, please cite:

```bibtex
@article{saco2025dml,
  title={Finite-Sample Failures and Condition-Number Diagnostics in Double Machine Learning},
  author={Saco, Gabriel},
  journal={The Econometrics Journal},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
