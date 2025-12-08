# Finite-Sample Failures and Condition-Number Diagnostics in Double Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

> **Paper:** *Finite-Sample Failures and Condition-Number Diagnostics in Double Machine Learning*  
> **Author:** Gabriel Saco  
> Universidad del Pacífico, Lima, Peru  
> Email: ga.sacoa@up.edu.pe  
> ORCID: [0009-0009-8751-4154](https://orcid.org/0009-0009-8751-4154)

---

## Abstract

Standard Double Machine Learning (DML; Chernozhukov et al., 2018) confidence intervals can exhibit substantial finite-sample coverage distortions when the underlying score equations are ill-conditioned, even if nuisance functions are estimated with state-of-the-art methods. Focusing on the partially linear regression (PLR) model, we show that a simple, easily computed condition number for the orthogonal score ($\kappa_{\mathrm{DML}} := 1/|\hat{J}_\theta|$) largely determines when DML inference is reliable.

**Key contributions:**
1. **Berry–Esseen-type bound:** Coverage error of the DML $t$-statistic is of order $n^{-1/2} + \sqrt{n}\,r_n$
2. **Refined linearization:** Both estimation error and CI length scale as $\kappa_{\mathrm{DML}}/\sqrt{n} + \kappa_{\mathrm{DML}} r_n$
3. **Three conditioning regimes:** Well-conditioned, moderately ill-conditioned, and severely ill-conditioned
4. **Diagnostic proposal:** Report $\kappa_{\mathrm{DML}}$ alongside DML estimates (analogous to weak-instrument diagnostics)

---

## Repository Structure

```
dml-condition/
├── src/
│   └── dml_condition/
│       ├── __init__.py          # Package exports
│       └── core.py              # Core simulation module (1400+ lines)
├── notebooks/
│   ├── simulation.ipynb         # Main Monte Carlo study (low-dimensional)
│   └── high_dimensional_study.ipynb  # High-dimensional (p > n) simulations
├── results/                     # Generated tables, figures, and CSV files
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
```

---

## Theoretical Framework

### The DML Condition Number

For the Partially Linear Regression model:
$$Y = D \cdot \theta_0 + g_0(X) + \varepsilon, \quad D = m_0(X) + U$$

The DML condition number is defined as:
$$\kappa_{\mathrm{DML}} := \frac{1}{|\hat{J}_\theta|} = \frac{n}{\sum_i \hat{U}_i^2}$$

where $\hat{U}_i = D_i - \hat{m}(X_i)$ are the residualized treatments from cross-fitting.

### Key Linearization

The paper's main result shows:
$$\hat{\theta} - \theta_0 = \kappa_{\mathrm{DML}} \cdot (S_n + B_n) + R_n$$

This implies the condition number directly amplifies both variance ($S_n$) and nuisance bias ($B_n$):
$$|\hat{\theta} - \theta_0| = O_P\left(\frac{\kappa_{\mathrm{DML}}}{\sqrt{n}} + \kappa_{\mathrm{DML}} \cdot r_n\right)$$

### Three Conditioning Regimes

| Regime | Condition | CI Length | Interpretation |
|--------|-----------|-----------|----------------|
| Well-conditioned | $\kappa_n = O_P(1)$ | $O_P(n^{-1/2})$ | Standard inference reliable |
| Moderately ill-conditioned | $\kappa_n = O_P(n^\beta)$, $0 < \beta < \tfrac{1}{2}$ | $O_P(n^{\beta-1/2})$ | Slower convergence |
| Severely ill-conditioned | $\kappa_n \asymp c\sqrt{n}$ | $O_P(1)$ | CI fails to shrink |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/gsaco/dml-condition.git
cd dml-condition

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Simulation Design

### Data-Generating Process

The PLR model (Robinson, 1988; Chernozhukov et al., 2018):

$$Y = D \cdot \theta_0 + g_0(X) + \varepsilon, \quad \varepsilon \sim N(0, 1)$$
$$D = X^\top \gamma + \xi, \quad \xi \sim N(0, \sigma_\xi^2)$$
$$X \sim N(0, \Sigma(\rho)), \quad \Sigma_{jk} = \rho^{|j-k|}$$

where:
- $\theta_0 = 1$ is the true treatment effect
- $g_0(X) = 0.5 X_1^2 + 0.5 \sin(X_2) + 0.3 X_3 X_4$ (nonlinear nuisance)
- $\gamma_j = 0.7^{j-1}$ (geometric decay)
- $\rho = 0.5$ (Toeplitz correlation)

### Overlap Calibration via $R^2(D|X)$

We calibrate $\sigma_\xi^2$ to achieve target $R^2(D|X)$ values:

| Overlap Level | Target $R^2(D|X)$ | Interpretation |
|---------------|-------------------|----------------|
| High          | 0.75              | Good overlap: substantial residual variation |
| Moderate      | 0.90              | Limited residual variation |
| Low           | 0.97              | Poor overlap: $D$ nearly deterministic given $X$ |

### Nuisance Learners

Three learner specifications are compared:

| Label | Method | Description |
|-------|--------|-------------|
| **LIN** | RidgeCV | Ridge regression with CV-tuned regularization |
| **LAS** | LassoCV | Lasso with 5-fold CV |
| **RF** | Random Forest | 200 trees, max_depth=5, min_samples_leaf=10 |

### Design Grid

| Dimension | Values |
|-----------|--------|
| Sample sizes ($n$) | 500, 2000 |
| $R^2(D|X)$ levels | 0.75, 0.90, 0.97 |
| Learners | LIN, LAS, RF |
| Replications | 500 per cell |

**Total:** 18 design cells × 500 replications = 9,000 Monte Carlo draws

---

## Reproducing the Results

### Low-Dimensional Study ($p = 10$)

```bash
jupyter notebook notebooks/simulation.ipynb
```

Run all cells to execute the Monte Carlo simulation.

### High-Dimensional Study ($p > n$)

```bash
jupyter notebook notebooks/high_dimensional_study.ipynb
```

Tests DML performance when $p = 200$ and $n \in \{100, 500, 2000\}$.

### Quick Test

For rapid verification (reduced precision):

```python
from dml_condition import run_full_study

# Quick test with fewer replications
results_df, cell_summary, table1, table2 = run_full_study(
    n_list=[500],
    R2_list=[0.75, 0.97],
    learners=["LIN", "RF"],
    B=50,  # Reduced from 500
    verbose=True,
)
```

**Full reproduction:** ~15–30 minutes with `B = 500` on a modern laptop.

---

## Key Empirical Results

### Coverage by Conditioning Regime

| $\kappa_{\mathrm{DML}}$ Range | Coverage | Interpretation |
|-------------------------------|----------|----------------|
| $< 1$ (well-conditioned) | 93–95% | Near-nominal |
| $1$–$2$ (moderate) | 85–93% | Some degradation |
| $> 2$ (severe) | **40–70%** | Substantial undercoverage |

### The Paradox of Larger Samples

A striking finding: increasing $n$ can **worsen** coverage in ill-conditioned designs:

- Larger $n$ shrinks standard errors
- But bias (amplified by $\kappa_{\mathrm{DML}}$) persists
- Result: worse coverage despite more data

### Diagnostic Correlations

- Correlation($\kappa_{\mathrm{DML}}$, Coverage): $r \approx -0.77$
- Correlation($\kappa_{\mathrm{DML}}$, RMSE): $r \approx 0.88$

---

## Output Files

After running the notebooks, the following files are generated in `results/`:

| File | Description |
|------|-------------|
| `simulation_results.csv` | Raw replication-level results |
| `cell_summary.csv` | Summary statistics by design cell |
| `table1_design_summary.csv/tex` | Table 1: Design summary |
| `table2_coverage_by_regime.csv/tex` | Table 2: Coverage by $\kappa$-regime |
| `high_dim_simulation_results.csv` | High-dimensional study results |
| `high_dim_cell_summary.csv` | High-dimensional summary |
| `high_dim_coverage_vs_kappa.png` | Coverage vs $\kappa$ (high-dim) |
| `high_dim_kappa_distribution.png` | $\kappa$ distribution (high-dim) |

---

## API Reference

### Core Functions

```python
from dml_condition import (
    # Data generation
    generate_plr_data,      # Generate PLR data with calibrated overlap
    g0_function,            # Nonlinear nuisance function
    calibrate_sigma_xi_sq,  # Calibrate σ_ξ² for target R²(D|X)
    
    # DML estimation
    run_dml_plr,            # Cross-fitted DML estimator
    get_nuisance_model,     # Factory for LIN/LAS/RF learners
    
    # Monte Carlo
    run_simulation,         # Full simulation across design grid
    run_full_study,         # Complete study with tables and figures
    
    # Summary and visualization
    compute_cell_summary,   # Aggregate replication results
    plot_coverage_vs_kappa, # Main diagnostic figure
)
```

### Example Usage

```python
import numpy as np
from dml_condition import generate_plr_data, run_dml_plr

# Generate data with moderate overlap
Y, D, X, info = generate_plr_data(
    n=1000,
    R2_target=0.90,
    random_state=42
)

# Run DML estimation
result = run_dml_plr(Y, D, X, learner_label="RF")

print(f"θ̂ = {result.theta_hat:.3f}")
print(f"κ_DML = {result.kappa_dml:.2f}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
print(f"SE = {result.se_dml:.4f}")
```

---

## Practical Recommendations

We recommend reporting $\kappa_{\mathrm{DML}}$ alongside DML estimates, analogous to first-stage $F$-statistics in IV regression:

| $\kappa_{\mathrm{DML}}$ | Interpretation | Recommendation |
|-------------------------|----------------|----------------|
| $< 1$ | Well-conditioned | Standard inference reliable |
| $1$–$2$ | Moderately ill-conditioned | Exercise caution; robustness checks advised |
| $\geq 2$ | Severely ill-conditioned | CIs may be substantially distorted |

**Required conditions for informative inference:**
- $\kappa_{\mathrm{DML}} = o_p(\sqrt{n})$ — condition number grows slower than $\sqrt{n}$
- $\kappa_{\mathrm{DML}} \cdot r_n \to 0$ — bias term vanishes

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.21.0 | Array operations |
| `scipy` | ≥1.7.0 | Statistical functions |
| `pandas` | ≥1.3.0 | Data manipulation |
| `scikit-learn` | ≥1.0.0 | ML models (Lasso, RF) |
| `matplotlib` | ≥3.4.0 | Visualization |
| `seaborn` | ≥0.11.0 | Statistical plots |
| `jupyter` | ≥1.0.0 | Notebook interface |
| `tqdm` | ≥4.62.0 | Progress bars |

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{saco2025dml_conditioning,
  title     = {Finite-Sample Failures and Condition-Number Diagnostics 
               in Double Machine Learning},
  author    = {Saco, Gabriel},
  journal   = {Working Paper},
  year      = {2025},
  institution = {Universidad del Pac{\'i}fico},
  address   = {Lima, Peru},
  note      = {ORCID: 0009-0009-8751-4154}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.

- Robinson, P. M. (1988). Root-N-consistent semiparametric regression. *Econometrica*, 56(4), 931–954.

- Chernozhukov, V., Newey, W. K., & Singh, R. (2023). A simple and general debiased machine learning theorem with finite-sample guarantees. *Biometrika*, 110(1), 257–264.

- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML: An object-oriented implementation of double machine learning in Python. *Journal of Machine Learning Research*, 23(53), 1–6.

---

## Contact

For questions or feedback, please open an issue on GitHub or contact the author at ga.sacoa@up.edu.pe.