# Finite-Sample Conditioning in Double Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

> **"Finite-Sample Conditioning in Double Machine Learning: A Short Communication"**
>
> Gabriel Saco  
> Universidad del Pacífico

---

## Abstract

Double Machine Learning (DML) provides asymptotically valid inference for low-dimensional parameters in high-dimensional settings. However, finite-sample performance can deteriorate when the empirical Jacobian is nearly singular—a situation arising from poor overlap or strong collinearity between treatment and covariates. This repository accompanies a short communication that:

1. **Introduces** a simple, interpretable **condition number** $\kappa_{\mathrm{DML}} := 1/|\hat{J}_\theta|$ for the Partially Linear Regression (PLR) model
2. **Establishes** a coverage error bound of order $n^{-1/2} + \sqrt{n}\,r_n + o(1)$
3. **Derives** a $\kappa$-amplified linearization showing parameter-scale error grows as $\kappa_{\mathrm{DML}}/\sqrt{n} + \kappa_{\mathrm{DML}}\,r_n$
4. **Characterizes** three conditioning regimes (well-conditioned, moderately ill-conditioned, severely ill-conditioned)
5. **Provides** Monte Carlo evidence validating $\kappa_{\mathrm{DML}}$ as a practical diagnostic
6. **Offers** practical recommendations for applied researchers

---

## Repository Structure

```
dml-condition/
├── src/dml_condition/          # Core simulation code
│   ├── __init__.py
│   └── core.py                 # DGP and DML estimator
├── notebooks/
│   └── paper_simulations.ipynb # Monte Carlo simulation
├── results/                    # Output tables
├── paper/                      # LaTeX source
├── requirements.txt
└── README.md
```

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

## Reproducing the Results

Run the Monte Carlo simulation:

```bash
jupyter notebook notebooks/paper_simulations.ipynb
```

Results are saved to `results/table1_kappa_coverage.csv`.
3. Run all cells

**Note:** Full reproduction takes approximately 10–20 minutes with `N_REPS = 500`.

For quick testing, set `N_REPS = 50` (results will be less precise but faster).

---

## Simulation Design

### Data-Generating Process

We use the canonical Partially Linear Regression (PLR) model:

$$Y = D \cdot \theta_0 + g_0(X) + \varepsilon, \quad \varepsilon \sim N(0, 1)$$

$$D = X^\top \beta_D + U, \quad U \sim N(0, \sigma_U^2)$$

$$X \sim N(0, \Sigma(\rho)), \quad \Sigma_{jk} = \rho^{|j-k|}$$

where:
- $\theta_0 = 1$ is the true treatment effect
- $g_0(X) = \gamma^\top \sin(X)$ is a nonlinear nuisance function
- $\beta_D = (1, 0.8, 0.6, 0.4, 0.2, 0, \ldots, 0)^\top$ has decaying coefficients
- $\gamma = (1, 0.5, 0.25, 0.125, 0.0625, 0, \ldots, 0)^\top$

### Overlap Calibration

We calibrate $\sigma_U^2$ to achieve target $R^2(D|X)$ values:

| Overlap Level | Target $R^2(D|X)$ | Interpretation |
|---------------|-------------------|----------------|
| High          | 0.75              | Good overlap: substantial residual variation in $D$ |
| Moderate      | 0.90              | Limited residual variation |
| Low           | 0.97              | Poor overlap: $D$ nearly deterministic given $X$ |

### DGP Configurations

We consider 9 DGP configurations spanning three conditioning regimes:

| Group | DGPs | Overlap | Expected $\kappa_{\mathrm{DML}}$ |
|-------|------|---------|----------------------------------|
| A     | A1, A2, A3 | High | < 1 (well-conditioned) |
| B     | B1, B2, B3 | Moderate | 1–2 (moderately ill-conditioned) |
| C     | C1, C2, C3 | Low | > 2 (severely ill-conditioned) |

---

## Key Results

### Main Findings

| Regime | $\kappa_{\mathrm{DML}}$ Range | Coverage | Interpretation |
|--------|-------------------------------|----------|----------------|
| Well-conditioned | 0.5–0.9 | 92.6%–94.0% | Near-nominal (95%) |
| Moderately ill-conditioned | 1.2–1.6 | 85.2%–93.6% | Degradation at large $n$ |
| Severely ill-conditioned | 2.1–2.7 | **59.2%–92.4%** | Dramatic undercoverage |

### The Paradox of Larger Samples

A striking finding: increasing $n$ from 500 to 2000 can **worsen** coverage in ill-conditioned designs. For example, comparing C1 ($n=500$) to C2 ($n=2000$):
- Coverage drops from 92.4% to 62.0%
- This occurs because larger $n$ shrinks the standard error, but bias (amplified by $\kappa_{\mathrm{DML}}$) remains comparable

### Diagnostic Value

- Correlation between $\kappa_{\mathrm{DML}}$ and coverage: $r = -0.77$
- Correlation between $\kappa_{\mathrm{DML}}$ and RMSE: $r = 0.88$

---

## Practical Recommendations

We recommend reporting $\kappa_{\mathrm{DML}}$ alongside DML estimates, analogous to first-stage $F$-statistics in IV regression:

| $\kappa_{\mathrm{DML}}$ | Interpretation | Recommendation |
|-------------------------|----------------|----------------|
| < 1 | Well-conditioned | Standard inference reliable |
| 1–2 | Moderately ill-conditioned | Exercise caution; robustness checks advised |
| ≥ 2 | Severely ill-conditioned | CIs may be substantially distorted |

---

## Output Files

After running the notebook, the following files are generated in `results/`:

| File | Description |
|------|-------------|
| `simulation_results_full.csv` | 4,500 rows: one per replication (9 DGPs × 500 reps) |
| `simulation_summary.csv` | 9 rows: summary statistics by DGP |
| `coverage_vs_kappa.png/pdf` | **Main figure:** Coverage vs. $\kappa_{\mathrm{DML}}$ |
| `ci_length_vs_kappa.png/pdf` | CI length vs. $\kappa_{\mathrm{DML}}$ |

---

## Compiling the Paper

```bash
cd paper
pdflatex main.tex
pdflatex main.tex  # Run twice for references
```

---

## Dependencies

See `requirements.txt` for the complete list. Key dependencies:

- `numpy >= 1.21.0` — Numerical computing
- `pandas >= 1.3.0` — Data manipulation
- `scikit-learn >= 1.0.0` — Random Forest, Lasso, cross-validation
- `matplotlib >= 3.4.0` — Visualization
- `jupyter >= 1.0.0` — Notebook interface

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{saco2024dml_conditioning,
  title   = {Finite-Sample Conditioning in Double Machine Learning: A Short Communication},
  author  = {Saco, Gabriel},
  journal = {Working Paper},
  year    = {2024},
  note    = {Universidad del Pacífico}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Robinson, P. M. (1988). Root-N-consistent semiparametric regression. *Econometrica*, 56(4), 931–954.
- Bach, P., Chernozhukov, V., Kurz, M. S., & Spindler, M. (2022). DoubleML: An object-oriented implementation of double machine learning in Python. *Journal of Machine Learning Research*, 23(53), 1–6.
- Chernozhukov, V., Newey, W. K., & Singh, R. (2023). A simple and general debiased machine learning theorem with finite-sample guarantees. *Biometrika*, 110(1), 257–264.

---

## Contact

For questions or feedback, please open an issue on GitHub or contact the author.
---

## Contact

For questions or feedback, please open an issue on GitHub or contact the author.
