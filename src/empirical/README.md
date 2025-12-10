# Empirical Application: DML Condition Number Diagnostic

This directory contains a complete Python implementation of the DML condition number diagnostic ($\kappa_{\text{DML}}$) applied to the LaLonde/NSW job training programme data.

## Contents

```
empirical/
├── __init__.py                    # Package initialisation
├── data_lalonde.py                # Data loading and preprocessing
├── dml_kappa.py                   # DML estimator with κ_DML diagnostic
├── utils_tables.py                # Utility functions for tables and plots
├── empirical_application.ipynb    # Main analysis notebook
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Quick Start

### Installation

```bash
cd src/empirical
pip install -r requirements.txt
```

### Running the Analysis

```bash
jupyter notebook empirical_application.ipynb
```

Or with JupyterLab:
```bash
jupyter lab empirical_application.ipynb
```

## Module Descriptions

### `data_lalonde.py`

Functions for loading and preprocessing the LaLonde/NSW data:

- `load_lalonde_data(include_psid=True, include_cps=False, verbose=True)`: Load combined LaLonde/NSW dataset and return a single pd.DataFrame
- `get_experimental_sample(df)`: Extract the randomised experimental subsample (NSW treated + NSW controls) from the combined DataFrame
- `get_observational_sample(df)`: Extract the observational subsample (NSW treated + non-experimental PSID/CPS controls) from the combined DataFrame
- `get_covariate_matrix(df, covariates)`: Extract covariate matrix
- `summary_statistics(df, columns)`: Compute summary statistics by treatment status

### `dml_kappa.py`

The main DML estimator with $\kappa_{\text{DML}}$ diagnostic:

```python
from empirical.dml_kappa import PLRDoubleMLKappa, get_learner

# Get learners for nuisance functions
ml_l = get_learner('LASSO')  # E[Y|X]
ml_m = get_learner('LASSO')  # E[D|X]

# Create and fit DML model
dml = PLRDoubleMLKappa(
    ml_l=ml_l,
    ml_m=ml_m,
    n_folds=5,
    n_rep=1,
    random_state=42
)
dml.fit(Y, D, X)

# Get results
results = dml.summary_dict()
print(f"θ̂ = {results['theta']:.2f} (SE = {results['se']:.2f})")
print(f"κ_DML = {results['kappa_dml']:.2f}")
```

Available learners:
- `'linear'`: Linear regression (OLS)
- `'lasso'`: L1-regularised regression with cross-validation
- `'ridge'`: L2-regularised regression with cross-validation
- `'rf'`: Random Forest regressor

### `utils_tables.py`

Utility functions for results presentation:

- `results_to_dataframe(results_list)`: Convert results to pandas DataFrame
- `results_to_latex(df, caption, label)`: Generate LaTeX table
- `plot_propensity_histogram(ps, D, ax)`: Plot propensity score overlap
- `plot_kappa_by_design(df, ax)`: Bar chart of κ_DML by design
- `plot_ci_length_vs_kappa(df, ax)`: Scatter plot of CI length vs κ_DML

## Key Concepts

### The DML Condition Number

The condition number $\kappa_{\text{DML}}$ measures the stability of the DML estimator:

$$\kappa_{\text{DML}} = \frac{n}{\sum_{i=1}^n \hat{U}_i^2}$$

where $\hat{U}_i = D_i - \hat{m}(X_i)$ is the residual treatment variation after partialling out covariates.

**Interpretation**:
- $\kappa_{\text{DML}} \approx 1$: Ideal conditioning (rarely achievable)
- $\kappa_{\text{DML}} \approx 4$: Good conditioning (typical of randomised experiments)
- $\kappa_{\text{DML}} > 10$: Poor conditioning (warrants scrutiny)
- $\kappa_{\text{DML}} > 100$: Severe ill-conditioning (results may be unreliable)

### Connection to Standard Errors

The variance of the DML estimator scales with $\kappa_{\text{DML}}$:

$$\text{se}(\hat{\theta}) \approx \frac{\sigma_\varepsilon}{\sqrt{n}} \times \sqrt{\kappa_{\text{DML}}}$$

This means that doubling $\kappa_{\text{DML}}$ inflates the standard error by a factor of $\sqrt{2} \approx 1.41$.

## Data

The analysis uses the LaLonde/NSW job training programme data:

- **Experimental sample**: NSW participants + randomised control group
- **Observational sample**: NSW participants + CPS or PSID comparison group

Key variables:
- `treat`: Treatment indicator (job training programme)
- `re78`: Post-programme earnings (1978)
- `age`, `education`, `married`, `nodegree`, `re74`, `re75`: Covariates

## Output

Running the notebook produces:

1. `results/empirical_dml_results.csv`: Full results table
2. `results/empirical_results.tex`: LaTeX table for paper
3. `results/propensity_overlap.pdf`: Propensity score histograms
4. `results/kappa_by_design.pdf`: Bar chart of κ_DML
5. `results/ci_length_vs_kappa.pdf`: CI length vs κ_DML scatter
6. `results/trimming_analysis.pdf`: Effect of propensity score trimming
7. `results/forest_plot.pdf`: Forest plot of all estimates

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). "Double/debiased machine learning for treatment and structural parameters." *The Econometrics Journal*, 21(1), C1-C68.

- LaLonde, R.J. (1986). "Evaluating the econometric evaluations of training programs with experimental data." *American Economic Review*, 76(4), 604-620.

- Dehejia, R.H. and Wahba, S. (1999). "Causal effects in nonexperimental studies: Reevaluating the evaluation of training programs." *Journal of the American Statistical Association*, 94(448), 1053-1062.
