# Ill-Conditioned Orthogonal Scores in Double Machine Learning: Replication Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

This repository contains the replication materials for the paper **"Ill-Conditioned Orthogonal Scores in Double Machine Learning"**. It provides the source code and data necessary to reproduce all simulation results (Section 5) and empirical applications (Section 6) presented in the manuscript.

## Replication Capabilities

This package is designed for transparency and ease of replication:

| Feature | Status | Implementation |
|---------|--------|----------------|
| **One-Click Reproduction** | Yes | `run_all.py` master script |
| **Deterministic Execution** | Yes | Fixed random seeds (e.g., `BASE_SEED=42`) |
| **Dependency Management** | Yes | `requirements.txt` for python environment |
| **Data Access** | Yes | Automatic retrieval of LaLonde data from NBER |

---

## Quick Start

To reproduce all figures and tables from the paper with a single command:

```bash
# 1. Clone the repository
git clone https://github.com/gsaco/dml-diagnostic.git
cd dml-diagnostic

# 2. Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Generate all results
python run_all.py
```

**Expected Runtime:** ~10-20 minutes on standard hardware.

---

## Results Mapping

The master script `run_all.py` executes the following experiments, which correspond directly to the paper's elements:

| Paper Element | Description | Simulation Script | Output Filename |
|---------------|-------------|-------------------|-----------------|
| **Figure 1** | Bias Amplification Mechanism | `corrupted_oracle_analysis.py` | `figure1_bias_amplification.pdf` |
| **Figure 2** | Coverage Degradation | `corrupted_oracle_analysis.py` | `figure2_coverage_analysis.pdf` |
| **Tables S1-S6** | Simulation Aggregates (Supplement) | `corrupted_oracle_analysis.py` | `corrupted_oracle_aggregates.csv` |
| **Figure 3** | LaLonde Forest Plot | `lalonde_application.py` | `lalonde_forest_plot.pdf` |
| **Table 1** | LaLonde Estimates | `lalonde_application.py` | `lalonde_baseline_results.csv` |

All outputs are saved to the `results/` directory.

---

## Repository Structure

```
dml-diagnostic/
├── run_all.py               # Master replication script
├── requirements.txt         # Python dependencies
├── notebooks/               # Experiment source code
│   ├── corrupted_oracle_analysis.py   # Main simulation study (Section 5)
│   └── lalonde_application.py         # Empirical application (Section 6)
├── src/                     # Core computational modules
│   ├── dml.py               # Double Machine Learning estimator
│   ├── learners.py          # Machine learning models (RF, Lasso, etc.)
│   ├── dgp.py               # Data generating processes
│   ├── data.py              # LaLonde data loader
│   └── tuning.py            # Hyperparameter tuning logic
└── results/                 # Directory for generated artifacts
```

---

## Detailed Usage

### Running Individual Experiments
Each script in `notebooks/` is a self-contained Jupytext file that can be executed as a standard Python script or opened as a Jupyter notebook.

```bash
# Run simulation study only
python notebooks/corrupted_oracle_analysis.py

# Run empirical application only
python notebooks/lalonde_application.py
```

### Reproducibility Notes
- **Random Seeds**: All stochastic processes are controlled via global seeds initialized in `run_all.py` and individual scripts.
- **Hardware**: Tested on macOS (M-series) and Linux. Runtime may vary by core count.

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{saco2025illconditioned,
  title={Ill-Conditioned Orthogonal Scores in Double Machine Learning},
  author={Saco, Gabriel},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
