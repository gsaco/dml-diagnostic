#!/usr/bin/env python3
"""
================================================================================
MASTER REPLICATION SCRIPT
================================================================================

Reproduces all figures and tables in:
    "Ill-Conditioned Orthogonal Scores in Double Machine Learning"
    by Gabriel Saco

This script fulfills the manuscript promise:
    "All tables and figures are generated from raw inputs using a single
    master script."

Usage:
    python run_all.py

Output:
    All figures and tables are written to results/

Random Seeds:
    All experiments use fixed seeds for reproducibility.
    - Simulation experiments: BASE_SEED = 42 (set in corrupted_oracle_analysis.py)
    - LaLonde application: SEED = 42 (set in lalonde_application.py)

================================================================================
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_ROOT = Path(__file__).parent.absolute()
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
RESULTS_DIR = REPO_ROOT / "results"

# Scripts to execute (in order)
EXPERIMENTS = [
    {
        "name": "Corrupted Oracle Analysis",
        "script": "corrupted_oracle_analysis.py",
        "outputs": [
            "figure1_bias_amplification.pdf",
            "figure2_coverage_analysis.pdf",
            "corrupted_oracle_aggregates.csv",
            "corrupted_oracle_results.csv",
        ],
        "description": "Monte Carlo simulations demonstrating κ-amplification of bias",
    },
    {
        "name": "LaLonde Application",
        "script": "lalonde_application.py",
        "outputs": [
            "lalonde_forest_plot.pdf",
            "lalonde_results.csv",
            "lalonde_baseline_results.csv",
        ],
        "description": "Empirical application to LaLonde (1986) job training data",
    },
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_header(text: str, char: str = "=") -> None:
    """Print a formatted header."""
    width = 70
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")


def run_experiment(experiment: dict) -> bool:
    """
    Run a single experiment script.
    
    Returns True if successful, False otherwise.
    """
    script_path = NOTEBOOKS_DIR / experiment["script"]
    
    print(f"📊 Running: {experiment['name']}")
    print(f"   Script: {experiment['script']}")
    print(f"   {experiment['description']}")
    print()
    
    start_time = time.time()
    
    # Set environment to use non-interactive matplotlib backend
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    
    try:
        # Stream output in real-time instead of capturing
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(NOTEBOOKS_DIR),
            env=env,
            check=True,
        )
        
        elapsed = time.time() - start_time
        print(f"   ✓ Completed in {elapsed:.1f}s")
        
        # Verify outputs exist
        missing = []
        for output in experiment["outputs"]:
            if not (RESULTS_DIR / output).exists():
                missing.append(output)
        
        if missing:
            print(f"   ⚠ Missing outputs: {missing}")
            return False
        else:
            print(f"   ✓ All {len(experiment['outputs'])} outputs generated")
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"   ✗ Failed after {elapsed:.1f}s")
        print(f"   Exit code: {e.returncode}")
        return False



def verify_outputs() -> None:
    """Print summary of all generated outputs."""
    print_header("OUTPUT SUMMARY", "-")
    
    print("Generated files in results/:\n")
    
    # Group by type
    pdfs = sorted(RESULTS_DIR.glob("*.pdf"))
    pngs = sorted(RESULTS_DIR.glob("*.png"))
    csvs = sorted(RESULTS_DIR.glob("*.csv"))
    
    if pdfs:
        print("  Figures (PDF):")
        for f in pdfs:
            print(f"    • {f.name}")
    
    if pngs:
        print("\n  Figures (PNG):")
        for f in pngs:
            print(f"    • {f.name}")
    
    if csvs:
        print("\n  Tables (CSV):")
        for f in csvs:
            print(f"    • {f.name}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> int:
    """
    Main entry point for replication.
    
    Returns exit code: 0 for success, 1 for failure.
    """
    print_header("DML DIAGNOSTIC REPLICATION")
    
    print("This script reproduces all figures and tables from:")
    print("  'Ill-Conditioned Orthogonal Scores in Double Machine Learning'")
    print("  by Gabriel Saco")
    print()
    print(f"Repository: {REPO_ROOT}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)
    
    total_start = time.time()
    successes = 0
    failures = 0
    
    # Run each experiment
    for i, experiment in enumerate(EXPERIMENTS, 1):
        print_header(f"EXPERIMENT {i}/{len(EXPERIMENTS)}", "-")
        
        if run_experiment(experiment):
            successes += 1
        else:
            failures += 1
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print_header("REPLICATION COMPLETE")
    
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"  Experiments: {successes} succeeded, {failures} failed")
    
    if failures == 0:
        verify_outputs()
        print("\n✓ All figures and tables have been reproduced successfully.\n")
        return 0
    else:
        print("\n✗ Some experiments failed. Check output above for details.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
