
# Linear-ODE

A structured and modular implementation for running and comparing Linear ODE learning methods in a clean workflow.

This repository is designed to make experiments easier to run, compare, and present. It supports multiple models under one pipeline, generates plots, and displays a formatted summary table inside the notebook.

---

## Features

- Structured project layout
- Multiple solver support under one interface
- Easy experiment control from a single notebook
- Inline plots for loss curves and trajectories
- Formatted summary table for final results
- Clean separation of core logic, solvers, utilities, and outputs

---

## Project Structure

```text
Linear-ODE/
├── configs/
│   └── default_config.py          # Default experiment settings
├── outputs/
│   ├── figures/                   # Saved plots (if saving is enabled)
│   └── tables/                    # Saved tables (if saving is enabled)
├── solvers/
│   ├── __init__.py
│   ├── autodiff.py                # Autodiff-based Linear ODE solver
│   └── our_model.py               # Proposed custom model
├── utils/
│   ├── __init__.py
│   ├── data_generation.py         # Synthetic data generation and mask creation
│   ├── frechet.py                 # Frechet / matrix exponential utilities
│   ├── plotting.py                # Plotting functions
│   └── summary.py                 # Summary table creation and styling
├── example.ipynb                  # Main notebook for running experiments
├── main_code_experiment.py        # Main experiment pipeline
├── model_runner.py                # Model registry and execution helpers
├── original_notebook.ipynb        # Older notebook version
├── requirements.txt               # Python dependencies
├── README.md
└── .gitignore
