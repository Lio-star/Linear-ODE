
# Linear-ODE

A modular and reproducible framework for running, comparing, and visualizing Linear ODE learning methods under a shared experimental pipeline.

This repository is designed to keep the workflow clean and extensible. Model implementations are separated from experiment orchestration and utility functions, making the project easier to read, maintain, and expand. The code supports side-by-side benchmarking of multiple methods, inline visualization in Jupyter notebooks, and formatted summary tables for reporting final results.

---

## Overview

Linear-ODE provides a structured environment for:

- generating trajectory data under controlled settings
- running multiple Linear ODE solvers with shared experiment configurations
- comparing runtime and optimization performance across methods
- visualizing loss curves and trajectories
- summarizing final experiment results in a clean tabular format

The current framework includes two primary methods:

- **Autodiff**
- **OurModel**

---

## Repository Structure

```text
Linear-ODE/
├── configs/
│   └── default_config.py
├── solvers/
│   ├── __init__.py
│   ├── autodiff.py
│   └── our_model.py
├── utils/
│   ├── __init__.py
│   ├── data_generation.py
│   ├── frechet.py
│   ├── plotting.py
│   └── summary.py
├── example.ipynb
├── main_code_experiment.py
├── model_runner.py
├── original_notebook.ipynb
├── requirements.txt
├── README.md
└── .gitignore

git clone https://github.com/Lio-star/Linear-ODE.git
cd Linear-ODE
